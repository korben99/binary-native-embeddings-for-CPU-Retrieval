"""
Train float or native-binary embedding model on NLI triplets.

Usage:
  python train.py --mode float             # baseline float (384-dim)
  python train.py --mode binary            # native binary  (4096-dim)
  python train.py --mode float --max_samples 5000   # smoke test
"""
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_from_disk, load_dataset

from models.float_embedder import FloatEmbedder, mnrl_loss
from models.binary_embedder import BinaryEmbedder, binary_contrastive_loss, entropy_loss, decorr_loss

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_cache"
CKPT_DIR = BASE_DIR / "checkpoints"


def get_device(prefer_mps=True):
    if prefer_mps and torch.backends.mps.is_available():
        print("Device: Apple MPS (Metal Performance Shaders)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Device: CUDA")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


class NLIPairDataset(Dataset):
    def __init__(self, ds):
        self.anchors = ds["anchor"]
        self.positives = ds["positive"]

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx]


def tokenize(texts, tokenizer, device):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)


def train(mode, epochs=3, batch_size=64, lr=2e-5, max_samples=None, no_mps=False,
          binary_dim=4096, tag="", temperature=0.05, lambda_e=0.0, lambda_d=0.0, seed=42):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(prefer_mps=not no_mps)

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")

    if mode == "float":
        model = FloatEmbedder(output_dim=384).to(device)
        ckpt_name = "float_embedder.pt"
    else:
        model = BinaryEmbedder(binary_dim=binary_dim).to(device)
        parts = [tag] if tag else []
        if seed != 42:
            parts.append(f"s{seed}")
        suffix = "_".join(parts)
        ckpt_name = f"binary_embedder_{binary_dim}{'_' + suffix if suffix else ''}.pt"

    # Load NLI dataset
    cache = DATA_DIR / "nli_train"
    if cache.exists():
        ds = load_from_disk(str(cache))
    else:
        print("NLI cache not found, downloading...")
        ds = load_dataset("sentence-transformers/all-nli", "triplet", split="train")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    loader = DataLoader(
        NLIPairDataset(ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    if mode == "binary":
        # Projection is randomly init — needs much higher LR than pretrained encoder
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": lr},
            {"params": model.projection.parameters(), "lr": lr * 50},
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * epochs
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reg_info = f" | λe={lambda_e} λd={lambda_d} T={temperature}" if mode == "binary" else ""
    print(f"\nMode: {mode} | Params: {n_params/1e6:.1f}M | "
          f"Samples: {len(ds):,} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}{reg_info}")
    print(f"Checkpoint: {ckpt_name}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (anchors, positives) in enumerate(loader):
            a_enc = tokenize(anchors, tokenizer, device)
            p_enc = tokenize(positives, tokenizer, device)

            if mode == "float":
                a = model(a_enc["input_ids"], a_enc["attention_mask"])
                p = model(p_enc["input_ids"], p_enc["attention_mask"])
                loss = mnrl_loss(a, p)
            else:
                a_logits = model(a_enc["input_ids"], a_enc["attention_mask"], binarize_output=False)
                p_logits = model(p_enc["input_ids"], p_enc["attention_mask"], binarize_output=False)
                loss = binary_contrastive_loss(a_logits, p_logits, temperature=temperature)
                if lambda_e > 0 or lambda_d > 0:
                    a_tanh = torch.tanh(a_logits)
                    p_tanh = torch.tanh(p_logits)
                    if lambda_e > 0:
                        loss = loss + lambda_e * (entropy_loss(a_tanh) + entropy_loss(p_tanh)) / 2
                    if lambda_d > 0:
                        loss = loss + lambda_d * (decorr_loss(a_tanh) + decorr_loss(p_tanh)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / max(step + 1, 1) * (len(loader) - step - 1)
                pct = (step + 1) / len(loader) * 100
                print(
                    f"  [{epoch+1}/{epochs}] step={step:>5}/{len(loader)} ({pct:.1f}%)"
                    f"  loss={loss.item():.4f}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
                )

        avg = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1} done — avg_loss={avg:.4f}  ({time.time()-t0:.0f}s)")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / ckpt_name
    torch.save(model.state_dict(), str(ckpt_path))
    print(f"\nCheckpoint -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["float", "binary"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training samples (smoke test)")
    parser.add_argument("--no_mps", action="store_true",
                        help="Force CPU even on Apple Silicon")
    parser.add_argument("--binary_dim", type=int, default=4096,
                        help="Binary embedding dimension (e.g. 1024, 2048, 4096)")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix appended to checkpoint name, e.g. 'bs256' → binary_embedder_2048_bs256.pt")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Contrastive loss temperature (default 0.05)")
    parser.add_argument("--lambda_e", type=float, default=0.0,
                        help="Entropy regularization weight (0=disabled, try 0.1)")
    parser.add_argument("--lambda_d", type=float, default=0.0,
                        help="Decorrelation regularization weight (0=disabled, try 0.01)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. If != 42, auto-appended to checkpoint name (e.g. _s123)")
    args = parser.parse_args()
    train(args.mode, args.epochs, args.batch_size, args.lr, args.max_samples, args.no_mps,
          args.binary_dim, args.tag, args.temperature, args.lambda_e, args.lambda_d, args.seed)
