import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .ste import binarize


class BinaryEmbedder(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-mini", binary_dim=4096):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 256 for bert-mini
        self.projection = nn.Sequential(
            nn.Linear(hidden, binary_dim),
            nn.LayerNorm(binary_dim),
        )
        self.binary_dim = binary_dim

    def _mean_pool(self, token_embs, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, binarize_output=True):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self._mean_pool(out, attention_mask)
        projected = self.projection(pooled)          # pre-binarization logits
        if binarize_output:
            return binarize(projected)               # {-1,+1}^D
        return projected                             # float logits, for loss computation

    def encode(self, texts, tokenizer, device="cpu", batch_size=64):
        self.eval()
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                embs = self.forward(enc["input_ids"], enc["attention_mask"], binarize_output=True)
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)


def entropy_loss(tanh_acts):
    """Force each bit to be balanced (~50% +1). Operates on tanh logits ∈ (-1,+1)."""
    p = ((tanh_acts + 1) / 2).mean(dim=0)   # soft [0,1] probability per bit, differentiable
    return ((p - 0.5) ** 2).mean()


def decorr_loss(tanh_acts):
    """Penalize inter-bit correlation (Barlow Twins style). O(batch × D²) memory."""
    z = (tanh_acts - tanh_acts.mean(0)) / (tanh_acts.std(0) + 1e-6)
    c = (z.T @ z) / z.shape[0]              # (D, D) correlation matrix
    off_diag = c - torch.diag(torch.diag(c))
    return (off_diag ** 2).sum() / z.shape[1]


def binary_contrastive_loss(anchors_logits, positives_logits, temperature=0.05):
    """
    Loss on pre-binarization logits via tanh + cosine.
    tanh maps to (-1,+1), aligned with the {-1,+1} STE output so the training
    signal directly optimizes what the eval metric measures.
    """
    a = torch.tanh(anchors_logits)
    p = torch.tanh(positives_logits)
    a_norm = F.normalize(a, dim=-1)
    p_norm = F.normalize(p, dim=-1)
    sim = torch.mm(a_norm, p_norm.T) / temperature
    labels = torch.arange(len(a), device=anchors_logits.device)
    return nn.CrossEntropyLoss()(sim, labels)
