"""
Full benchmark: STS-B Spearman, SciFact Recall@10, CPU latency.
Run after training both models:
  python benchmark.py
Results saved to results/benchmark_results.json
"""
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import BertTokenizer

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_cache"
CKPT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"


# ── Similarity ────────────────────────────────────────────────────────────────

def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return torch.mm(a_n, b_n.T)


def hamming_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Similarity for {-1,+1} binary vectors via normalized dot product.
    Equivalent to 1 - 2*hamming_distance/D, range [-1, +1].
    """
    D = a.shape[1]
    return torch.mm(a, b.T) / D


# ── Bit diagnostics ───────────────────────────────────────────────────────────

def bit_diagnostics(binary_vecs: np.ndarray) -> dict:
    """
    binary_vecs: np.array (N, D), values in {-1, +1}

    Note: LayerNorm before STE guarantees balance≈0.5 and entropy≈1.0 by
    construction — those metrics are uninformative for this architecture.
    The meaningful signal is inter-bit correlation (redundancy).
    """
    N, D = binary_vecs.shape
    balance = (binary_vecs == 1).mean(axis=0)
    p = balance
    entropy = -p * np.log2(p + 1e-9) - (1 - p) * np.log2(1 - p + 1e-9)
    dead_bits = ((balance < 0.05) | (balance > 0.95)).sum()

    # Inter-bit correlation — cap at 2048 bits to control memory (D×D matrix)
    sample = binary_vecs[:, :min(D, 2048)].astype(np.float32)
    sample -= sample.mean(axis=0)
    std = sample.std(axis=0) + 1e-6
    sample /= std
    corr = (sample.T @ sample) / N          # (D', D') correlation matrix
    d_ = corr.shape[0]
    mask = ~np.eye(d_, dtype=bool)
    off_diag = np.abs(corr[mask])
    mean_corr = float(off_diag.mean())
    max_corr  = float(off_diag.max())

    print(f"    Bits morts           : {dead_bits} / {D}")
    print(f"    Entropie moyenne     : {entropy.mean():.4f} ± {entropy.std():.4f}  (idéal 1.0)")
    print(f"    Balance moyenne      : {balance.mean():.4f} ± {balance.std():.4f}  (idéal 0.5)")
    print(f"    Corrélation inter-bits (|r|) : mean={mean_corr:.4f}  max={max_corr:.4f}  (idéal 0.0)")
    if D > 2048:
        print(f"    [corrélation calculée sur les 2048 premiers bits]")

    return {
        "dead_bits": int(dead_bits),
        "entropy_mean": float(entropy.mean()),
        "entropy_std":  float(entropy.std()),
        "balance_mean": float(balance.mean()),
        "balance_std":  float(balance.std()),
        "mean_abs_corr": mean_corr,
        "max_abs_corr":  max_corr,
    }


def run_bit_diagnostics(model, tokenizer, n_samples=5000) -> dict:
    """
    Encode a diverse random sample of NLI sentences for bit-level statistics.
    STS-B is unsuitable (semantically similar pairs → artificially smooth stats).
    NLI covers 550k varied topics, giving an honest picture of bit utilization.
    """
    from datasets import load_from_disk, load_dataset
    cache = DATA_DIR / "nli_train"
    ds = load_from_disk(str(cache)) if cache.exists() else \
         load_dataset("sentence-transformers/all-nli", "triplet", split="train")
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(ds), n_samples)
    texts = [ds["anchor"][int(i)] for i in idx]
    print(f"    corpus: NLI {n_samples} random samples")
    vecs = model.encode(texts, tokenizer).numpy()
    return bit_diagnostics(vecs)


# ── STS-B ─────────────────────────────────────────────────────────────────────

def eval_stsb(model, tokenizer, use_binary=False):
    from datasets import load_from_disk, load_dataset

    cache = DATA_DIR / "sts_test"
    if cache.exists():
        ds = load_from_disk(str(cache))
    else:
        print("  Downloading STS-B...")
        ds = load_dataset("mteb/stsbenchmark-sts", split="test")

    human = np.array(ds["score"]) / 5.0  # normalize to [0,1]
    embs1 = model.encode(list(ds["sentence1"]), tokenizer)
    embs2 = model.encode(list(ds["sentence2"]), tokenizer)

    sim_fn = hamming_sim_matrix if use_binary else cosine_sim_matrix
    pred = sim_fn(embs1, embs2).diag().numpy()

    corr, _ = spearmanr(pred, human)
    return float(corr)


# ── SciFact Recall@10 ─────────────────────────────────────────────────────────

def load_scifact():
    cache = DATA_DIR / "scifact"
    if cache.exists():
        corpus = json.loads((cache / "corpus.json").read_text())
        queries = json.loads((cache / "queries.json").read_text())
        qrels = json.loads((cache / "qrels.json").read_text())
        return corpus, queries, qrels

    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
        path = util.download_and_unzip(url, str(DATA_DIR / "beir"))
        return GenericDataLoader(data_folder=path).load(split="test")
    except Exception as e:
        print(f"  SciFact unavailable: {e}")
        return None, None, None


def _recall_from_corpus(model, tokenizer, corpus, queries, qrels, use_binary=False, top_k=10):
    """Shared recall computation. Returns (mean_recall, per_query_recalls)."""
    doc_ids   = list(corpus.keys())
    doc_texts = [f"{corpus[d].get('title','') } {corpus[d].get('text','') }".strip()
                 for d in doc_ids]
    valid_qids = [qid for qid in queries if qid in qrels]
    q_texts    = [queries[qid] for qid in valid_qids]

    print(f"  Encoding {len(doc_texts):,} docs...")
    corpus_embs = model.encode(doc_texts, tokenizer)
    print(f"  Encoding {len(q_texts)} queries...")
    query_embs  = model.encode(q_texts, tokenizer)

    sim_fn = hamming_sim_matrix if use_binary else cosine_sim_matrix
    recalls = []
    for i, qid in enumerate(valid_qids):
        sims    = sim_fn(query_embs[i:i+1], corpus_embs)[0]
        top_idx = sims.topk(min(top_k, len(doc_ids))).indices.tolist()
        retrieved = {doc_ids[j] for j in top_idx}
        relevant  = set(qrels[qid].keys())
        recalls.append(len(retrieved & relevant) / max(len(relevant), 1))

    return float(np.mean(recalls)), recalls


def eval_scifact_recall(model, tokenizer, use_binary=False, top_k=10):
    corpus, queries, qrels = load_scifact()
    if corpus is None:
        return None, None
    return _recall_from_corpus(model, tokenizer, corpus, queries, qrels, use_binary, top_k)


def eval_beir_recall(model, tokenizer, dataset_name, use_binary=False, top_k=10):
    """
    Generic BEIR dataset evaluation. Downloads on first run, caches locally.
    Recommended: 'scidocs' (1000 queries, 25k docs), 'nfcorpus' (323 queries, 3.6k docs).
    Returns (mean_recall, per_query_recalls) or (None, None) on failure.
    """
    beir_dir = DATA_DIR / "beir" / dataset_name
    try:
        from beir.datasets.data_loader import GenericDataLoader
        if not beir_dir.exists():
            from beir import util
            url = (f"https://public.ukp.informatik.tu-darmstadt.de"
                   f"/thakur/BEIR/datasets/{dataset_name}.zip")
            print(f"  Downloading {dataset_name}...")
            util.download_and_unzip(url, str(DATA_DIR / "beir"))
        corpus, queries, qrels = GenericDataLoader(str(beir_dir)).load(split="test")
    except Exception as e:
        print(f"  {dataset_name} unavailable: {e}")
        return None, None
    return _recall_from_corpus(model, tokenizer, corpus, queries, qrels, use_binary, top_k)


# ── Bootstrap significance test ───────────────────────────────────────────────

def bootstrap_recall_diff(per_query_a, per_query_b, n_boot=2000, seed=0):
    """
    Two-sided bootstrap test: is mean(a) - mean(b) significantly ≠ 0?
    per_query_a/b : lists of per-query recall values (same queries, same order).
    Returns dict with diff, p_value, ci_95.
    """
    a   = np.array(per_query_a, dtype=float)
    b   = np.array(per_query_b, dtype=float)
    obs = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    n   = len(a)
    boot_diffs = np.array([
        a[rng.integers(0, n, n)].mean() - b[rng.integers(0, n, n)].mean()
        for _ in range(n_boot)
    ])
    p = float((boot_diffs <= 0).mean() if obs > 0 else (boot_diffs >= 0).mean())
    ci = (float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5)))
    return {"diff": round(obs, 5), "p_value": round(p, 4), "ci_95": ci, "n_queries": n}


# ── Latency ───────────────────────────────────────────────────────────────────

def benchmark_latency(model, tokenizer, n_runs=100, batch_size=32):
    texts = ["the quick brown fox jumps over the lazy dog"] * batch_size
    model.eval()

    for _ in range(5):  # warmup
        model.encode(texts, tokenizer, device="cpu")

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.encode(texts, tokenizer, device="cpu")
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": round(float(np.mean(times)), 2),
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
    }


def memory_per_1k(dim, is_binary):
    bytes_each = dim / 8 if is_binary else dim * 4
    total = bytes_each * 1000
    if total < 1024:
        return f"{total:.0f} B"
    elif total < 1024**2:
        return f"{total/1024:.0f} KB"
    else:
        return f"{total/1024**2:.2f} MB"


# ── Post-hoc binary wrapper ───────────────────────────────────────────────────

class PostHocBinaryWrapper:
    """Wraps a float model, applies sign binarization at inference time."""
    def __init__(self, base):
        self.base = base

    def encode(self, texts, tokenizer, device="cpu", batch_size=64):
        floats = self.base.encode(texts, tokenizer, device=device, batch_size=batch_size)
        return torch.sign(floats).float()  # {-1, +1}

    def eval(self):
        self.base.eval()


# ── Q4 quantized float wrapper ────────────────────────────────────────────────

class Q4FloatWrapper:
    """
    Float embedder with INT4 weight-only quantization (torchao).
    Falls back to PyTorch INT8 dynamic quantization if torchao is unavailable.
    Output is still float32 384-dim — index memory unchanged.
    """
    def __init__(self, base):
        import copy
        m = copy.deepcopy(base)
        try:
            from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig
            try:
                quantize_(m, Int4WeightOnlyConfig())
                self._backend = "torchao INT4"
            except Exception:
                m = copy.deepcopy(base)
                quantize_(m, Int8WeightOnlyConfig())
                self._backend = "torchao INT8 (fallback)"
        except Exception as e:
            raise RuntimeError(f"torchao unavailable: {e}. Run: pip install torchao")
        self.model = m
        print(f"  [Q4] backend: {self._backend}")

    def encode(self, texts, tokenizer, device="cpu", batch_size=64):
        with torch.no_grad():
            return self.model.encode(texts, tokenizer, device=device, batch_size=batch_size).float()

    def eval(self):
        self.model.eval()


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_suffix(suffix: str) -> int:
    """Extract the dim from a checkpoint suffix. '1024_bs256' → 1024."""
    return int(suffix.split("_")[0])


def main(checkpoints=("2048", "4096"), datasets=("scifact",)):
    from models.float_embedder import FloatEmbedder
    from models.binary_embedder import BinaryEmbedder

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")

    print("\n=== Loading models ===")
    float_model = FloatEmbedder(output_dim=384)
    ckpt = CKPT_DIR / "float_embedder.pt"
    if ckpt.exists():
        float_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"  float_embedder.pt loaded")
    else:
        print(f"  WARNING: {ckpt} not found — using random weights")
    float_model.eval()

    posthoc_model = PostHocBinaryWrapper(float_model)

    print("  Applying Q4 quantization...")
    q4_model = Q4FloatWrapper(float_model)
    q4_model.eval()

    configs = [
        ("float32_384",        float_model,   False, 384, False),
        ("float32_q4_384",     q4_model,      False, 384, False),
        ("binary_posthoc_384", posthoc_model, True,  384, True),
    ]

    for suffix in checkpoints:
        dim = _parse_suffix(suffix)
        label = f"binary_native_{suffix}"
        binary_model = BinaryEmbedder(binary_dim=dim)
        ckpt = CKPT_DIR / f"binary_embedder_{suffix}.pt"
        if ckpt.exists():
            binary_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"  binary_embedder_{suffix}.pt loaded")
        else:
            print(f"  WARNING: {ckpt} not found — skipping")
            continue
        binary_model.eval()
        configs.append((label, binary_model, True, dim, True))

    results = {}

    for name, model, use_binary, dim, is_binary in configs:
        label = "binary" if is_binary else "float"
        print(f"\n[{label}] Evaluating {name}...")

        print("  STS-B Spearman...")
        stsb = eval_stsb(model, tokenizer, use_binary=use_binary)

        dataset_results = {}
        for ds_name in datasets:
            print(f"  Recall@10 [{ds_name}]...")
            if ds_name == "scifact":
                mean_r, per_q = eval_scifact_recall(model, tokenizer, use_binary=use_binary)
            else:
                mean_r, per_q = eval_beir_recall(model, tokenizer, ds_name, use_binary=use_binary)
            dataset_results[ds_name] = {
                "recall10": round(mean_r, 4) if mean_r is not None else None,
                "per_query": per_q,
            }

        print("  CPU latency (batch=32, 100 runs)...")
        lat = benchmark_latency(model, tokenizer)

        bit_diag = None
        if is_binary:
            print("  Bit diagnostics...")
            bit_diag = run_bit_diagnostics(model, tokenizer)

        dtype = "binary" if is_binary else ("float32_q4" if "q4" in name else "float32")
        # primary recall = first dataset for backward compat
        primary_ds   = datasets[0]
        primary_r10  = dataset_results[primary_ds]["recall10"]
        results[name] = {
            "dims": dim,
            "dtype": dtype,
            "stsb_spearman": round(stsb, 4),
            "scifact_recall10": dataset_results.get("scifact", {}).get("recall10"),
            "recall_by_dataset": {k: v["recall10"] for k, v in dataset_results.items()},
            "per_query_by_dataset": {k: v["per_query"] for k, v in dataset_results.items()},
            "memory_1k_vecs": memory_per_1k(dim, is_binary),
            "latency_cpu": lat,
            "bit_diagnostics": bit_diag,
            **({"q4_backend": model._backend} if hasattr(model, "_backend") else {}),
        }

        r10_str = "  ".join(
            f"{k}={v['recall10']:.4f}" for k, v in dataset_results.items() if v["recall10"]
        )
        print(f"  STS-B={stsb:.4f}  {r10_str}  lat={lat['mean_ms']}ms")

    RESULTS_DIR.mkdir(exist_ok=True)
    from datetime import date
    out = RESULTS_DIR / f"benchmark_results_{date.today():%Y%m%d}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {out}")

    # Pretty table
    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'Dims':>6} {'Type':>8} {'STS-B':>8} {'R@10':>8} {'Memory':>10} {'Lat (ms)':>10}")
    print("-" * 90)
    for name, r in results.items():
        r10 = f"{r['scifact_recall10']:.4f}" if r["scifact_recall10"] else "   N/A"
        print(
            f"{name:<25} {r['dims']:>6} {r['dtype']:>8} "
            f"{r['stsb_spearman']:>8.4f} {r10:>8} "
            f"{r['memory_1k_vecs']:>10} {r['latency_cpu']['mean_ms']:>9.1f}ms"
        )
    print("=" * 90)

    # Bit diagnostics table (binary models only)
    binary_results = {n: r for n, r in results.items() if r["bit_diagnostics"]}
    if binary_results:
        print("\n" + "=" * 95)
        print(f"{'Model':<25} {'Dims':>6} {'Dead':>6} {'H mean':>8} {'H std':>7} {'Bal std':>8} {'|r| mean':>9} {'|r| max':>8}")
        print("-" * 95)
        for name, r in binary_results.items():
            d = r["bit_diagnostics"]
            print(
                f"{name:<25} {r['dims']:>6} {d['dead_bits']:>6}"
                f" {d['entropy_mean']:>8.4f} {d['entropy_std']:>7.4f}"
                f" {d['balance_std']:>8.4f}"
                f" {d['mean_abs_corr']:>9.4f} {d['max_abs_corr']:>8.4f}"
            )
        print("=" * 95)
        print("  ideal: dead=0  H=1.0000  H_std>0 (dispersion)  bal_std>0  |r|_mean≈0  |r|_max≈0")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Checkpoint suffixes, e.g. '1024' '1024_s42' '1024_s123' '2048'")
    parser.add_argument("--binary_dims", type=int, nargs="+", default=None,
                        help="Shorthand: --binary_dims 1024 2048 → same as --checkpoints 1024 2048")
    parser.add_argument("--datasets", type=str, nargs="+", default=["scifact"],
                        help="BEIR datasets to evaluate, e.g. 'scifact scidocs nfcorpus'")
    args = parser.parse_args()

    if args.checkpoints:
        checkpoints = args.checkpoints
    elif args.binary_dims:
        checkpoints = [str(d) for d in args.binary_dims]
    else:
        checkpoints = ["2048", "4096"]

    main(checkpoints=checkpoints, datasets=tuple(args.datasets))
