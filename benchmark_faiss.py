"""
Retrieval speed & memory benchmark at scale — pure NumPy, no FAISS dependency.
Compares float32 matmul vs binary popcount at 10k / 100k / 1M vectors.

Note: a production FAISS IndexBinaryFlat (x86 POPCNT / ARM NEON) would show
even larger speedups; this gives a conservative lower bound.

Usage: python benchmark_faiss.py
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import BertTokenizer

load_dotenv()

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
FLOAT_DIM   = 384
BINARY_DIM  = 4096
BINARY_BYTES = BINARY_DIM // 8   # 512 bytes / vector packed

# lookup table: number of 1-bits in each possible byte value
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ── Core search routines ──────────────────────────────────────────────────────

def float_search(queries: np.ndarray, db: np.ndarray, k: int = 10) -> np.ndarray:
    """Cosine similarity via normalized matmul (BLAS-accelerated)."""
    q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    d = db     / (np.linalg.norm(db,      axis=1, keepdims=True) + 1e-9)
    sims = q @ d.T                                   # [Q, N]
    return np.argpartition(-sims, k, axis=1)[:, :k]  # [Q, k]


def pack_binary(vecs: np.ndarray) -> np.ndarray:
    """Convert {-1,+1} float32 [N, D] -> packed uint8 [N, D//8]."""
    return np.packbits((vecs > 0).astype(np.uint8), axis=1)


def binary_search(q_packed: np.ndarray, db_packed: np.ndarray, k: int = 10,
                  chunk: int = 8_000) -> np.ndarray:
    """
    Hamming distance search — fully vectorised over Q and chunked over N.
    Per chunk: [Q, chunk, B] XOR then popcount lookup → no Python inner loop.
    Peak memory ≈ Q × chunk × B bytes (16 × 8k × 512 = 64 MB).
    """
    Q = len(q_packed)
    N = len(db_packed)
    distances = np.empty((Q, N), dtype=np.int32)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        xor = q_packed[:, None, :] ^ db_packed[None, start:end, :]  # [Q, chunk, B]
        distances[:, start:end] = POPCOUNT[xor].sum(axis=2)
    return np.argpartition(distances, k, axis=1)[:, :k]


# ── Timing helper ─────────────────────────────────────────────────────────────

def bench(fn, *args, n_runs=10, warmup=3):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


# ── Corpus generation ─────────────────────────────────────────────────────────

def make_float_corpus(seeds: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(seeds), n)
    corpus = seeds[idx].copy().astype(np.float32)
    corpus += rng.standard_normal(corpus.shape).astype(np.float32) * 0.3
    return corpus


def make_binary_corpus_packed(seeds_packed: np.ndarray, n: int) -> np.ndarray:
    """Directly generate packed binary corpus — avoids the 4096×4 bytes float blowup."""
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(seeds_packed), n)
    corpus = seeds_packed[idx].copy()
    # flip ~15 % of bits by XOR-ing random bytes
    flip_mask = rng.random(corpus.shape) < 0.15
    corpus[flip_mask] ^= np.uint8(0xFF)
    return corpus


# ── Main ──────────────────────────────────────────────────────────────────────

def get_platform_label() -> str:
    machine = platform.machine()          # arm64 / x86_64
    cpu     = platform.processor()       # Apple M4 Pro / Intel ...
    node    = platform.node()
    return f"{node} | {cpu or machine} | Python {platform.python_version()}"


def main():
    from models.float_embedder  import FloatEmbedder
    from models.binary_embedder import BinaryEmbedder

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")

    float_model = FloatEmbedder(output_dim=FLOAT_DIM)
    ckpt = BASE_DIR / "checkpoints" / "float_embedder.pt"
    if ckpt.exists():
        float_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    float_model.eval()

    binary_model = BinaryEmbedder(binary_dim=BINARY_DIM)
    ckpt = BASE_DIR / "checkpoints" / "binary_embedder.pt"
    if ckpt.exists():
        binary_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    binary_model.eval()

    seed_queries = [
        "what causes alzheimer disease",
        "climate change effects on biodiversity",
        "neural network training optimization",
        "covid-19 vaccine efficacy",
        "quantum computing applications",
        "protein folding structure prediction",
        "machine learning interpretability",
        "antibiotic resistance mechanisms",
        "deep learning natural language processing",
        "solar energy efficiency improvements",
        "cancer immunotherapy treatment",
        "autonomous vehicle safety systems",
        "gene editing CRISPR technology",
        "black hole gravitational waves detection",
        "renewable energy battery storage",
        "microbiome gut health research",
    ]

    print(f"Encoding {len(seed_queries)} seed queries...")
    float_seeds  = float_model.encode(seed_queries, tokenizer).numpy().astype(np.float32)
    binary_seeds = binary_model.encode(seed_queries, tokenizer).numpy().astype(np.float32)
    binary_seeds_packed = pack_binary(binary_seeds)

    platform_label = get_platform_label()
    print(f"\nPlatform: {platform_label}")

    scales  = [10_000, 100_000, 1_000_000]
    results = {"platform": platform_label}

    for n in scales:
        print(f"\n{'='*58}")
        print(f"  Scale: {n:,} vectors")

        float_mem_mb  = n * FLOAT_DIM  * 4     / 1e6
        binary_mem_mb = n * BINARY_BYTES        / 1e6

        print(f"  Building float  corpus  ({float_mem_mb:.0f} MB)...")
        float_corpus  = make_float_corpus(float_seeds, n)

        print(f"  Building binary corpus  ({binary_mem_mb:.0f} MB packed)...")
        binary_corpus = make_binary_corpus_packed(binary_seeds_packed, n)

        print(f"  Timing search (16 queries, top-10, 10 runs)...")
        float_ms  = bench(float_search,  float_seeds,         float_corpus,  10)
        binary_ms = bench(binary_search, binary_seeds_packed, binary_corpus, 10)

        speedup   = float_ms / binary_ms   # >1 = binary faster, <1 = binary slower
        mem_ratio = float_mem_mb / binary_mem_mb

        results[str(n)] = {
            "n_vectors":        n,
            "float_mem_mb":     round(float_mem_mb,  1),
            "binary_mem_mb":    round(binary_mem_mb, 2),
            "mem_ratio_x":      round(mem_ratio,  1),
            "float_search_ms":  round(float_ms,  2),
            "binary_search_ms": round(binary_ms, 2),
            "speedup_x":        round(speedup, 1),
        }

        ratio_str = f"{speedup:.1f}x faster" if speedup >= 1 else f"{1/speedup:.1f}x slower"
        print(f"  Float:  {float_ms:8.2f} ms  |  {float_mem_mb:6.0f} MB")
        print(f"  Binary: {binary_ms:8.2f} ms  |  {binary_mem_mb:6.2f} MB (packed bits)")
        print(f"  => binary {ratio_str} than float  |  {mem_ratio:.0f}x smaller index")

    RESULTS_DIR.mkdir(exist_ok=True)
    machine_slug = platform.machine().lower()          # arm64 / x86_64
    out = RESULTS_DIR / f"retrieval_benchmark_{machine_slug}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {out}")

    print("\n" + "=" * 76)
    print(f"{'Scale':>12} | {'Float (ms)':>10} | {'Binary (ms)':>11} | {'vs Float':>10} | {'Mem ratio':>10}")
    print("-" * 76)
    for n_str, r in results.items():
        if n_str == "platform":
            continue
        s = r["speedup_x"]
        vs = f"{s:.1f}x faster" if s >= 1 else f"{1/s:.1f}x slower"
        print(
            f"{r['n_vectors']:>12,} | {r['float_search_ms']:>10.2f} | "
            f"{r['binary_search_ms']:>11.2f} | {vs:>10} | "
            f"{r['mem_ratio_x']:>9.0f}x"
        )
    print("=" * 76)
    print("\nNote: hardware POPCNT (FAISS IndexBinaryFlat) would show additional speedup.")
    print("Numpy binary search is memory-bandwidth bound; BLAS float matmul uses SIMD.")


if __name__ == "__main__":
    main()
