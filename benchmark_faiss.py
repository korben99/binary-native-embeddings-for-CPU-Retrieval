"""
Retrieval speed & memory benchmark at scale.

Backend selection (automatic):
  - FAISS available (x86_64 Intel/AMD) → IndexFlatIP + IndexBinaryFlat (AVX2 + POPCNT)
  - FAISS unavailable (Apple ARM64)    → pure NumPy fallback (conservative lower bound)

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

# Corporate proxy / self-signed cert fix: set CURL_CA_BUNDLE= in .env
import os as _os
if _os.environ.get("CURL_CA_BUNDLE", "NOT_SET") == "":
    import ssl as _ssl
    _ssl._create_default_https_context = _ssl._create_unverified_context

BASE_DIR     = Path(__file__).parent
RESULTS_DIR  = BASE_DIR / "results"
FLOAT_DIM    = 384
BINARY_DIM   = 4096
BINARY_BYTES = BINARY_DIM // 8   # 512 bytes / vector

# Lookup table: number of set bits in each byte value (0-255)
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# ── FAISS detection ───────────────────────────────────────────────────────────
# faiss-cpu pip wheel segfaults on ARM64 (Apple Silicon) with Python 3.13.
# Force numpy backend there; use FAISS only on x86_64 where the wheel works.
try:
    import faiss
    HAVE_FAISS = platform.machine() != "arm64"   # arm64 = Apple Silicon (pip wheel segfaults)
except ImportError:
    HAVE_FAISS = False


# ── Bit packing ───────────────────────────────────────────────────────────────

def pack_binary(vecs: np.ndarray) -> np.ndarray:
    """Convert {-1,+1} float32 [N, D] -> packed uint8 [N, D//8]."""
    return np.packbits((vecs > 0).astype(np.uint8), axis=1)


# ── Search backends ───────────────────────────────────────────────────────────

# --- FAISS (x86 AVX2 + POPCNT) ---

def faiss_float_search(queries, db, k=10):
    q = queries.copy().astype(np.float32)
    d = db.copy().astype(np.float32)
    faiss.normalize_L2(q)
    faiss.normalize_L2(d)
    idx = faiss.IndexFlatIP(FLOAT_DIM)
    idx.add(d)
    _, I = idx.search(q, k)
    return I

def faiss_binary_search(q_packed, db_packed, k=10):
    idx = faiss.IndexBinaryFlat(BINARY_DIM)
    idx.add(db_packed)
    _, I = idx.search(q_packed, k)
    return I

# --- NumPy fallback ---

def numpy_float_search(queries, db, k=10):
    q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    d = db     / (np.linalg.norm(db,      axis=1, keepdims=True) + 1e-9)
    sims = q @ d.T
    return np.argpartition(-sims, k, axis=1)[:, :k]

def numpy_binary_search(q_packed, db_packed, k=10, chunk=8_000):
    """Vectorised XOR + popcount lookup, chunked over N to cap memory."""
    Q, N = len(q_packed), len(db_packed)
    distances = np.empty((Q, N), dtype=np.int32)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        xor = q_packed[:, None, :] ^ db_packed[None, start:end, :]  # [Q, c, B]
        distances[:, start:end] = POPCOUNT[xor].sum(axis=2)
    return np.argpartition(distances, k, axis=1)[:, :k]


# ── Corpus generation ─────────────────────────────────────────────────────────

def make_float_corpus(seeds: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(seeds), n)
    corpus = seeds[idx].copy().astype(np.float32)
    corpus += rng.standard_normal(corpus.shape).astype(np.float32) * 0.3
    return corpus

def make_binary_corpus_packed(seeds_packed: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(seeds_packed), n)
    corpus = seeds_packed[idx].copy()
    flip = rng.random(corpus.shape) < 0.15
    corpus[flip] ^= np.uint8(0xFF)
    return corpus


# ── Timing ────────────────────────────────────────────────────────────────────

def bench(fn, *args, n_runs=10, warmup=3):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def speedup_str(float_ms: float, binary_ms: float) -> str:
    ratio = float_ms / binary_ms
    return f"{ratio:.1f}x faster" if ratio >= 1 else f"{1/ratio:.1f}x slower"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from models.float_embedder  import FloatEmbedder
    from models.binary_embedder import BinaryEmbedder

    machine = platform.machine()
    backend = "FAISS (AVX2+POPCNT)" if HAVE_FAISS else "NumPy (no hardware POPCNT)"
    platform_label = (
        f"{platform.node()} | {platform.processor() or machine} "
        f"| Python {platform.python_version()} | backend: {backend}"
    )
    print(f"\nPlatform : {platform_label}")
    print(f"Backend  : {backend}")

    if HAVE_FAISS:
        float_search_fn  = faiss_float_search
        binary_search_fn = faiss_binary_search
    else:
        float_search_fn  = numpy_float_search
        binary_search_fn = numpy_binary_search

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

    print(f"\nEncoding {len(seed_queries)} seed queries...")
    float_seeds         = float_model.encode(seed_queries, tokenizer).numpy().astype(np.float32)
    binary_seeds        = binary_model.encode(seed_queries, tokenizer).numpy().astype(np.float32)
    binary_seeds_packed = pack_binary(binary_seeds)

    scales  = [10_000, 100_000, 1_000_000]
    results = {"platform": platform_label, "backend": backend}

    for n in scales:
        print(f"\n{'='*58}")
        print(f"  Scale: {n:,} vectors")

        float_mem_mb  = n * FLOAT_DIM  * 4  / 1e6
        binary_mem_mb = n * BINARY_BYTES    / 1e6

        print(f"  Building float  corpus  ({float_mem_mb:.0f} MB)...")
        float_corpus  = make_float_corpus(float_seeds, n)

        print(f"  Building binary corpus  ({binary_mem_mb:.0f} MB packed)...")
        binary_corpus = make_binary_corpus_packed(binary_seeds_packed, n)

        print(f"  Timing (16 queries, top-10, 10 runs)...")
        float_ms  = bench(float_search_fn,  float_seeds,         float_corpus,  10)
        binary_ms = bench(binary_search_fn, binary_seeds_packed, binary_corpus, 10)

        mem_ratio = float_mem_mb / binary_mem_mb
        vs        = speedup_str(float_ms, binary_ms)

        results[str(n)] = {
            "n_vectors":        n,
            "float_mem_mb":     round(float_mem_mb,  1),
            "binary_mem_mb":    round(binary_mem_mb, 2),
            "mem_ratio_x":      round(mem_ratio, 1),
            "float_search_ms":  round(float_ms,  2),
            "binary_search_ms": round(binary_ms, 2),
            "vs_float":         vs,
        }

        print(f"  Float:  {float_ms:8.2f} ms  |  {float_mem_mb:6.0f} MB")
        print(f"  Binary: {binary_ms:8.2f} ms  |  {binary_mem_mb:6.2f} MB (packed bits)")
        print(f"  => binary {vs}  |  {mem_ratio:.0f}x smaller index")

    RESULTS_DIR.mkdir(exist_ok=True)
    slug = f"{machine.lower()}_{'faiss' if HAVE_FAISS else 'numpy'}"
    out  = RESULTS_DIR / f"retrieval_benchmark_{slug}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {out}")

    print("\n" + "=" * 78)
    print(f"  {backend}")
    print(f"{'Scale':>12} | {'Float (ms)':>10} | {'Binary (ms)':>11} | {'vs Float':>14} | {'Mem ratio':>10}")
    print("-" * 78)
    for n_str, r in results.items():
        if not n_str.isdigit():
            continue
        print(
            f"{r['n_vectors']:>12,} | {r['float_search_ms']:>10.2f} | "
            f"{r['binary_search_ms']:>11.2f} | {r['vs_float']:>14} | "
            f"{r['mem_ratio_x']:>9.0f}x"
        )
    print("=" * 78)

    if not HAVE_FAISS:
        print("\n[!] NumPy backend active — binary is memory-bandwidth bound.")
        print("    On x86 with faiss-cpu, IndexBinaryFlat uses AVX2+POPCNT")
        print("    and typically shows 5-20x speedup over float at 1M scale.")


if __name__ == "__main__":
    main()
