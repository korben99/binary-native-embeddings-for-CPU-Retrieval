# Binary Native Embeddings

> **Hypothesis** — A transformer trained natively with a 4096-bit binary head and a contrastive binary loss produces *better semantic retrieval* than the same transformer binarized post-hoc, at lower memory cost than float32 — no GPU required at inference.

Backbone: `prajjwal1/bert-mini` (4 layers, 256 hidden, ~11M params).  
Hardware: Mac Mini M4 Pro (ARM64) + Intel Core Ultra 7 155H (x86_64), **CPU only**.

---

## Results — Embedding quality

| Model | Dims | Type | STS-B Spearman ↑ | Recall@10 ↑ | Memory / 1k vecs ↓ | Encode latency |
|---|---|---|---|---|---|---|
| Float baseline | 384 | float32 | 0.7355 | 0.3131 | 1.46 MB | ~4.7 ms |
| Post-hoc binary | 384 | binary | 0.7271 | 0.2358 | 48 KB | ~4.5 ms |
| **Native binary** | **4096** | **binary** | **0.7275** | **0.2958** | **500 KB** | **~4.8 ms** |

**Hypothesis validated on Recall@10**: native binary 4096-dim beats post-hoc binary 384-dim by **+25%** (0.296 vs 0.236) while matching it on STS-B (0.7275 vs 0.7271, within noise).

Encode latency is identical across all three models because it is dominated by the BERT forward pass, not the vector dimension.

---

## Results — Retrieval at scale

Benchmark: 16 queries, top-10, averaged over 10 runs.  
Backend: pure NumPy (BLAS float matmul vs XOR + popcount lookup table).

### Mac Mini M4 Pro — Apple Accelerate BLAS

| Scale | Float (ms) | Binary (ms) | Memory ratio |
|---|---|---|---|
| 10k | 2.2 | 76.6 | 3× smaller |
| 100k | 25.8 | 770.0 | 3× smaller |
| 1M | **238** | 7 752 | 3× smaller |

### Intel Core Ultra 7 155H — OpenBLAS / MKL

| Scale | Float (ms) | Binary (ms) | Memory ratio |
|---|---|---|---|
| 10k | 33.9 | 291.0 | 3× smaller |
| 100k | 146.7 | 2 961.1 | 3× smaller |
| 1M | **1 127** | 29 766 | 3× smaller |

**M4 Pro is 4.7× faster than Intel for float32 retrieval** at 1M scale (238 ms vs 1 127 ms) — Apple's Accelerate BLAS benefits from ARM NEON/AMX and unified memory bandwidth.

### Why binary is slower in these benchmarks

Our numpy binary search is *structurally* slower than float matmul:

| | Float | Binary (numpy) |
|---|---|---|
| Kernel | BLAS `SGEMM` (SIMD-vectorized) | Python loop + XOR + lookup table |
| Hardware used | NEON / AVX2 SIMD | scalar + cache-unfriendly indirect access |

**This comparison is unfair by construction.** The real binary advantage appears with hardware POPCNT:

| Backend | Expected 1M retrieval |
|---|---|
| NumPy float (BLAS) | 238 ms (M4 Pro) |
| NumPy binary (lookup) | 7 752 ms — this benchmark |
| **FAISS `IndexBinaryFlat` (POPCNT)** | **~15–50 ms** (estimated) |

`faiss-cpu` currently has no wheel for Python 3.14 (Windows test machine) and segfaults on ARM64/Python 3.13, which prevented direct measurement. The theoretical argument stands: POPCNT processes 64 bits in a single instruction vs 64 multiplications for float, giving a ×32 operation count reduction that SIMD implementations exploit fully.

---

## Architecture

```
Input text
    │
    ▼
bert-mini encoder (4L × 256d, ~11M params, shared)
    │  mean pooling
    ▼
[256-dim pooled]
    │
    ├── FloatEmbedder:   Linear(256 → 384)                → 384-dim float32
    │
    └── BinaryEmbedder: Linear(256 → 4096) + LayerNorm   → STE → {-1,+1}^4096
```

### Straight-Through Estimator (STE)

Standard `sign()` has zero gradient almost everywhere. STE fixes this:

```python
class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x).float()   # {-1, +1} — discrete

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output             # identity — gradient flows unchanged
```

Forward produces true binary outputs; backward treats binarization as identity, letting gradients reach the projection head and encoder.

### Training losses

**Float baseline** — MultipleNegativesRankingLoss:
```python
def mnrl_loss(anchors, positives, temperature=0.05):
    a, p = F.normalize(anchors), F.normalize(positives)
    sim  = torch.mm(a, p.T) / temperature    # [B, B] cosine
    return CrossEntropyLoss()(sim, torch.arange(B))
```

**Native binary** — tanh contrastive loss on pre-binarization logits:
```python
def binary_contrastive_loss(a_logits, p_logits, temperature=0.05):
    a = F.normalize(torch.tanh(a_logits))    # tanh ≈ {-1,+1}, differentiable
    p = F.normalize(torch.tanh(p_logits))
    sim = torch.mm(a, p.T) / temperature
    return CrossEntropyLoss()(sim, torch.arange(B))
```

`tanh` is used instead of `sigmoid` because it maps to `(-1, +1)`, aligned with the `{-1, +1}` STE output — training directly optimizes the metric used at evaluation.

**Differential learning rate**: the projection layer (randomly initialized) uses `lr × 50` relative to the encoder (pretrained), allowing it to converge within the same number of epochs.

---

## Why native binary outperforms post-hoc on retrieval

Post-hoc binarization collapses a 384-dim float space into 384 bits. It discards sign information in an uncontrolled way — bits that were "on the fence" (near-zero activations) flip arbitrarily.

Native binary training gives the model:
1. **10× more dimensions** (4096 vs 384) to distribute information across bits
2. **A loss that explicitly optimizes binary similarity** — `tanh` cosine aligns with the {-1,+1} Hamming metric used at eval
3. **Redundancy** — semantically related concepts are encoded across multiple bits, making the representation robust to individual bit noise

The result: same semantic precision (STS-B), better recall coverage (+25% Recall@10).

---

## Memory breakdown

| Representation | Formula | 1k vectors |
|---|---|---|
| float32 × 384 | 384 × 4 B × 1k | 1.46 MB |
| binary × 384 (post-hoc) | 384 / 8 B × 1k | 48 KB |
| **binary × 4096 (native)** | **4096 / 8 B × 1k** | **500 KB** |

Native binary at 4096 dims uses **3× less memory than float** and **10× more than post-hoc binary**, but delivers **25% better Recall@10** than the memory-cheaper alternative.

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/binary-native-embeddings
cd binary-native-embeddings
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```python
from transformers import BertTokenizer
from models.binary_embedder import BinaryEmbedder

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
model     = BinaryEmbedder(binary_dim=4096)
# model.load_state_dict(torch.load("checkpoints/binary_embedder.pt"))

vecs = model.encode(["binary embeddings are fast on CPU"], tokenizer)
# vecs.shape → (1, 4096), values in {-1, +1}
```

---

## Reproduce

### 1 — Environment
```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Download datasets (~2 GB)
```bash
python data/prepare.py
```
Downloads NLI triplets (550k pairs), STS-B test set, SciFact corpus + qrels.

### 3 — Smoke test (2 min)
```bash
python smoke_test.py
# All tests passed. Ready for full training.
```

### 4 — Train

```bash
# Float baseline  (~15 min on M4 Pro MPS)
python train.py --mode float  --epochs 3 --batch_size 64

# Native binary   (~15 min on M4 Pro MPS)
python train.py --mode binary --epochs 3 --batch_size 64
```

Add `--max_samples 5000` for a quick sanity run. Add `--no_mps` to force CPU.

### 5 — Benchmark quality
```bash
python benchmark.py
# → results/benchmark_results.json
```

### 6 — Benchmark retrieval at scale
```bash
python benchmark_faiss.py
# → results/retrieval_benchmark_arm64_numpy.json
# On x86 with Python ≤3.12: pip install faiss-cpu  (activates POPCNT backend)
```

---

## Project structure

```
binary-native-embeddings/
├── README.md
├── requirements.txt
├── smoke_test.py           ← run first
├── train.py                ← --mode float | binary
├── benchmark.py            ← quality metrics (STS-B, Recall@10, latency)
├── benchmark_faiss.py      ← retrieval speed at scale (10k / 100k / 1M)
├── models/
│   ├── ste.py              ← Straight-Through Estimator {-1,+1}
│   ├── float_embedder.py   ← baseline + mnrl_loss
│   └── binary_embedder.py  ← native binary + binary_contrastive_loss
├── data/
│   └── prepare.py          ← download NLI / STS-B / SciFact
└── results/
    ├── benchmark_results.json
    └── retrieval_benchmark_*.json
```

---

## Limitations & future work

- **FAISS binary benchmark** requires Python ≤ 3.12 on x86_64; not yet available for ARM64/Python 3.13
- **Larger backbones** (bert-base, MiniLM) would likely widen the quality gap further
- **Dimension sweep** (1024, 2048, 8192) to find the optimal bit budget
- **INT8 quantization** of the encoder itself for additional inference speedup

---

## License

MIT
