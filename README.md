# Binary Native Embeddings

**Native high-dimensional binary embeddings outperform post-hoc binarization on CPU retrieval — no GPU required.**

> *Hypothesis* — A transformer trained natively with a binary head and a contrastive binary loss produces better semantic retrieval than the same transformer binarized post-hoc, at lower memory cost than float32.

Backbone: `prajjwal1/bert-mini` (4 layers, 256 hidden, ~11M params).  
Hardware: Mac Mini M4 Pro + Intel Core Ultra 7 155H, **CPU only**.

---

## TL;DR

| Model | Dims | STS-B | Recall@10 | Memory/1k | Retrieval @ 1M (FAISS) |
|---|---|---|---|---|---|
| Float baseline | 384 | 0.736 | 0.313 | 1.46 MB | 3 601 ms |
| Post-hoc binary | 384 | 0.727 | 0.236 | 48 KB | — |
| **Native binary** | **2048** | **0.729** | **0.276** | **250 KB** | **292 ms (12x faster)** |
| **Native binary** | **4096** | **0.728** | **0.296** | **500 KB** | **596 ms (6x faster)** |

**Hypothesis validated**: native binary beats post-hoc on Recall@10 at every dimension (+17% at 2048, +25% at 4096), while retrieving 6–12× faster than float32 at scale.

---

## Results — Embedding quality

Evaluated on STS-B Spearman correlation and SciFact Recall@10.  
Encode latency measured on CPU (batch=32, 100 runs) — Mac Mini M4 Pro.

| Model | Dims | BS | STS-B ↑ | R@10 ↑ | Memory/1k ↓ | Latency |
|---|---|---|---|---|---|---|
| Float baseline | 384 | 64 | 0.7355 | 0.3131 | 1.46 MB | 6.0 ms |
| Float Q4 (INT8 fallback¹) | 384 | 64 | 0.7350 | 0.3097 | 1.46 MB | 6.9 ms |
| Post-hoc binary | 384 | 64 | 0.7271 | 0.2358 | 47 KB | 5.2 ms |
| **Native binary** | **1024** | **64** | **0.7256** | **0.2925** | **125 KB** | **5.5 ms** |
| Native binary | 2048 | 64 | 0.7293 | 0.2761 | 250 KB | 6.3 ms |
| Native binary | 4096 | 64 | 0.7275 | 0.2958 | 500 KB | 7.4 ms |

> ¹ `Int4WeightOnlyConfig` requires `mslk` on Apple Silicon — fallback to torchao INT8. Float Q4 does not change index memory (output remains float32 384-dim); gain is in encoding weight bandwidth. At bert-mini scale the model fits in L2 cache so the fallback shows no speedup.

**Encode latency is near-identical** across binary models — dominated by the BERT forward pass, not the vector dimension.

---

## Ongoing experiments — 2048-dim quality improvement

Goal: close the Recall@10 gap between native-2048 (0.2761) and native-4096 (0.2958).  
Three levers tested independently, then combined.

| Checkpoint | Dim | BS | T | λe | λd | STS-B | R@10 | Δ R@10 |
|---|---|---|---|---|---|---|---|---|
| `binary_embedder_2048` ← baseline | 2048 | 64 | 0.05 | 0 | 0 | 0.7293 | 0.2761 | — |
| `binary_embedder_1024` | 1024 | 64 | 0.05 | 0 | 0 | — | — | — |
| `binary_embedder_2048_bs256` | 2048 | 256 | 0.05 | 0 | 0 | — | — | — |
| `binary_embedder_2048_t002` | 2048 | 64 | 0.02 | 0 | 0 | — | — | — |
| `binary_embedder_2048_t010` | 2048 | 64 | 0.10 | 0 | 0 | — | — | — |
| `binary_embedder_2048_reg` | 2048 | 64 | 0.05 | 0.1 | 0.01 | — | — | — |
| `binary_embedder_2048_bs256_reg` | 2048 | 256 | 0.05 | 0.1 | 0.01 | — | — | — |
| `binary_embedder_4096` ← target | 4096 | 64 | 0.05 | 0 | 0 | 0.7275 | 0.2958 | +0.0197 |

---

## Results — Retrieval at scale

Intel Core Ultra 7 155H · FAISS `IndexBinaryFlat` (AVX2 + POPCNT) vs `IndexFlatIP`  
16 queries · top-10 · averaged over 10 runs

| Scale | Float (ms) | Bin-2048 (ms) | Bin-4096 (ms) | 2048 vs Float | 4096 vs Float |
|---|---|---|---|---|---|
| 10k | 47.9 | 2.2 | 4.4 | **21.8x faster** | **10.8x faster** |
| 100k | 254.3 | 24.7 | 52.9 | **10.3x faster** | **4.8x faster** |
| **1M** | **3 601** | **293** | **596** | **12.3x faster** | **6.0x faster** |

| Model | Memory @ 1M vecs | vs Float |
|---|---|---|
| Float 384 | 1 536 MB | — |
| Binary 2048 | 256 MB | **6× smaller** |
| Binary 4096 | 512 MB | **3× smaller** |

**2048-dim is the sweet spot**: 6× smaller index, 12× faster retrieval at 1M vectors, +17% Recall@10 over post-hoc — all on CPU, no GPU.

> **Note:** float uses `IndexFlatIP` (cosine similarity) and binary uses `IndexBinaryFlat` (Hamming distance) — different metrics, but timings are comparable for measuring ranking latency at scale.

### Why POPCNT changes everything

| | Float32 (384-dim) | Binary (2048-dim, POPCNT) |
|---|---|---|
| Kernel | 384 multiply-adds | 32 × `POPCNT` on 64-bit words |
| Memory read / vector | 1 536 bytes | 256 bytes |
| Cache pressure | High | 6× lower |

`POPCNT` counts all set bits in a 64-bit word in a single CPU cycle. For 2048-bit vectors: 32 POPCNT instructions vs 384 multiply-accumulates, compounded by 6× better cache utilization.

---

## Architecture

```
Input text
    │
    ▼
bert-mini (4L × 256d, ~11M params, shared backbone)
    │  mean pooling
    ▼
[256-dim pooled representation]
    │
    ├── FloatEmbedder:    Linear(256 → 384)              → float32
    │
    ├── BinaryEmbedder:  Linear(256 → 2048) + LayerNorm → STE → {-1,+1}²⁰⁴⁸
    │
    └── BinaryEmbedder:  Linear(256 → 4096) + LayerNorm → STE → {-1,+1}⁴⁰⁹⁶
```

### Straight-Through Estimator (STE)

`sign()` has zero gradient almost everywhere. STE fixes this by passing the gradient unchanged through the binarization step:

```python
class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x).float()   # {-1, +1} — discrete in forward

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output             # identity in backward
```

### Training loss — tanh alignment

```python
def binary_contrastive_loss(a_logits, p_logits, temperature=0.05):
    # tanh maps to (-1, +1) — same range as {-1,+1} STE output
    # training directly optimizes the metric used at evaluation
    a = F.normalize(torch.tanh(a_logits), dim=-1)
    p = F.normalize(torch.tanh(p_logits), dim=-1)
    sim = torch.mm(a, p.T) / temperature
    return CrossEntropyLoss()(sim, torch.arange(len(a)))
```

Using `tanh` instead of `sigmoid` aligns the continuous approximation with the `{-1,+1}` output — the model optimizes what the evaluation metric measures.

### Differential learning rate

The projection head is randomly initialized; the encoder starts from pretrained BERT weights. Using the same LR for both leads to slow convergence of the projection:

```python
optimizer = AdamW([
    {"params": model.encoder.parameters(),    "lr": 2e-5},
    {"params": model.projection.parameters(), "lr": 1e-3},  # 50× higher
])
```

This was the single most impactful fix: binary loss dropped from 2.32 → 0.31 over 3 epochs.

---

## Why native binary outperforms post-hoc

Post-hoc binarization collapses a 384-dim float space into 384 bits, discarding sign information in an uncontrolled way — near-zero activations flip arbitrarily.

Native binary training gives the model three advantages:

1. **More dimensions** — 2048 bits vs 384 bits: 5× more capacity to distribute semantic information
2. **Loss alignment** — `tanh` contrastive loss directly optimizes `{-1,+1}` cosine similarity, the same metric used at eval
3. **Redundancy** — semantically related concepts are encoded across multiple bits, making the representation robust to individual bit noise

---

## Quick start

```bash
git clone https://github.com/korben99/binary-native-embeddings-for-CPU-Retrieval
cd binary-native-embeddings
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```python
import torch
from transformers import BertTokenizer
from models.binary_embedder import BinaryEmbedder

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
model = BinaryEmbedder(binary_dim=2048)
model.load_state_dict(torch.load("checkpoints/binary_embedder_2048.pt", map_location="cpu"))
model.eval()

vecs = model.encode(["binary embeddings are fast on CPU"], tokenizer)
# vecs.shape → (1, 2048), values in {-1, +1}
```

---

## Reproduce

### 1 — Environment
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Download datasets (~2 GB)
```bash
python data/prepare.py
# NLI 550k pairs · STS-B test set · SciFact corpus + qrels
```

### 3 — Smoke test (2 min)
```bash
python smoke_test.py
# All tests passed. Ready for full training.
```

### 4 — Train baselines

```bash
# Float baseline (~15 min M4 Pro)
python train.py --mode float --epochs 3 --batch_size 64

# Native binary — standard dims
python train.py --mode binary --epochs 3 --batch_size 64 --binary_dim 1024
python train.py --mode binary --epochs 3 --batch_size 64 --binary_dim 2048
python train.py --mode binary --epochs 3 --batch_size 64 --binary_dim 4096
```

MPS (Apple Silicon) is used automatically. Add `--no_mps` to force CPU.

### 5 — Train experiments (2048-dim quality sweep)

Each run produces a distinct checkpoint — existing models are never overwritten.

```bash
# Lever 1 — batch size (more hard negatives)
python train.py --mode binary --binary_dim 2048 --batch_size 256 --epochs 3 \
    --tag bs256

# Lever 2 — temperature sweep
python train.py --mode binary --binary_dim 2048 --batch_size 64 --epochs 3 \
    --temperature 0.02 --tag t002
python train.py --mode binary --binary_dim 2048 --batch_size 64 --epochs 3 \
    --temperature 0.10 --tag t010

# Lever 3 — entropy + decorrelation regularization
python train.py --mode binary --binary_dim 2048 --batch_size 64 --epochs 3 \
    --lambda_e 0.1 --lambda_d 0.01 --tag reg

# Combined best levers
python train.py --mode binary --binary_dim 2048 --batch_size 256 --epochs 3 \
    --lambda_e 0.1 --lambda_d 0.01 --tag bs256_reg
```

### 6 — Benchmark encoding quality

```bash
python benchmark.py --binary_dims 1024 2048 4096
# → results/benchmark_results_YYYYMMDD.json
```

To benchmark a specific experiment checkpoint, load it manually — or extend
`benchmark.py --binary_dims` once experiment checkpoints are named in configs.

### 7 — Benchmark retrieval speed at scale

```bash
# x86 only, Python ≤ 3.12, FAISS AVX2+POPCNT
pip install faiss-cpu
python benchmark_faiss.py --binary_dims 1024 2048 4096
# → results/retrieval_benchmark_amd64_faiss_YYYYMMDD.json
```

### 8 — Q4 quantization diagnostic

```bash
pip install torchao
python quantize_q4.py
# reports: latency, STS-B delta, model weight memory
```

---

## Project structure

```
binary-native-embeddings/
├── README.md
├── requirements.txt
├── smoke_test.py              ← run first
├── train.py                   ← --mode --binary_dim --tag --temperature --lambda_e --lambda_d
├── benchmark.py               ← encoding quality: STS-B, Recall@10, latency, bit diagnostics
├── benchmark_faiss.py         ← retrieval speed at scale (x86 + FAISS only)
├── quantize_q4.py             ← INT4/INT8 quantization diagnostic
├── publish_hf.py              ← push to HuggingFace Hub
├── models/
│   ├── ste.py                 ← Straight-Through Estimator {-1,+1}
│   ├── float_embedder.py      ← baseline + mnrl_loss
│   └── binary_embedder.py     ← binary_contrastive_loss + entropy_loss + decorr_loss
├── data/
│   └── prepare.py
└── results/
    ├── benchmark_results_YYYYMMDD.json
    └── retrieval_benchmark_*_YYYYMMDD.json
```

---

## Limitations & future work

- FAISS binary not yet available for ARM64/Python 3.13 (pip wheel incompatibility)
- Larger backbones (bert-base, MiniLM-L6) would likely widen the quality gap
- Dimension sweep below 2048 (512, 1024) to find the minimum viable bit budget
- INT8 quantization of the encoder itself for additional memory reduction
- Matryoshka-style training to support multiple dims from a single model

---

## Models on HuggingFace

- [`korben99/bne-float-384`](https://huggingface.co/korben99/bne-float-384) — float32 baseline
- [`korben99/bne-binary-2048`](https://huggingface.co/korben99/bne-binary-2048) — **recommended**
- [`korben99/bne-binary-4096`](https://huggingface.co/korben99/bne-binary-4096)

---

## Discussion

Feedback and questions on the [HuggingFace forum thread](https://discuss.huggingface.co/t/native-binary-embeddings-experiment-curious-about-your-thoughts/177107).

---

## License

MIT
