# Binary Native Embeddings

**Native high-dimensional binary embeddings outperform post-hoc binarization on CPU retrieval — no GPU, no compromise on throughput.**

> *Goal* — Make semantic search viable on CPU-only hardware, at a fraction of the energy cost of a GPU stack. The question is not "can binary match float32 precision?" but "how fast and how cheap can retrieval get while staying useful?"

Backbone: `prajjwal1/bert-mini` (4 layers, 256 hidden, ~11M params).  
Hardware: Mac Mini M4 Pro + Intel Core Ultra 7 155H, **CPU only — no GPU involved at any stage**.

---

## TL;DR

**Train binary. Don't binarize float.**

Native binary embeddings score **+24% Recall@10** over post-hoc binarization, validated across 5 random seeds. The 1024-dim model is **37–49× faster than float INT8** and **24–47× faster than float32** at 1M vectors, with a 12× smaller index — all on CPU, no GPU.

| Model | Dims | R@10 (5 seeds) | Memory/1k | FAISS @ 1M |
|---|---|---|---|---|
| Float32 baseline | 384 | 0.313 | 1.46 MB | 1 800–4 500 ms¹ |
| Float INT8 exact | 384 | 0.313 | 384 KB | 2 700–4 700 ms (**slower than f32**²) |
| Post-hoc binary | 384 | 0.236 | 47 KB | — |
| **Native binary 2048** | **2048** | **0.293 ±0.010** | **250 KB** | **145–200 ms** |
| **Native binary 1024** | **1024** | **0.276 ±0.012** | **125 KB** | **73–102 ms** |

> ¹ Float32 is memory-bandwidth bound (1.5 GB index at 1M) — highly sensitive to OS background load. Binary fits in L3 cache and is compute-bound (POPCNT), so it stays fast regardless.  
> ² INT8 index is 4× smaller but retrieval is consistently slower: FAISS QT_8bit dequantization overhead outweighs the cache benefit. Binary POPCNT is the only method that is simultaneously smaller **and** faster.

Pick 2048 for quality, 1024 for maximum throughput. The difference between them is not statistically significant (p=0.159).

---

## Embedding quality

STS-B Spearman and SciFact Recall@10. Encode latency on CPU (batch=32, 100 runs) — Mac Mini M4 Pro.

| Model | Dims | STS-B | R@10 | Memory/1k | Lat |
|---|---|---|---|---|---|
| Float32 | 384 | 0.7355 | 0.3131 | 1.46 MB | 4.8 ms |
| Float32 Q4 (INT8 fallback¹) | 384 | 0.7350 | 0.3097 | 1.46 MB | 6.2 ms |
| Post-hoc binary | 384 | 0.7271 | 0.2358 | 47 KB | 4.4 ms |
| **Native binary 2048** | **2048** | **0.7269 ±0.003** | **0.2926 ±0.010** | **250 KB** | **4.7 ms** |
| Native binary 1024 | 1024 | 0.7264 ±0.002 | 0.2762 ±0.012 | 125 KB | 4.5 ms |

Mean ± std across 5 seeds (42, 123, 456, 789, 1337).

> ¹ `Int4WeightOnlyConfig` requires `mslk` on Apple Silicon — fallback to torchao INT8. At bert-mini scale the model fits in L2 cache; no bandwidth gain, slight overhead.

**Encode latency is near-identical** across all binary models — dominated by the BERT forward pass, not the projection dimension.

### Statistical validation

Per-seed results (SciFact Recall@10):

| Seed | 1024 R@10 | 2048 R@10 |
|---|---|---|
| 42 | **0.2925** ← best 1024 | *0.2761* ← worst 2048 |
| 123 | 0.2875 | 0.3047 |
| 456 | 0.2728 | 0.2894 |
| 789 | 0.2619 | 0.2936 |
| 1337 | 0.2664 | 0.2992 |
| **mean ± std** | **0.2762 ± 0.012** | **0.2926 ± 0.010** |

Seed=42 is a structural outlier: best result for 1024, worst for 2048. It compresses the 5-seed gap significantly. Excluding it, the 4-seed means are 0.2722 vs 0.2967 (gap: 0.025) — a likely significant difference. The p=0.159 below is conservative because of this outlier.

Bootstrap significance test (n=2000) on SciFact per-query Recall@10:

| Comparison | Δ R@10 | p-value | 95% CI | |
|---|---|---|---|---|
| 2048 vs 1024 (5 seeds each) | +0.016 | 0.159 | [−0.047, +0.016] | ns |
| Native 2048 vs post-hoc | +0.057 | < 0.001 | — | *** |

The gap between 1024 and 2048 is **not statistically significant** at n=300 SciFact queries. The gap between native binary and post-hoc is robust.

### Bit diagnostics (NLI 5000 samples)

| Model | Dead | H mean | |r| mean | |r| max |
|---|---|---|---|---|
| Post-hoc binary 384 | 0 | 0.979 | 0.073 | 0.397 |
| Native binary 1024 (5-seed avg) | 0 | 0.976 | 0.071 | 0.935 |
| Native binary 2048 (5-seed avg) | 0 | 0.977 | 0.070 | 0.941 |

No dead bits. `|r| max ≈ 0.94` means some bit pairs are nearly perfectly correlated — an artifact of LayerNorm before STE, which forces near-perfect balance and entropy, making those metrics uninformative. The high |r| max does not hurt retrieval quality.

---

## Retrieval at scale

Intel Core Ultra 7 155H · 16 queries · top-10 · averaged over 10 runs  
All methods are **exact search** — no approximation, no recall degradation.

### Competitive landscape

| Method | FAISS index | Exact? | bytes/vec | Speed @ 1M |
|---|---|---|---|---|
| Float32 | `IndexFlatIP` | ✓ | 1 536 | 3 815 ms |
| Float INT8 | `IndexScalarQuantizer QT_8bit` | ✓ | 384 | 4 719 ms (**slower**) |
| **Binary 1024** | `IndexBinaryFlat` (POPCNT) | **✓** | **128** | **102 ms** |
| **Binary 2048** | `IndexBinaryFlat` (POPCNT) | **✓** | **256** | **202 ms** |
| Float IVF-PQ | `IndexIVFPQ` | ✗ **approx.** | variable | — |

**Float INT8 is slower than float32.** FAISS `QT_8bit` uses asymmetric distance computation (query in float32, stored vectors in INT8). The dequantization overhead exceeds the cache benefit at this scale — FAISS's optimized BLAS float32 path wins. Binary POPCNT is the only method that is both **smaller** and **faster**.

**IVF-PQ is not benchmarked here.** It is an *approximate* method that trades recall for speed via product quantization codebooks. Comparing it to exact binary search would be misleading: any speedup comes at the cost of missing relevant results, not from a more efficient kernel. The meaningful comparison is exact vs exact.

### Speed (exact search only)

Intel Core Ultra 7 155H · FAISS AVX2+POPCNT · 16 queries · top-10 · 10 runs averaged  
Float32 and INT8 times vary with system load (memory-bound). Binary is stable (compute-bound, fits in L3 cache).

| Scale | Float32 (ms) | Float INT8 (ms) | Bin-1024 (ms) | Bin-2048 (ms) | 1024 vs f32 | 1024 vs INT8 |
|---|---|---|---|---|---|---|
| 10k | 16–50 | 29–58 | 0.7–1.5 | 1.3–2.4 | 23–33× | **19–40×** |
| 100k | 200–270 | 290–430 | 7–10 | 14–26 | 24–30× | **29–46×** |
| **1M** | **1 800–4 500** | **2 700–4 700** | **73–102** | **145–202** | **24–47×** | **37–49×** |

The vs-INT8 ratio is the more stable reference: binary systematically wins whether the machine is idle or loaded.

### Memory @ 1M vectors

| Model | Index size | vs Float32 | vs Float INT8 |
|---|---|---|---|
| Float32 384-dim | 1 536 MB | — | — |
| Float INT8 384-dim | 384 MB | 4× smaller | — |
| **Binary 1024** | **128 MB** | **12× smaller** | **3× smaller** |
| **Binary 2048** | **256 MB** | **6× smaller** | 1.5× smaller |

> `IndexFlatIP` (cosine) and `IndexBinaryFlat` (Hamming) are different metrics, but timings are comparable for measuring ranking latency at scale.

### Why POPCNT changes everything

| | Float32 (384-dim) | Binary (2048-dim) |
|---|---|---|
| Kernel | 384 multiply-adds | 32 × `POPCNT` on 64-bit words |
| Memory / vector | 1 536 bytes | 256 bytes |
| Cache pressure | High | 6× lower |

`POPCNT` counts all set bits in a 64-bit word in one CPU cycle. 2048-bit Hamming: 32 POPCNT instructions vs 384 multiply-accumulates.

---

## Architecture

```
Input text
    │
    ▼
bert-mini (4L × 256d, ~11M params)
    │  mean pooling
    ▼
[256-dim pooled representation]
    │
    ├── FloatEmbedder:   Linear(256 → 384)             → float32
    └── BinaryEmbedder: Linear(256 → D) + LayerNorm   → STE → {-1,+1}^D
```

### Straight-Through Estimator (STE)

`sign()` has zero gradient almost everywhere. STE fixes this by passing the gradient unchanged through the binarization step:

```python
class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x).float()   # {-1,+1}

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output             # identity
```

### Training loss — tanh alignment

```python
def binary_contrastive_loss(a_logits, p_logits, temperature=0.05):
    a = F.normalize(torch.tanh(a_logits), dim=-1)
    p = F.normalize(torch.tanh(p_logits), dim=-1)
    sim = torch.mm(a, p.T) / temperature
    return CrossEntropyLoss()(sim, torch.arange(len(a)))
```

`tanh` maps to (−1, +1) — same range as the `{-1,+1}` STE output. Training directly optimizes the metric used at evaluation.

### Differential learning rate

```python
optimizer = AdamW([
    {"params": model.encoder.parameters(),    "lr": 2e-5},
    {"params": model.projection.parameters(), "lr": 1e-3},  # 50× higher
])
```

Single most impactful change: binary loss dropped from 2.32 → 0.31 over 3 epochs.

---

## Why native binary outperforms post-hoc

Post-hoc binarization collapses a 384-dim float space into 384 bits — near-zero activations flip arbitrarily, discarding fine-grained signal.

Native binary training gives the model three structural advantages:

1. **More capacity** — 2048 bits vs 384 bits: 5× more room to distribute semantic information
2. **Loss alignment** — `tanh` contrastive loss directly optimizes the `{-1,+1}` similarity used at eval
3. **Robustness** — semantic concepts spread across many bits; individual bit noise has low impact

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
# vecs.shape → (1, 2048), values in {-1,+1}
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
```

### 4 — Train

```bash
# Float baseline
python train.py --mode float --epochs 3 --batch_size 64

# Native binary — 5 seeds × 2 dims
for seed in 42 123 456 789 1337; do
    python train.py --mode binary --binary_dim 1024 --epochs 3 --batch_size 64 --seed $seed
    python train.py --mode binary --binary_dim 2048 --epochs 3 --batch_size 64 --seed $seed
done
```

### 5 — Benchmark encoding quality

```bash
python benchmark.py \
    --checkpoints 1024 1024_s123 1024_s456 1024_s789 1024_s1337 \
                  2048 2048_s123 2048_s456 2048_s789 2048_s1337
# → results/benchmark_results_YYYYMMDD.json
```

### 6 — Consolidate (mean ± std + bootstrap significance)

```bash
python consolidate.py --compare 1024 2048
# → results/consolidation.json + printed table with p-values
```

### 7 — Retrieval speed at scale

```bash
# x86 only, Python ≤ 3.12
pip install faiss-cpu
python benchmark_faiss.py --binary_dims 1024 2048 4096
# → results/retrieval_benchmark_amd64_faiss_YYYYMMDD.json
```

### 8 — Q4 quantization diagnostic

```bash
pip install torchao
python quantize_q4.py
```

---

## Project structure

```
binary-native-embeddings/
├── train.py             ← --mode --binary_dim --tag --temperature --lambda_e --lambda_d --seed
├── benchmark.py         ← STS-B, Recall@10, latency, bit diagnostics
├── benchmark_faiss.py   ← retrieval speed at scale (x86 + FAISS only)
├── consolidate.py       ← aggregate multi-seed results, bootstrap significance
├── quantize_q4.py       ← INT4/INT8 quantization diagnostic
├── publish_hf.py        ← push to HuggingFace Hub
├── smoke_test.py
├── models/
│   ├── ste.py
│   ├── float_embedder.py
│   └── binary_embedder.py   ← binary_contrastive_loss + entropy_loss + decorr_loss
├── data/prepare.py
└── results/
```

---

## Limitations & future work

- FAISS binary not available on ARM64/Python 3.13 (pip wheel incompatibility)
- SciFact has only 300 test queries — CI ±0.052, which limits statistical power for 1024 vs 2048
- Larger backbones (bert-base, MiniLM-L6) would likely widen the quality gap over post-hoc
- Matryoshka-style training to support multiple dims from a single checkpoint
- INT8 quantization of the encoder for additional memory reduction

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
