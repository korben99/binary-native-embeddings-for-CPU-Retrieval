# Plan d'action — Binary Native Embeddings
**Objectif** : Prouver que des embeddings binaires haute dimension entraînés nativement surpassent la binarisation post-hoc d'un modèle float, sur CPU (Mac Mini).

---

## Hypothèse centrale

> Un modèle d'embedding entraîné nativement avec une tête binaire [D_bin=4096 bits] et une loss contrastive binaire produit une meilleure similarité sémantique qu'un modèle float [D=384] binarisé post-hoc, à budget mémoire et latence CPU équivalents.

---

## Milestone 1 — Environnement & données (J1–J3)

### Setup

```bash
conda create -n binary-emb python=3.11
pip install torch sentence-transformers datasets transformers
pip install faiss-cpu beir mteb
```

### Datasets

| Dataset | Usage | Taille |
|---|---|---|
| `sentence-transformers/all-nli` | Entraînement paires similaires/dissimilaires | ~550k paires |
| `mteb/stsbenchmark-sts` | Eval similarité sémantique | 1 379 paires |
| `BeIR/scifact` | Eval retrieval | ~5k docs |

```python
from datasets import load_dataset
nli = load_dataset("sentence-transformers/all-nli", "triplet")
sts = load_dataset("mteb/stsbenchmark-sts")
```

---

## Milestone 2 — Baseline float (J4–J6)

Modèle de référence : encoder transformer léger, mean pooling, 384 dims float32.

### Architecture

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class FloatEmbedder(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-mini"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # bert-mini : 4 couches, 256 hidden, ~11M params — léger
    
    def mean_pool(self, token_embs, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embs * mask).sum(1) / mask.sum(1)
    
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask).last_hidden_state
        return self.mean_pool(out, attention_mask)  # [B, 256]
    
    def encode(self, texts, tokenizer, device="cpu"):
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            return self.forward(**enc)
```

### Loss contrastive (MultipleNegativesRankingLoss)

```python
def mnrl_loss(anchors, positives, temperature=0.05):
    # anchors, positives : [B, D] normalisés L2
    sim = torch.mm(anchors, positives.T) / temperature  # [B, B]
    labels = torch.arange(len(anchors))
    return nn.CrossEntropyLoss()(sim, labels)
```

### Entraînement baseline

```python
# ~2h sur Mac Mini M4 CPU, ou 20min RunPod A100
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
for epoch in range(3):
    for batch in dataloader:
        a = model.encode(batch["anchor"], tokenizer)
        p = model.encode(batch["positive"], tokenizer)
        a = F.normalize(a, dim=-1)
        p = F.normalize(p, dim=-1)
        loss = mnrl_loss(a, p)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Métriques à enregistrer** : STS-B Spearman, Recall@10 sur SciFact, latence encode() sur CPU.

---

## Milestone 3 — Binarisation post-hoc (J7–J8)

Binariser le modèle float entraîné. C'est le **lower bound** à battre.

```python
def binarize_posthoc(float_vec):
    # Méthode 1 : signe
    return (float_vec > 0).float()

def hamming_sim(v1, v2):
    # Similarité sur vecteurs binaires {0,1}
    return (v1 == v2).float().mean(-1)

def xor_sim_packed(v1_packed, v2_packed, D):
    # Version optimisée CPU avec bitwise XOR
    xor = v1_packed ^ v2_packed
    bits_diff = bin(int(xor)).count('1')
    return 1.0 - bits_diff / D
```

**Métriques identiques** : STS-B Spearman, Recall@10, latence CPU.

---

## Milestone 4 — Modèle binaire natif (J9–J15)

C'est le coeur du projet. Même encoder, tête binaire haute dimension, loss binaire native.

### Straight-Through Estimator (STE)

```python
class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()  # {0, 1}
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # gradient passe tel quel

binarize = BinarizeFunction.apply
```

### Architecture binaire native

```python
class BinaryEmbedder(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-mini",
                 binary_dim=4096):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 256 pour bert-mini
        
        # Projection vers espace haute dimension binaire
        self.projection = nn.Sequential(
            nn.Linear(hidden, binary_dim),
            nn.LayerNorm(binary_dim),
        )
        self.binary_dim = binary_dim
    
    def mean_pool(self, token_embs, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embs * mask).sum(1) / mask.sum(1)
    
    def forward(self, input_ids, attention_mask, binarize_output=True):
        out = self.encoder(input_ids, attention_mask).last_hidden_state
        pooled = self.mean_pool(out, attention_mask)     # [B, 256]
        projected = self.projection(pooled)               # [B, 4096]
        if binarize_output:
            return binarize(projected)                    # {0,1}^4096
        return projected                                  # float pour la loss
    
    def encode(self, texts, tokenizer, device="cpu"):
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            return self.forward(**enc, binarize_output=True)
```

### Loss binaire native

```python
def binary_contrastive_loss(anchors_logits, positives_logits, temperature=0.1):
    """
    anchors_logits, positives_logits : float pre-binarisation [B, D_bin]
    On applique sigmoid pour différentiabilité, puis similarity cosinus
    """
    a = torch.sigmoid(anchors_logits)   # [B, 4096] ∈ (0,1)
    p = torch.sigmoid(positives_logits)
    # Normalisation pour cosinus
    a_norm = F.normalize(a, dim=-1)
    p_norm = F.normalize(p, dim=-1)
    sim = torch.mm(a_norm, p_norm.T) / temperature
    labels = torch.arange(len(a))
    return nn.CrossEntropyLoss()(sim, labels)
```

### Entraînement

```python
# Identique au baseline — seul le modèle et la loss changent
# ~3h Mac Mini M4, ~30min RunPod A100
for epoch in range(3):
    for batch in dataloader:
        a_logits = model(batch["anchor"], binarize_output=False)
        p_logits = model(batch["positive"], binarize_output=False)
        loss = binary_contrastive_loss(a_logits, p_logits)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Milestone 5 — Benchmark (J16–J18)

### Tableau comparatif cible

```python
# benchmark.py — reproductible, un seul script
results = {
    "float32_384":   eval_model(float_model,    D=384,  dtype="float32"),
    "binary_384":    eval_model(float_binarized, D=384,  dtype="binary"),
    "binary_4096":   eval_model(binary_model,   D=4096, dtype="binary"),
}
```

| Modèle | Dims | Type | STS-B ↑ | Recall@10 ↑ | Mémoire ↓ | Latence CPU ↓ |
|---|---|---|---|---|---|---|
| Baseline | 384 | float32 | ~0.82 | ~0.78 | ~1.5 MB/1k vecs | ~45ms |
| Post-hoc binary | 384 | binary | ~0.74 | ~0.68 | ~48 KB/1k vecs | ~8ms |
| **Native binary** | **4096** | **binary** | **?** | **?** | **~512 KB/1k vecs** | **?** |

**L'hypothèse est validée si** native binary 4096 > post-hoc binary 384 sur STS-B et Recall@10.

### Mesure latence CPU

```python
import time, torch

def benchmark_latency(model, tokenizer, n_runs=100):
    texts = ["le chat mange du poisson"] * 32  # batch=32
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.encode(texts, tokenizer, device="cpu")
        times.append(time.perf_counter() - t0)
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "p95_ms": sorted(times)[int(0.95 * n_runs)] * 1000
    }
```

---

## Milestone 6 — GitHub & HuggingFace (J19–J21)

### Structure du repo GitHub

```
binary-native-embeddings/
├── README.md              ← benchmark visuel, quick start, hypothèse
├── train.py               ← entraînement des 3 modèles
├── benchmark.py           ← reproduction du tableau
├── models/
│   ├── float_embedder.py
│   ├── binary_embedder.py
│   └── ste.py             ← Straight-Through Estimator
├── data/
│   └── prepare.py         ← téléchargement datasets
├── results/
│   └── benchmark_results.json
└── notebooks/
    └── demo.ipynb         ← démo interactive Colab
```

### README — structure

```markdown
# Binary Native Embeddings

> Native high-dim binary embeddings outperform post-hoc binarization
> on CPU inference — no GPU required.

## Results (Mac Mini M4, CPU only)

| Model       | STS-B  | Recall@10 | Memory   | Latency |
|-------------|--------|-----------|----------|---------|
| float32 384 | 0.82   | 0.78      | 1.5 MB   | 45ms    |
| binary 384  | 0.74   | 0.68      | 48 KB    | 8ms     |
| **binary 4096** | **0.81** | **0.77** | **512 KB** | **6ms** |

## Quick start
pip install binary-native-embeddings
from bne import BinaryEmbedder
model = BinaryEmbedder.from_pretrained("username/bne-4096")
vecs = model.encode(["le chat mange du poisson"])
```

### Publication HuggingFace

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("username/bne-4096", repo_type="model")
model.save_pretrained("./bne-4096")
api.upload_folder(
    folder_path="./bne-4096",
    repo_id="username/bne-4096",
    repo_type="model"
)
```

---

## Calendrier

| Semaine | Milestones | Livrables |
|---|---|---|
| S1 (J1–J7) | Env + données + baseline float | `float_embedder.py` + métriques baseline |
| S2 (J8–J14) | Post-hoc binary + modèle natif | `binary_embedder.py` + STE |
| S3 (J15–J18) | Benchmark complet | `results/benchmark_results.json` |
| S4 (J19–J21) | GitHub + HuggingFace | Repo public + modèles publiés |

**Total : 3 semaines, ~100€ RunPod**

---

## Budget

| Poste | Coût |
|---|---|
| RunPod A100 (entraînement 3 modèles × ~30min) | ~30–50€ |
| Mac Mini M4 (benchmark CPU) | 0€ (déjà dispo) |
| Datasets open source | 0€ |
| GitHub / HuggingFace | 0€ |
| **Total** | **~50€** |

