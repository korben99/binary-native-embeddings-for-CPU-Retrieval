"""
Aggregate multi-seed benchmark results and compute significance tests.

Groups results by model family (strips seed suffix _sN), computes mean ± std
across seeds, and runs bootstrap significance tests between specified pairs.

Usage:
  # All JSON files from today's runs
  python consolidate.py --pattern "results/benchmark_results_*.json"

  # Explicit files
  python consolidate.py --files results/benchmark_results_20260626.json ...

  # With significance tests between dim families
  python consolidate.py --pattern "results/*.json" --compare 1024 2048

  # With additional BEIR datasets
  python consolidate.py --pattern "results/*.json" --compare 1024 2048 --datasets scifact scidocs
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def family_of(model_name: str) -> str:
    """
    'binary_native_1024_s123'  → 'binary_native_1024'
    'binary_native_2048_bs256' → 'binary_native_2048_bs256'  (non-seed tag kept)
    'float32_384'              → 'float32_384'
    """
    return re.sub(r"_s\d+$", "", model_name)


def bootstrap_diff(a: list, b: list, n_boot: int = 2000, seed: int = 0) -> dict:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    obs  = float(a.mean() - b.mean())
    rng  = np.random.default_rng(seed)
    n    = len(a)
    boot = np.array([
        a[rng.integers(0, n, n)].mean() - b[rng.integers(0, n, n)].mean()
        for _ in range(n_boot)
    ])
    p  = float((boot <= 0).mean() if obs > 0 else (boot >= 0).mean())
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {"diff": round(obs, 5), "p_value": round(p, 4), "ci_95": ci, "n_queries": n}


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── Load & group ──────────────────────────────────────────────────────────────

def load_results(files: list[str]) -> dict:
    """
    Returns {model_name: [result_dict, ...]} — one entry per seed/file.
    """
    grouped: dict[str, list] = {}
    for f in files:
        data = json.loads(Path(f).read_text())
        for model_name, r in data.items():
            if not isinstance(r, dict) or "stsb_spearman" not in r:
                continue
            grouped.setdefault(model_name, []).append(r)
    return grouped


def aggregate(runs: list[dict], datasets: list[str]) -> dict:
    """Compute mean ± std across seeds for a list of result dicts."""
    stsb   = [r["stsb_spearman"]  for r in runs if r.get("stsb_spearman") is not None]
    out    = {
        "n_seeds":     len(runs),
        "stsb_mean":   round(float(np.mean(stsb)), 4)  if stsb else None,
        "stsb_std":    round(float(np.std(stsb)),  4)  if stsb else None,
        "dims":        runs[0].get("dims"),
        "dtype":       runs[0].get("dtype"),
        "memory":      runs[0].get("memory_1k_vecs"),
        "recall":      {},
        "per_query":   {},
    }
    for ds in datasets:
        vals = []
        pqs  = []
        for r in runs:
            # support both old (scifact_recall10) and new (recall_by_dataset) format
            if ds == "scifact" and r.get("scifact_recall10") is not None and "recall_by_dataset" not in r:
                vals.append(r["scifact_recall10"])
            elif r.get("recall_by_dataset", {}).get(ds) is not None:
                vals.append(r["recall_by_dataset"][ds])
            pq = r.get("per_query_by_dataset", {}).get(ds) or \
                 (r.get("per_query") if ds == "scifact" else None)
            if pq:
                pqs.extend(pq)
        out["recall"][ds] = {
            "mean": round(float(np.mean(vals)), 4) if vals else None,
            "std":  round(float(np.std(vals)),  4) if vals else None,
        }
        out["per_query"][ds] = pqs   # concatenated across seeds for bootstrap
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main(files: list[str], datasets: list[str], compare_dims: list[int]):
    if not files:
        print("No files found.")
        return

    print(f"\nLoading {len(files)} result file(s)...")
    raw = load_results(files)

    # Group by family
    families: dict[str, list] = {}
    for model_name, runs in raw.items():
        fam = family_of(model_name)
        families.setdefault(fam, []).extend(runs)

    # Aggregate each family
    agg = {fam: aggregate(runs, datasets) for fam, runs in families.items()}

    # ── Quality table ──
    W = 14
    ds_cols = "  ".join(f"{ds[:12]:>12}" for ds in datasets)
    print(f"\n{'=' * (55 + 14 * len(datasets))}")
    print(f"  Multi-seed quality summary  ({len(files)} file(s))")
    print(f"{'Model':<30} {'Seeds':>5} {'Dims':>6}  {'STS-B':>12}  {ds_cols}")
    print(f"{'-' * (55 + 14 * len(datasets))}")

    for fam, a in sorted(agg.items()):
        stsb_s  = f"{a['stsb_mean']:.4f} ±{a['stsb_std']:.4f}" if a["stsb_mean"] else "   —"
        r10_cols = "  ".join(
            f"{a['recall'].get(ds, {}).get('mean', 0):.4f} ±{a['recall'].get(ds, {}).get('std', 0):.4f}"
            if a['recall'].get(ds, {}).get('mean') is not None else f"{'—':>12}"
            for ds in datasets
        )
        print(f"{fam:<30} {a['n_seeds']:>5} {str(a['dims']):>6}  {stsb_s:>12}  {r10_cols}")

    print(f"{'=' * (55 + 14 * len(datasets))}")

    # ── Bootstrap significance tests ──
    if compare_dims and len(compare_dims) >= 2:
        print(f"\n── Bootstrap significance (n_boot=2000) ──")
        for ds in datasets:
            print(f"\n  Dataset: {ds}")
            print(f"  {'Comparison':<45} {'Diff':>8} {'p-value':>9} {'CI 95%':>22} {'sig':>5}")
            print(f"  {'-'*90}")

            # Find families matching each dim
            dim_to_fams: dict[int, list] = {}
            for fam, a in agg.items():
                if a["dims"] in compare_dims:
                    dim_to_fams.setdefault(a["dims"], []).append(fam)

            # Pairwise between consecutive dims
            for i in range(len(compare_dims) - 1):
                da, db = compare_dims[i], compare_dims[i + 1]
                for fa in dim_to_fams.get(da, []):
                    for fb in dim_to_fams.get(db, []):
                        pq_a = agg[fa]["per_query"].get(ds, [])
                        pq_b = agg[fb]["per_query"].get(ds, [])
                        if not pq_a or not pq_b:
                            print(f"  {fa} vs {fb}: no per-query data")
                            continue
                        # Bootstrap needs same-length arrays → use min length
                        n = min(len(pq_a), len(pq_b))
                        res = bootstrap_diff(pq_a[:n], pq_b[:n])
                        ci  = f"[{res['ci_95'][0]:+.4f}, {res['ci_95'][1]:+.4f}]"
                        label = f"{fa} vs {fb}"
                        print(
                            f"  {label:<45} {res['diff']:>+8.4f} {res['p_value']:>9.4f} "
                            f"{ci:>22} {sig_stars(res['p_value']):>5}"
                        )

    # ── Save aggregated results ──
    out_path = RESULTS_DIR / "consolidation.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    # Remove per_query from saved output (too large)
    save = {fam: {k: v for k, v in a.items() if k != "per_query"} for fam, a in agg.items()}
    out_path.write_text(json.dumps(save, indent=2))
    print(f"\nAggregated results → {out_path}")
    print("\nLegend: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default=None,
                        help="Glob pattern, e.g. 'results/benchmark_results_*.json'")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                        help="Explicit list of JSON result files")
    parser.add_argument("--compare", type=int, nargs="+", default=[1024, 2048],
                        help="Dims to compare pairwise (e.g. --compare 1024 2048 4096)")
    parser.add_argument("--datasets", type=str, nargs="+", default=["scifact"],
                        help="Datasets to include in significance tests")
    args = parser.parse_args()

    files = []
    if args.files:
        files = args.files
    elif args.pattern:
        files = sorted(glob.glob(args.pattern))
    else:
        files = sorted(glob.glob(str(RESULTS_DIR / "benchmark_results_*.json")))

    main(files=files, datasets=args.datasets, compare_dims=args.compare)
