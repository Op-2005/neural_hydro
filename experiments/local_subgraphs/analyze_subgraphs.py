"""Analyze the local-subgraph sweep: report the 3-seed loss-distribution
invariant (mean +/- std NSE) per (subgraph, condition), plus the key paired
contrast G+T+M - L on each subgraph.

This is the metric the professor asked for: a quantity stable across runs,
reported as a distribution across 3 seeds, that we track as we revise the
model and the data.

Reads:  runs/local_subgraphs/<subgraph>/<cond>_seed<N>/
Writes: experiments/local_subgraphs/analysis/
          invariant_table.csv        mean+/-std NSE per (subgraph, cond)
          contrasts.csv              G+T+M - L and G - L per subgraph
          INVARIANT.md               human-readable summary

Usage:
    python experiments/local_subgraphs/analyze_subgraphs.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
RUN_ROOT = ROOT / "runs" / "local_subgraphs"
OUT_DIR = Path(__file__).parent / "analysis"
MANIFEST = Path(__file__).parent / "basin_lists" / "subgraph_manifest.csv"

CONDITIONS = ["L", "G", "G_T", "G_M", "G_T_M"]


def load_nse(run_dir: Path, is_nh: bool):
    """Return {basin: NSE} for one run, or None."""
    if is_nh:
        p = run_dir / "test" / "model_epoch030" / "test_metrics.csv"
    else:
        p = run_dir / "test_metrics.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p, dtype={"basin": str})
    if "NSE" not in df.columns:
        return None
    return df.set_index("basin")["NSE"].to_dict()


def discover(subgraph):
    """Return {cond: {seed: {basin: NSE}}} for one subgraph."""
    out = {}
    sg_dir = RUN_ROOT / subgraph
    if not sg_dir.is_dir():
        return out
    for cond in CONDITIONS:
        is_nh = (cond == "L")
        for run_dir in sorted(sg_dir.glob(f"{cond}_seed*")):
            name = run_dir.name
            # exact match cond_seedN (avoid L matching nothing else; G vs G_T etc.)
            suffix = name[len(cond):]
            if not suffix.startswith("_seed"):
                continue
            try:
                seed = int(suffix.replace("_seed", ""))
            except ValueError:
                continue
            nse = load_nse(run_dir, is_nh)
            if nse:
                out.setdefault(cond, {})[seed] = nse
    return out


def invariant(cond_seed_nse):
    """mean +/- std of per-seed median NSE across seeds."""
    per_seed_medians = []
    per_seed_means = []
    for seed, basins in cond_seed_nse.items():
        vals = np.array([v for v in basins.values() if not np.isnan(v)])
        if len(vals):
            per_seed_medians.append(np.median(vals))
            per_seed_means.append(np.mean(vals))
    if not per_seed_medians:
        return None
    return {
        "n_seeds": len(per_seed_medians),
        "median_mean": float(np.mean(per_seed_medians)),
        "median_std": float(np.std(per_seed_medians)),
        "mean_mean": float(np.mean(per_seed_means)),
        "mean_std": float(np.std(per_seed_means)),
        "per_seed_median": [float(x) for x in per_seed_medians],
    }


def paired_contrast(cond_a_seed_nse, cond_b_seed_nse):
    """Paired per-basin (a - b) across shared seeds. Returns median + n."""
    deltas = []
    common_seeds = set(cond_a_seed_nse) & set(cond_b_seed_nse)
    for seed in common_seeds:
        a, b = cond_a_seed_nse[seed], cond_b_seed_nse[seed]
        for basin in set(a) & set(b):
            if not (np.isnan(a[basin]) or np.isnan(b[basin])):
                deltas.append(a[basin] - b[basin])
    if not deltas:
        return None
    arr = np.array(deltas)
    return {
        "n": len(arr), "median": float(np.median(arr)),
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "frac_positive": float((arr > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgraphs", nargs="*", default=None,
                        help="Restrict to these subgraphs; default = all in manifest")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST.exists():
        subgraphs = pd.read_csv(MANIFEST)["name"].tolist()
    else:
        subgraphs = sorted(d.name for d in RUN_ROOT.iterdir() if d.is_dir())
    if args.subgraphs:
        subgraphs = [s for s in subgraphs if s in args.subgraphs]

    inv_rows, contrast_rows = [], []
    for sg in subgraphs:
        data = discover(sg)
        if not data:
            print(f"[no data] {sg}")
            continue
        print(f"\n=== {sg} ===")
        for cond in CONDITIONS:
            if cond not in data:
                continue
            inv = invariant(data[cond])
            if inv is None:
                continue
            inv_rows.append({"subgraph": sg, "condition": cond, **inv})
            print(f"  {cond:>6}: median NSE = {inv['median_mean']:.4f} "
                   f"+/- {inv['median_std']:.4f}  ({inv['n_seeds']} seeds)")
        # Key contrasts
        for a, b, label in [("G_T_M", "L", "GTM_minus_L"),
                              ("G", "L", "G_minus_L"),
                              ("G_T_M", "G", "GTM_minus_G")]:
            if a in data and b in data:
                c = paired_contrast(data[a], data[b])
                if c:
                    contrast_rows.append({"subgraph": sg, "contrast": label, **c})
                    print(f"    {label}: median Δ = {c['median']:+.4f} "
                           f"(n={c['n']}, {c['frac_positive']*100:.0f}% positive)")

    inv_df = pd.DataFrame(inv_rows)
    inv_df.to_csv(OUT_DIR / "invariant_table.csv", index=False)
    con_df = pd.DataFrame(contrast_rows)
    con_df.to_csv(OUT_DIR / "contrasts.csv", index=False)

    # INVARIANT.md
    md = ["# Local-Subgraph Loss-Distribution Invariant", "",
          "Tracked metric: per-seed median NSE, reported as **mean ± std across 3 seeds**.",
          "Stable across runs by design; this is the quantity we watch as we revise model/data.", ""]
    if not inv_df.empty:
        md.append("## Invariant table (median NSE, mean ± std across seeds)")
        md.append("")
        md.append("| Subgraph | " + " | ".join(CONDITIONS) + " |")
        md.append("|" + "---|" * (len(CONDITIONS) + 1))
        for sg in subgraphs:
            sub = inv_df[inv_df["subgraph"] == sg]
            if sub.empty:
                continue
            cells = [sg]
            for cond in CONDITIONS:
                r = sub[sub["condition"] == cond]
                if r.empty:
                    cells.append("—")
                else:
                    cells.append(f"{r.iloc[0]['median_mean']:.3f}±{r.iloc[0]['median_std']:.3f}")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    if not con_df.empty:
        md.append("## Key contrasts (paired per-basin median Δ NSE)")
        md.append("")
        md.append("| Subgraph | G+T+M − L | G − L | G+T+M − G |")
        md.append("|---|---|---|---|")
        for sg in subgraphs:
            sub = con_df[con_df["subgraph"] == sg]
            if sub.empty:
                continue
            def g(lbl):
                r = sub[sub["contrast"] == lbl]
                return f"{r.iloc[0]['median']:+.3f}" if not r.empty else "—"
            md.append(f"| {sg} | {g('GTM_minus_L')} | {g('G_minus_L')} | {g('GTM_minus_G')} |")
        md.append("")
        md.append("**Reading:** the paper claim ('graph features beat standard LSTM') is "
                  "supported on a subgraph iff `G+T+M − L > 0`. Watch whether this turns "
                  "positive on the smaller, more locally-coherent subgraphs.")
    with open(OUT_DIR / "INVARIANT.md", "w") as f:
        f.write("\n".join(md))

    print(f"\nWrote invariant table + contrasts + INVARIANT.md to {OUT_DIR}")


if __name__ == "__main__":
    main()
