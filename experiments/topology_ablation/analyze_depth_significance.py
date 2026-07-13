"""Step B of the routing-baseline chain — per-depth significance of the realizable gain.

Zero training. Tests whether the routing gain has statistical teeth PER STRATUM (not just a
median gradient): realizable Δ significantly > 0 at depth >= 1, and NOT at depth 0 (headwaters).

Per-depth paired Wilcoxon signed-rank (one-sided), realizable per-basin Δ pooled seeds.

Writes analysis/DEPTH_SIGNIFICANCE.md.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
DEPTH = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_depth.csv"
OUT = ROOT / "experiments/topology_ablation/analysis/DEPTH_SIGNIFICANCE.md"
SEEDS = [11, 13, 17]


def nse(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    return pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"]


def main():
    topo = pd.read_csv(DEPTH, dtype={"basin": str}).set_index("basin")
    rows = []
    for s in SEEDS:
        L = nse("L", s)
        P = nse("L_upQpred", s)
        common = L.index.intersection(P.index)
        for b in common:
            if b in topo.index:
                rows.append((int(topo.loc[b, "depth"]), float(P.loc[b] - L.loc[b])))
    df = pd.DataFrame(rows, columns=["depth", "delta"])

    md = ["# Step B — Per-Depth Significance of the Realizable Gain", "",
          f"Zero training. Realizable per-basin Δ (L+upQ_pred − L) pooled seeds {SEEDS} "
          f"(n={len(df)} basin×seed). Per-depth paired Wilcoxon signed-rank, one-sided "
          f"(H1: Δ>0). Pre-reg: `preregistration_routing_baseline_chain.md`.", "",
          "| depth | n | median Δ | Wilcoxon p (1-sided) | significant (p<0.05) |",
          "|---|---|---|---|---|"]
    sig = {}
    for d in sorted(df.depth.unique()):
        x = df[df.depth == d]["delta"].values
        if len(x) >= 6 and np.any(x != 0):
            p = wilcoxon(x, alternative="greater").pvalue
        else:
            p = np.nan
        ok = (p < 0.05) if np.isfinite(p) else False
        sig[d] = ok
        ps = f"{p:.2e}" if np.isfinite(p) else "n/a (n<6)"
        md.append(f"| {d} | {len(x)} | {np.median(x):+.4f} | {ps} | {ok} |")
    md.append("")

    head_ns = not sig.get(0, False)          # depth0 should be NOT significant
    deep_sig = sig.get(1, False) and sig.get(2, False)  # depths 1,2 should be significant
    md += ["## Pre-registered verdict", "",
           f"- depth 0 (headwaters) NOT significant: **{head_ns}**",
           f"- depths 1 AND 2 significant: **{deep_sig}**", ""]
    verdict = ("PASS — the routing gain is statistically significant specifically in the "
               "downstream strata (depth≥1) and absent at headwaters. The depth gradient has "
               "per-stratum statistical teeth, not just a median trend."
               if (head_ns and deep_sig) else
               "PARTIAL — see criteria; scope the depth claim to 'median gradient' if strata "
               "are not individually significant.")
    md += [f"**{verdict}**"]

    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> wrote {OUT}")


if __name__ == "__main__":
    main()
