"""Step C of the routing-baseline chain — consolidated publication results table.

Zero training. Pure assembly of prior artifacts into the single Results table the paper needs:
every condition × {NSE, KGE, log-NSE} as mean±std across seeds, with paired-Δ-vs-L significance,
plus the R1/R2 routing-baseline rows. No new statistics beyond A/B and existing analysis files.

Writes analysis/PAPER_TABLE.md.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
OUT = ROOT / "experiments/topology_ablation/analysis/PAPER_TABLE.md"
SEEDS = [11, 13, 17]
CONDS = [("L", "L (baseline)"), ("L_upQ", "L+upQ (oracle)"),
         ("L_upQpred", "L+upQ_pred (realizable)"), ("L_upQshuf", "L+upQ_shuf (null)")]


def metrics_df(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    return pd.read_csv(p, dtype={"basin": str}).set_index("basin")


def lognse_series(cond, seed, eps_frac=1e-3):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_results.p"
    if not p.exists():
        return None
    res = pickle.load(open(p, "rb"))
    out = {}
    for b, d in res.items():
        xr = d["1D"]["xr"]
        o = xr["QObs(mm/d)_obs"].values.squeeze()
        s = xr["QObs(mm/d)_sim"].values.squeeze()
        m = np.isfinite(o) & np.isfinite(s)
        o, s = o[m], s[m]
        mo = np.mean(np.clip(o, 0, None)); eps = eps_frac * max(mo, 1e-6)
        lo, ls = np.log(np.clip(o, 0, None) + eps), np.log(np.clip(s, 0, None) + eps)
        den = np.sum((lo - lo.mean()) ** 2)
        out[b] = (1 - np.sum((lo - ls) ** 2) / den) if den > 0 else np.nan
    return pd.Series(out)


def cross_seed(getter):
    """mean±std of per-seed cross-basin medians."""
    meds = []
    for s in SEEDS:
        x = getter(s)
        if x is not None:
            meds.append(float(x.median()))
    return (np.mean(meds), np.std(meds)) if meds else (np.nan, np.nan)


def paired_p_vs_L(cond, metric_col):
    """pooled Wilcoxon of per-basin (cond - L) across seeds."""
    deltas = []
    for s in SEEDS:
        L = metrics_df("L", s)[metric_col]
        try:
            C = metrics_df(cond, s)[metric_col]
        except FileNotFoundError:
            continue
        common = L.index.intersection(C.index)
        deltas.extend((C.loc[common] - L.loc[common]).values)
    deltas = np.array(deltas); deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < 6 or np.all(deltas == 0):
        return np.nan
    return wilcoxon(deltas, alternative="greater").pvalue


def main():
    md = ["# Consolidated Publication Results Table", "",
          "Zero training — assembly of prior artifacts. Component 0, 183 basins, stock cudalstm, "
          "seeds [11,13,17]. All values on held-out test 2005-2008. Δ = paired per-basin vs L; "
          "p = pooled one-sided Wilcoxon vs L. Sources: SIGNIFICANCE / METRIC_HONESTY / "
          "ROUTING_BASELINE / DEPTH_SIGNIFICANCE.", ""]

    # --- main table: median metric (mean±std) per condition + Δ-vs-L p ---
    md += ["## Table 1 — median skill by condition (mean ± std across 3 seeds)", "",
           "| condition | NSE | KGE | log-NSE | ΔNSE vs L (p) |", "|---|---|---|---|---|"]
    for cond, label in CONDS:
        nse_m, nse_s = cross_seed(lambda s, c=cond: metrics_df(c, s)["NSE"])
        kge_m, kge_s = cross_seed(lambda s, c=cond: metrics_df(c, s)["KGE"])
        ln_m, ln_s = cross_seed(lambda s, c=cond: lognse_series(c, s))
        if cond == "L":
            dcell = "—"
        else:
            p = paired_p_vs_L(cond, "NSE")
            dNSE = nse_m - cross_seed(lambda s: metrics_df("L", s)["NSE"])[0]
            dcell = f"{dNSE:+.4f} (p={p:.1e})" if np.isfinite(p) else f"{dNSE:+.4f}"
        ln_cell = f"{ln_m:.3f} ± {ln_s:.3f}" if np.isfinite(ln_m) else "n/a"
        md.append(f"| {label} | {nse_m:.3f} ± {nse_s:.3f} | {kge_m:.3f} ± {kge_s:.3f} | "
                  f"{ln_cell} | {dcell} |")
    md.append("")

    # --- routing baseline rows (from ROUTING_BASELINE.md, seed 11, connected basins) ---
    md += ["## Table 2 — no-ML routing baselines vs LSTM (connected basins, seed 11)", "",
           "| predictor | median test NSE | ML? | uses upstream? |", "|---|---|---|---|",
           "| R1 — pure routing (a·upQ+b) | +0.324 | no | yes |",
           "| R2 — routing + local (a·upQ+c·L_sim+b) | +0.675 | no | yes |",
           "| L (LSTM baseline) | +0.654 | yes | no |",
           "| L+upQ_pred (realizable) | +0.686 | yes | yes |",
           "| L+upQ (oracle) | +0.717 | yes | yes |", "",
           "*The realizable LSTM beats every no-ML baseline (ML earns its complexity), but the "
           "margin over the strong R2 baseline (+0.010) is modest and honestly reported — the "
           "LSTM's real advantage is integrating upstream flow WITH local rainfall-runoff, which "
           "linear routing cannot.*", ""]

    # --- depth significance (from DEPTH_SIGNIFICANCE.md) ---
    md += ["## Table 3 — realizable gain by graph depth (pooled seeds, per-stratum Wilcoxon)", "",
           "| depth | n | median Δ | p | sig |", "|---|---|---|---|---|",
           "| 0 (headwater) | 99 | +0.002 | 0.24 | no |",
           "| 1 | 243 | +0.020 | 2.6e-9 | yes |",
           "| 2 | 153 | +0.031 | 4.7e-12 | yes |",
           "| 3 | 48 | +0.044 | 8.4e-4 | yes |",
           "| 4 | 6 | +0.015 | 0.34 | no (n=6) |", "",
           "*Routing signature with per-stratum significance: the gain is statistically present "
           "exactly where upstream flow arrives (depth≥1) and absent at headwaters. Confound-"
           "checked vs area and feature-magnitude (FEATURE_MAGNITUDE_CONFOUND.md).*", ""]

    # --- graph robustness (from GRAPH_ROBUSTNESS.md + K2_GRAPH_CHECK.md) ---
    md += ["## Table 4 — graph robustness: the gain is not a heuristic-edge artifact", "",
           "The heuristic edges over-connect (in-degree mean 4.16 / max 15 vs real confluences "
           "~2–3). Pruning to hydrography-realistic in-degree≤2 (266 edges vs 624):", "",
           "| level | metric | full graph | k=2 pruned | verdict |", "|---|---|---|---|---|",
           "| R1 signal proxy (zero-train) | median NSE | +0.325 | +0.326 | 100% retained |",
           "| LSTM realizable (seed 11) | Δ NSE (connected) | +0.034 | +0.021 (p=4e-4) | holds |",
           "| LSTM realizable (seed 11) | Δ log-NSE | — | +0.034 | holds |",
           "| LSTM oracle (seed 11) | Δ NSE (connected) | +0.046 | +0.049 (p=2e-12) | strengthens |",
           "",
           "*The routing gain lives in the physically-meaningful nearest-parent structure, not the "
           "heuristic's excess edges — confirmed at both the signal-content and trained-model level "
           "(GRAPH_ROBUSTNESS.md, K2_GRAPH_CHECK.md). k=2 model check is single-seed.*", ""]

    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> wrote {OUT}")


if __name__ == "__main__":
    main()
