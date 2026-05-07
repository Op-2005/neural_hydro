"""5-condition factorial analysis on Component 0 (183 basins).

Reads the 15 runs from `runs/5cond_factorial/` (5 conditions × 3 seeds), recomputes
NSE / KGE / log-NSE consistently from raw predictions for every condition, then
produces:

  experiments/analysis_outputs/5cond_component0/
    summary.json                  cross-seed medians + bootstrap 95% CIs per (condition, metric)
    per_basin_long.csv            one row per (condition, seed, basin, metric)
    per_basin_wide.csv            wide table: per-basin NSE/KGE/logNSE for all 5 conditions
    contrasts.csv                 6 pairwise contrasts + interaction, per metric
    contrasts.png                 per-basin ΔNSE histograms for the 6 contrasts
    nse_by_depth.png              depth-stratified median NSE per condition
    nse_by_area.png               area-tercile-stratified median NSE per condition
    outlier_trimmed.csv           cross-seed median after dropping bottom-5% NSE basins
    RESULTS.md                    slideshow-ready markdown summary

Usage:
    python experiments/analysis/compare_5conditions.py [--seeds 11 13 17] [--n-boot 2000]

If a condition is missing or has fewer seeds than requested, the script reports
what it found and proceeds with what's available.
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
RUN_ROOT = ROOT / "runs" / "5cond_factorial"
OUT_DIR = ROOT / "experiments" / "analysis_outputs" / "5cond_component0"
DEPTH_FILE = ROOT / "topology_analysis" / "phase1_network_discovery" / "outputs" / "component0_depth.csv"

# Condition definitions: (cond_id, run-folder-prefix, source)
# source = "nh" → predictions from NH pickle; "graph" → from test_predictions.csv
CONDITIONS = [
    ("L",     "L",     "nh"),
    ("G",     "G",     "graph"),
    ("G+T",   "G_T",   "graph"),
    ("G+M",   "G_M",   "graph"),
    ("G+T+M", "G_T_M", "graph"),
]
COND_IDS = [c[0] for c in CONDITIONS]

PAIRWISE_CONTRASTS = [
    ("L_minus_G",       "L",     "G"),
    ("GT_minus_G",      "G+T",   "G"),
    ("GM_minus_G",      "G+M",   "G"),
    ("GTM_minus_GT",    "G+T+M", "G+T"),
    ("GTM_minus_GM",    "G+T+M", "G+M"),
    ("GTM_minus_G",     "G+T+M", "G"),
]
INTERACTION = ("interaction_TxM", "(G+T+M) − (G+T) − (G+M) + G")


# ---------------------------------------------------------------------------
# Metric definitions (computed identically for every condition)
# ---------------------------------------------------------------------------
def nse(obs, pred):
    if len(obs) < 2 or obs.std() == 0:
        return float("nan")
    return float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))


def kge(obs, pred):
    if len(obs) < 2 or obs.std() == 0 or pred.std() == 0:
        return float("nan")
    r = float(np.corrcoef(obs, pred)[0, 1])
    alpha = float(pred.std() / obs.std())
    obs_mean = float(obs.mean())
    if obs_mean == 0:
        return float("nan")
    beta = float(pred.mean() / obs_mean)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def log_nse(obs, pred, eps=None):
    """log-NSE with ε = 1% of mean obs, per Pushpalatha et al. (2012)."""
    if len(obs) < 2:
        return float("nan")
    obs_mean = float(obs.mean())
    if obs_mean <= 0:
        return float("nan")
    if eps is None:
        eps = 0.01 * obs_mean
    lo = np.log(np.clip(obs, 0, None) + eps)
    lp = np.log(np.clip(pred, 0, None) + eps)
    if lo.std() == 0:
        return float("nan")
    return float(1 - np.sum((lo - lp) ** 2) / np.sum((lo - lo.mean()) ** 2))


def compute_all_metrics(obs, pred):
    return {"NSE": nse(obs, pred), "KGE": kge(obs, pred), "logNSE": log_nse(obs, pred)}


# ---------------------------------------------------------------------------
# Predictions loaders
# ---------------------------------------------------------------------------
def load_graph_predictions(run_dir: Path):
    """Load test_predictions.csv (long format) → {basin: (obs, pred)}."""
    csv_path = run_dir / "test_predictions.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path, dtype={"basin": str})
    out = {}
    for basin, sub in df.groupby("basin"):
        out[basin] = (sub["obs"].to_numpy(), sub["pred"].to_numpy())
    return out


def load_nh_predictions(run_dir: Path, epoch: int = 30):
    """Load NH's test_results.p pickle → {basin: (obs, pred)}.

    NH writes a nested dict: {basin: {freq: {'xr': xarray.Dataset(...)}}}, with
    data variables `QObs(mm/d)_obs` and `QObs(mm/d)_sim`.
    """
    p = run_dir / "test" / f"model_epoch{epoch:03d}" / "test_results.p"
    if not p.is_file():
        return None
    with open(p, "rb") as f:
        results = pickle.load(f)
    out = {}
    for basin, by_freq in results.items():
        # Take the only / first frequency; QObs naming is fixed.
        freq_key = next(iter(by_freq.keys()))
        ds = by_freq[freq_key]["xr"]
        # The NH variable names follow `<target>_obs` / `<target>_sim`.
        obs_var = next(v for v in ds.data_vars if v.endswith("_obs"))
        sim_var = next(v for v in ds.data_vars if v.endswith("_sim"))
        obs = ds[obs_var].values.flatten()
        sim = ds[sim_var].values.flatten()
        mask = ~np.isnan(obs) & ~np.isnan(sim)
        out[str(basin)] = (obs[mask].astype(np.float64), sim[mask].astype(np.float64))
    return out


def fallback_metrics_csv(run_dir: Path, source: str):
    """If raw predictions aren't available, fall back to test_metrics.csv (NSE only)."""
    if source == "nh":
        p = run_dir / "test" / "model_epoch030" / "test_metrics.csv"
    else:
        p = run_dir / "test_metrics.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p, dtype={"basin": str})
    out = {}
    for _, row in df.iterrows():
        out[row["basin"]] = {
            "NSE": float(row.get("NSE", np.nan)),
            "KGE": float(row["KGE"]) if "KGE" in row and pd.notna(row.get("KGE")) else float("nan"),
            "logNSE": float("nan"),
        }
    return out


def load_run(run_dir: Path, source: str):
    """Return {basin: {NSE, KGE, logNSE}} for one run, computed from raw predictions
    when available, otherwise from test_metrics.csv (logNSE → NaN)."""
    obs_pred = load_nh_predictions(run_dir) if source == "nh" else load_graph_predictions(run_dir)
    if obs_pred is None:
        return fallback_metrics_csv(run_dir, source)
    out = {}
    for basin, (obs, pred) in obs_pred.items():
        out[basin] = compute_all_metrics(obs, pred)
    return out


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------
def bootstrap_median_ci(values, n_boot=2000, alpha=0.05, rng=None):
    values = np.asarray([v for v in values if not np.isnan(v)])
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = rng or np.random.default_rng(0)
    boots = np.array([np.median(rng.choice(values, size=len(values), replace=True))
                       for _ in range(n_boot)])
    return float(np.median(values)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Discovery: walk RUN_ROOT for completed runs
# ---------------------------------------------------------------------------
def discover_runs(seeds_filter=None):
    """Return {cond_id: {seed: per_basin_metrics_dict}}."""
    out = {c: {} for c in COND_IDS}
    for cond_id, sub, source in CONDITIONS:
        for run_dir in sorted(RUN_ROOT.glob(f"{sub}_seed*")):
            try:
                seed = int(run_dir.name.split("seed")[-1])
            except ValueError:
                continue
            if seeds_filter is not None and seed not in seeds_filter:
                continue
            metrics = load_run(run_dir, source)
            if metrics:
                out[cond_id][seed] = metrics
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                         help="Restrict to these seeds; default = all found")
    parser.add_argument("--n-boot", type=int, default=2000,
                         help="Bootstrap iterations for cross-seed median CIs")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    seeds_filter = set(args.seeds) if args.seeds else None
    results = discover_runs(seeds_filter)

    print(f"Run discovery from {RUN_ROOT}:")
    for cond_id in COND_IDS:
        seeds = sorted(results[cond_id].keys())
        print(f"  {cond_id:>6}: {len(seeds)} seed(s) found: {seeds}")
    print()

    # ---- Long-format table: (condition, seed, basin, metric_name, value) ----
    long_rows = []
    for cond_id, by_seed in results.items():
        for seed, basin_metrics in by_seed.items():
            for basin, mvals in basin_metrics.items():
                for metric_name, v in mvals.items():
                    long_rows.append({"condition": cond_id, "seed": seed,
                                       "basin": basin, "metric": metric_name, "value": v})
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT_DIR / "per_basin_long.csv", index=False)

    # ---- Wide table indexed by basin: per-condition NSE/KGE/logNSE averaged across seeds ----
    # For each basin × condition, take the cross-seed mean (only basins present in every seed).
    wide_rows = {}
    for cond_id, by_seed in results.items():
        if not by_seed:
            continue
        all_basins = set.intersection(*[set(d.keys()) for d in by_seed.values()])
        for basin in all_basins:
            for metric_name in ["NSE", "KGE", "logNSE"]:
                vals = [by_seed[s][basin].get(metric_name, np.nan) for s in by_seed]
                vals = [v for v in vals if not np.isnan(v)]
                if not vals:
                    continue
                wide_rows.setdefault(basin, {})[f"{cond_id}__{metric_name}"] = float(np.mean(vals))
    wide_df = pd.DataFrame.from_dict(wide_rows, orient="index").reset_index().rename(columns={"index": "basin"})
    wide_df.to_csv(OUT_DIR / "per_basin_wide.csv", index=False)

    # ---- Per-condition cross-seed median + bootstrap CI for each metric ----
    summary = {"conditions": {}}
    for cond_id, by_seed in results.items():
        if not by_seed:
            continue
        per_seed_medians = {}  # {metric: [median across basins for each seed]}
        for metric_name in ["NSE", "KGE", "logNSE"]:
            per_seed_medians[metric_name] = []
            for seed, basin_metrics in by_seed.items():
                vals = [m[metric_name] for m in basin_metrics.values()
                         if not np.isnan(m.get(metric_name, np.nan))]
                if vals:
                    per_seed_medians[metric_name].append(np.median(vals))
        summary["conditions"][cond_id] = {
            "n_seeds": len(by_seed),
            "seeds": sorted(by_seed.keys()),
        }
        for metric_name, ms in per_seed_medians.items():
            med, lo, hi = bootstrap_median_ci(ms, n_boot=args.n_boot, rng=rng)
            summary["conditions"][cond_id][metric_name] = {
                "per_seed_medians": [float(x) for x in ms],
                "cross_seed_median": med,
                "bootstrap_95ci": [lo, hi],
                "cross_seed_std": float(np.std(ms)) if len(ms) > 1 else 0.0,
            }

    # ---- Pairwise contrasts: per-basin Δmetric (paired across same seed × basin) ----
    # For each contrast and metric, collect the per-(basin, seed) Δ values.
    contrasts_summary = {}
    contrasts_long_rows = []
    for contrast_name, cond_a, cond_b in PAIRWISE_CONTRASTS:
        if cond_a not in results or cond_b not in results:
            continue
        if not results[cond_a] or not results[cond_b]:
            continue
        common_seeds = set(results[cond_a].keys()) & set(results[cond_b].keys())
        for metric_name in ["NSE", "KGE", "logNSE"]:
            deltas = []
            for seed in common_seeds:
                a_metrics = results[cond_a][seed]
                b_metrics = results[cond_b][seed]
                common_basins = set(a_metrics.keys()) & set(b_metrics.keys())
                for basin in common_basins:
                    a_v = a_metrics[basin].get(metric_name, np.nan)
                    b_v = b_metrics[basin].get(metric_name, np.nan)
                    if np.isnan(a_v) or np.isnan(b_v):
                        continue
                    d = a_v - b_v
                    deltas.append(d)
                    contrasts_long_rows.append({
                        "contrast": contrast_name, "metric": metric_name,
                        "seed": seed, "basin": basin, "delta": float(d),
                    })
            if deltas:
                arr = np.array(deltas)
                med, lo, hi = bootstrap_median_ci(arr, n_boot=args.n_boot, rng=rng)
                contrasts_summary.setdefault(contrast_name, {})[metric_name] = {
                    "n": int(len(arr)),
                    "median": med, "bootstrap_95ci": [lo, hi],
                    "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                    "n_strongly_positive": int((arr > 0.05).sum()),
                    "n_strongly_negative": int((arr < -0.05).sum()),
                }

    # ---- Interaction term: ((G+T+M) − (G+T)) − ((G+M) − G), paired per basin × seed ----
    if all(c in results and results[c] for c in ["G+T+M", "G+T", "G+M", "G"]):
        common_seeds = (set(results["G+T+M"].keys()) & set(results["G+T"].keys())
                         & set(results["G+M"].keys()) & set(results["G"].keys()))
        for metric_name in ["NSE", "KGE", "logNSE"]:
            inter_vals = []
            for seed in common_seeds:
                a, b, c, d = (results["G+T+M"][seed], results["G+T"][seed],
                                results["G+M"][seed], results["G"][seed])
                common_basins = set(a) & set(b) & set(c) & set(d)
                for basin in common_basins:
                    va = a[basin].get(metric_name, np.nan)
                    vb = b[basin].get(metric_name, np.nan)
                    vc = c[basin].get(metric_name, np.nan)
                    vd = d[basin].get(metric_name, np.nan)
                    if any(np.isnan(x) for x in (va, vb, vc, vd)):
                        continue
                    inter = va - vb - vc + vd
                    inter_vals.append(inter)
                    contrasts_long_rows.append({
                        "contrast": "interaction_TxM", "metric": metric_name,
                        "seed": seed, "basin": basin, "delta": float(inter),
                    })
            if inter_vals:
                arr = np.array(inter_vals)
                med, lo, hi = bootstrap_median_ci(arr, n_boot=args.n_boot, rng=rng)
                contrasts_summary.setdefault("interaction_TxM", {})[metric_name] = {
                    "n": int(len(arr)),
                    "median": med, "bootstrap_95ci": [lo, hi],
                    "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                    "n_strongly_positive": int((arr > 0.05).sum()),
                    "n_strongly_negative": int((arr < -0.05).sum()),
                }

    summary["contrasts"] = contrasts_summary
    if contrasts_long_rows:
        pd.DataFrame(contrasts_long_rows).to_csv(OUT_DIR / "contrasts_long.csv", index=False)

    # ---- Outlier-trimmed cross-seed median (drop bottom 5% of NSE per seed) ----
    trimmed = {}
    for cond_id, by_seed in results.items():
        if not by_seed:
            continue
        per_seed = []
        for seed, basin_metrics in by_seed.items():
            nses = np.array([m["NSE"] for m in basin_metrics.values()
                              if not np.isnan(m.get("NSE", np.nan))])
            if len(nses) == 0:
                continue
            cutoff = np.quantile(nses, 0.05)
            kept = nses[nses >= cutoff]
            per_seed.append({
                "seed": seed, "condition": cond_id,
                "n_total": int(len(nses)), "n_kept": int(len(kept)),
                "median_full": float(np.median(nses)),
                "median_trimmed": float(np.median(kept)),
                "mean_trimmed": float(np.mean(kept)),
            })
        trimmed[cond_id] = per_seed
    pd.DataFrame([row for rows in trimmed.values() for row in rows]).to_csv(
        OUT_DIR / "outlier_trimmed.csv", index=False)
    summary["outlier_trimmed_5pct"] = {
        cond_id: {
            "cross_seed_median_full": float(np.median([r["median_full"] for r in rows])),
            "cross_seed_median_trimmed": float(np.median([r["median_trimmed"] for r in rows])),
        } for cond_id, rows in trimmed.items() if rows
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # =====================================================================
    # Plots
    # =====================================================================
    # Pairwise contrast distributions (NSE only — the publication-relevant one)
    contrast_names = [c[0] for c in PAIRWISE_CONTRASTS] + ["interaction_TxM"]
    contrast_label = {**{c[0]: c[0] for c in PAIRWISE_CONTRASTS},
                       "interaction_TxM": "interaction (T×M)"}
    plot_contrasts = [c for c in contrast_names if c in contrasts_summary]
    if plot_contrasts and contrasts_long_rows:
        contrasts_df = pd.DataFrame(contrasts_long_rows)
        n = len(plot_contrasts)
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4), squeeze=False)
        for ax, name in zip(axes[0], plot_contrasts):
            sub = contrasts_df[(contrasts_df["contrast"] == name) & (contrasts_df["metric"] == "NSE")]
            v = sub["delta"].dropna().to_numpy()
            if len(v) == 0:
                ax.set_visible(False)
                continue
            ax.hist(v, bins=40, color="C0", alpha=0.75, edgecolor="white", lw=0.4)
            ax.axvline(0, color="k", lw=0.7, ls="--")
            ax.axvline(np.median(v), color="C3", lw=1.5, label=f"median {np.median(v):+.3f}")
            ax.set_title(f"{contrast_label[name]}\nn={len(v)}, mean {v.mean():+.3f}")
            ax.set_xlabel("Δ NSE")
            ax.set_ylabel("# basin × seed")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Per-basin paired ΔNSE — 5-condition factorial (Component 0)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "contrasts.png", dpi=140)
        plt.close(fig)

    # Depth-stratified medians
    if DEPTH_FILE.exists():
        depth_df = pd.read_csv(DEPTH_FILE, dtype={"basin": str})
        depth_lookup = depth_df.set_index("basin")["depth"].to_dict()
        area_lookup = depth_df.set_index("basin")["area_km2"].to_dict() if "area_km2" in depth_df.columns else {}
        depths = sorted(set(int(d) for d in depth_lookup.values()
                              if not pd.isna(d)))

        # NSE by depth
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        depth_rows = []
        for cond_id, by_seed in results.items():
            if not by_seed:
                continue
            medians = []
            for d in depths:
                vals = []
                for seed, basin_metrics in by_seed.items():
                    for basin, m in basin_metrics.items():
                        if depth_lookup.get(basin) == d and not np.isnan(m.get("NSE", np.nan)):
                            vals.append(m["NSE"])
                medians.append(np.median(vals) if vals else np.nan)
                depth_rows.append({"condition": cond_id, "depth": d,
                                    "n_basin_seed": len(vals),
                                    "median_NSE": float(np.median(vals)) if vals else None})
            ax.plot(depths, medians, "o-", lw=1.6, label=cond_id)
        ax.set_xlabel("graph depth (0 = headwater leaf)")
        ax.set_ylabel("median NSE across basins × seeds")
        ax.set_title("Depth-stratified median NSE — 5-condition factorial")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "nse_by_depth.png", dpi=140)
        plt.close(fig)
        pd.DataFrame(depth_rows).to_csv(OUT_DIR / "depth_stratified.csv", index=False)

        # NSE by area tercile
        if area_lookup:
            areas = np.array([area_lookup[b] for b in depth_lookup if b in area_lookup
                                and not pd.isna(area_lookup[b])])
            if len(areas) > 3:
                q33, q67 = np.quantile(areas, [1/3, 2/3])
                def area_bin(b):
                    a = area_lookup.get(b)
                    if a is None or pd.isna(a):
                        return None
                    if a <= q33:
                        return "small"
                    if a <= q67:
                        return "medium"
                    return "large"
                bins = ["small", "medium", "large"]
                fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
                area_rows = []
                for cond_id, by_seed in results.items():
                    if not by_seed:
                        continue
                    medians = []
                    for bn in bins:
                        vals = []
                        for seed, basin_metrics in by_seed.items():
                            for basin, m in basin_metrics.items():
                                if area_bin(basin) == bn and not np.isnan(m.get("NSE", np.nan)):
                                    vals.append(m["NSE"])
                        medians.append(np.median(vals) if vals else np.nan)
                        area_rows.append({"condition": cond_id, "area_tercile": bn,
                                            "n_basin_seed": len(vals),
                                            "median_NSE": float(np.median(vals)) if vals else None})
                    ax.plot(bins, medians, "o-", lw=1.6, label=cond_id)
                ax.set_xlabel(f"area tercile (cuts: {q33:.0f}, {q67:.0f} km²)")
                ax.set_ylabel("median NSE across basins × seeds")
                ax.set_title("Area-stratified median NSE — 5-condition factorial")
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(OUT_DIR / "nse_by_area.png", dpi=140)
                plt.close(fig)
                pd.DataFrame(area_rows).to_csv(OUT_DIR / "area_stratified.csv", index=False)

    # =====================================================================
    # RESULTS.md (slideshow-ready summary)
    # =====================================================================
    md = []
    md.append("# 5-Condition Factorial — Results (Component 0, 183 basins)")
    md.append("")
    md.append("Auto-generated by `experiments/analysis/compare_5conditions.py`. "
                "Re-run the script after every fresh sweep.")
    md.append("")

    md.append("## Headline — cross-seed median (with bootstrap 95% CI)")
    md.append("")
    md.append("| Condition | NSE | KGE | log-NSE | Seeds |")
    md.append("|---|---|---|---|---|")
    for cond_id in COND_IDS:
        if cond_id not in summary["conditions"]:
            md.append(f"| {cond_id} | — | — | — | (no data) |")
            continue
        row = summary["conditions"][cond_id]
        seeds = row["seeds"]
        cells = [cond_id]
        for metric in ["NSE", "KGE", "logNSE"]:
            m = row[metric]
            cells.append(f"{m['cross_seed_median']:+.3f}  [{m['bootstrap_95ci'][0]:+.3f}, {m['bootstrap_95ci'][1]:+.3f}]")
        cells.append(", ".join(str(s) for s in seeds))
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Six pairwise contrasts (paired per basin × seed) — NSE")
    md.append("")
    md.append("| Contrast | n | median Δ | 95% CI | strongly + | strongly − |")
    md.append("|---|---|---|---|---|---|")
    contrast_titles = {
        "L_minus_G":     "L − G   (architecture / methodology delta)",
        "GT_minus_G":    "(G+T) − G   (effect of topology features alone)",
        "GM_minus_G":    "(G+M) − G   (effect of message passing alone)",
        "GTM_minus_GT":  "(G+T+M) − (G+T)   (adding messages on top of T)",
        "GTM_minus_GM":  "(G+T+M) − (G+M)   (adding T on top of M)",
        "GTM_minus_G":   "(G+T+M) − G   (combined effect)",
    }
    for name in [c[0] for c in PAIRWISE_CONTRASTS]:
        if name not in contrasts_summary or "NSE" not in contrasts_summary[name]:
            md.append(f"| {contrast_titles[name]} | — | — | — | — | — |")
            continue
        d = contrasts_summary[name]["NSE"]
        md.append(
            f"| {contrast_titles[name]} | {d['n']} | {d['median']:+.3f} | "
            f"[{d['bootstrap_95ci'][0]:+.3f}, {d['bootstrap_95ci'][1]:+.3f}] | "
            f"{d['n_strongly_positive']} | {d['n_strongly_negative']} |"
        )
    md.append("")

    md.append("## Interaction term  ((G+T+M) − (G+T) − (G+M) + G)")
    md.append("")
    if "interaction_TxM" in contrasts_summary:
        md.append("| Metric | n | median | 95% CI | mean | std |")
        md.append("|---|---|---|---|---|---|")
        for metric in ["NSE", "KGE", "logNSE"]:
            d = contrasts_summary["interaction_TxM"].get(metric)
            if d is None:
                md.append(f"| {metric} | — | — | — | — | — |")
                continue
            md.append(
                f"| {metric} | {d['n']} | {d['median']:+.3f} | "
                f"[{d['bootstrap_95ci'][0]:+.3f}, {d['bootstrap_95ci'][1]:+.3f}] | "
                f"{d['mean']:+.3f} | {d['std']:.3f} |"
            )
    else:
        md.append("Not enough conditions present yet to compute the interaction.")
    md.append("")
    md.append("**Reading guide:**")
    md.append("- Median ≈ 0 → topology features and messages are *additive* (each contributes independently).")
    md.append("- Median > 0 → super-additive (the combination is bigger than the sum of parts; complementary signals).")
    md.append("- Median < 0 → sub-additive (T and M overlap; combining gains less than the parts suggest).")
    md.append("")

    if "outlier_trimmed_5pct" in summary:
        md.append("## Outlier-trimmed (drop bottom-5% NSE per seed)")
        md.append("")
        md.append("| Condition | full median NSE | trimmed median NSE | shift |")
        md.append("|---|---|---|---|")
        for cond_id in COND_IDS:
            d = summary["outlier_trimmed_5pct"].get(cond_id)
            if d is None:
                md.append(f"| {cond_id} | — | — | — |")
                continue
            shift = d["cross_seed_median_trimmed"] - d["cross_seed_median_full"]
            md.append(f"| {cond_id} | {d['cross_seed_median_full']:+.3f} | "
                       f"{d['cross_seed_median_trimmed']:+.3f} | {shift:+.3f} |")
        md.append("")

    md.append("## Stratified plots")
    md.append("")
    md.append("![Per-basin ΔNSE histograms](contrasts.png)")
    md.append("")
    if (OUT_DIR / "nse_by_depth.png").exists():
        md.append("![Median NSE by graph depth](nse_by_depth.png)")
        md.append("")
    if (OUT_DIR / "nse_by_area.png").exists():
        md.append("![Median NSE by area tercile](nse_by_area.png)")
        md.append("")

    md.append("## Provenance")
    md.append("")
    md.append("- Run folder: `runs/5cond_factorial/`")
    md.append(f"- Conditions found: " + ", ".join(
        f"{c} ({summary['conditions'].get(c, {}).get('n_seeds', 0)} seed(s))" for c in COND_IDS))
    md.append("- Metrics computed identically from raw test predictions.")
    md.append("- Bootstrap CIs across seeds (paired Δ values for contrasts), "
                f"n_boot={args.n_boot}, alpha=0.05.")
    md.append("")

    with open(OUT_DIR / "RESULTS.md", "w") as f:
        f.write("\n".join(md))

    # Stdout headline summary
    print("\n=== HEADLINE ===")
    for cond_id in COND_IDS:
        if cond_id not in summary["conditions"]:
            continue
        row = summary["conditions"][cond_id]
        nse_med = row["NSE"]["cross_seed_median"]
        kge_med = row["KGE"]["cross_seed_median"]
        log_med = row["logNSE"]["cross_seed_median"]
        print(f"  {cond_id:>6}  NSE {nse_med:+.3f}  KGE {kge_med:+.3f}  logNSE {log_med:+.3f}")
    print(f"\nOutputs at: {OUT_DIR}")


if __name__ == "__main__":
    main()
