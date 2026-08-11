"""Quantify the two-stage train/test asymmetry in the realizable model.

The realizable condition builds its input from stage-1 baseline predictions produced over
the FULL span 1990-2008 (`build_predicted_upstream_q.py`). But stage 1 was trained on
1990-1999, so on the stage-2 training years its predictions are in-sample fits, while on
the 2005-2008 test years they are genuine out-of-sample forecasts.

Consequence: stage 2 learns to trust an upstream-flow input that is more accurate during
training than at deployment. The paper discloses this; this script measures it, so the
disclosure can carry a number instead of an assurance.

Reported per seed and pooled:
  - stage-1 NSE on the stage-2 TRAIN window (1990-1999), in-sample
  - stage-1 NSE on the TEST window (2005-2008), out-of-sample
  - the gap, which is the magnitude of the asymmetry

Usage:  python experiments/topology_ablation/analyze_stage1_gap.py
Writes: analysis/STAGE1_GAP.md
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
OUT = Path(__file__).parent / "analysis" / "STAGE1_GAP.md"

SEEDS = [11, 13, 17]
TRAIN = ("1990-01-01", "1999-12-31")
TEST = ("2005-01-01", "2008-12-31")


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    ok = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[ok], sim[ok]
    if obs.size < 2:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def per_basin_nse(seed: int) -> pd.DataFrame:
    path = BASE / f"_Lfullspan_eval_seed{seed}" / "test" / "model_epoch030" / "test_results.p"
    res = pickle.load(open(path, "rb"))
    rows = []
    for basin, byfreq in res.items():
        ds = byfreq[next(iter(byfreq))]["xr"]
        idx = pd.to_datetime(ds["date"].values)
        obs = np.asarray(ds["QObs(mm/d)_obs"].values).flatten()
        sim = np.asarray(ds["QObs(mm/d)_sim"].values).flatten()
        n = min(len(idx), len(obs), len(sim))
        idx, obs, sim = idx[:n], obs[:n], sim[:n]
        tr = (idx >= TRAIN[0]) & (idx <= TRAIN[1])
        te = (idx >= TEST[0]) & (idx <= TEST[1])
        if tr.sum() < 100 or te.sum() < 100:
            continue
        rows.append({
            "basin": basin,
            "nse_train_insample": nse(obs[tr], sim[tr]),
            "nse_test_oos": nse(obs[te], sim[te]),
        })
    df = pd.DataFrame(rows)
    df["gap"] = df["nse_train_insample"] - df["nse_test_oos"]
    return df


def main() -> None:
    lines, all_gaps = [], []
    per_seed = {}
    for seed in SEEDS:
        df = per_basin_nse(seed)
        per_seed[seed] = df
        all_gaps.append(df["gap"].to_numpy())
        lines.append(
            f"| {seed} | {len(df)} | {df['nse_train_insample'].median():+.3f} | "
            f"{df['nse_test_oos'].median():+.3f} | {df['gap'].median():+.3f} |"
        )
    pooled = np.concatenate(all_gaps)
    pooled = pooled[np.isfinite(pooled)]

    tr_med = np.median([per_seed[s]["nse_train_insample"].median() for s in SEEDS])
    te_med = np.median([per_seed[s]["nse_test_oos"].median() for s in SEEDS])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "# Stage-1 Train/Test Asymmetry in the Realizable Model\n\n"
        "Zero training. Re-analysis of the stored full-span (1990-2008) stage-1 baseline\n"
        "predictions used to build the realizable upstream-flow input.\n\n"
        "**Why this matters.** The realizable feature is built from stage-1 predictions over the\n"
        "full record, but stage 1 was trained on 1990-1999. Its predictions are therefore\n"
        "in-sample on the stage-2 training window and out-of-sample on the test window, so the\n"
        "input degrades between training and deployment. This table measures that degradation.\n\n"
        "## Median per-basin NSE of the stage-1 predictor\n\n"
        "| seed | n basins | train window 1990-1999 (in-sample) | test window 2005-2008 (OOS) | gap |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(lines)
        + "\n\n"
        f"**Cross-seed median: {tr_med:+.3f} in-sample vs {te_med:+.3f} out-of-sample, "
        f"a gap of {tr_med - te_med:+.3f} NSE.**\n"
        f"Pooled per-basin gap: median {np.median(pooled):+.3f}, "
        f"IQR [{np.percentile(pooled, 25):+.3f}, {np.percentile(pooled, 75):+.3f}].\n\n"
        "## Reading\n\n"
        "The gap is the amount by which the upstream-flow input is better during stage-2\n"
        "training than at test. Stage 2 therefore learns to weight a cleaner signal than it\n"
        "receives at evaluation, which biases the realizable gain **downward**: the reported\n"
        "+0.022 is a conservative estimate of what a matched-quality feature would deliver.\n"
        "It does not inflate the result. Removing the asymmetry entirely would require a\n"
        "held-out-fold stage 1 (train stage 1 on a subset, predict the rest), which is a\n"
        "retraining experiment and is left as future work.\n"
    )
    print(OUT.read_text())


if __name__ == "__main__":
    main()
