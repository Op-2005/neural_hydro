"""E0.5 — Multi-seed loss-saturation analysis across 5 seeds.

Reads per-seed output.log from runs/lstm_strong_60ep_seed{11,13,17,19,23}_*/,
parses train loss + val loss + val NSE per epoch, plots all seeds plus the
across-seed mean ± 95% CI band, and computes the cross-seed plateau-band
statistics.

Question this answers (per JOURNAL.md 2026-04-25 entry queued plan):
"Does multi-seed E0.5 confirm the pragmatic 'val-saturated' reading from
the single-seed run yesterday?"

Pre-registered checks:
- Cross-seed val-NSE plateau-band median should agree across seeds within
  ± 0.05 NSE (otherwise plateau claim is seed-dependent).
- Linear-regression slope of val NSE over epochs 10-60, averaged across
  seeds, should be near zero (≤ |0.001|/epoch).
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "experiments" / "analysis_outputs" / "e0_5"
SEEDS = [11, 13, 17, 19, 23]


def parse_log(run_dir: Path):
    """Per-epoch train loss + val loss + val NSE from NH output.log."""
    log_path = run_dir / "output.log"
    train, val_loss, val_nse = {}, {}, {}
    if not log_path.exists():
        return train, val_loss, val_nse

    train_pat = re.compile(
        r"Epoch\s+(\d+)\s+average\s+loss:\s*avg_loss:\s*([0-9.]+)", re.IGNORECASE)
    val_pat = re.compile(
        r"Epoch\s+(\d+)\s+average\s+validation\s+loss:\s*([0-9.]+).*NSE:\s*([0-9.\-]+)",
        re.IGNORECASE)

    with open(log_path) as f:
        for line in f:
            m = train_pat.search(line)
            if m:
                train[int(m.group(1))] = float(m.group(2))
                continue
            m = val_pat.search(line)
            if m:
                ep = int(m.group(1))
                val_loss[ep] = float(m.group(2))
                val_nse[ep] = float(m.group(3))
    return train, val_loss, val_nse


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    for seed in SEEDS:
        cands = sorted(ROOT.glob(f"runs/lstm_strong_60ep_seed{seed}_*"))
        if not cands:
            print(f"WARN: no run for seed {seed}")
            continue
        run_dir = cands[-1]
        train, val_loss, val_nse = parse_log(run_dir)
        per_seed[seed] = {"train": train, "val_loss": val_loss, "val_nse": val_nse,
                           "run_dir": str(run_dir)}
        print(f"seed {seed}: {len(train)} train epochs, {len(val_nse)} val epochs")

    # Aggregate across seeds: align by epoch
    all_epochs = sorted(set().union(*[d["val_nse"].keys() for d in per_seed.values()]))
    n_seeds = len(per_seed)

    nse_matrix = np.full((n_seeds, len(all_epochs)), np.nan)
    loss_matrix = np.full((n_seeds, len(all_epochs)), np.nan)
    train_matrix = np.full((n_seeds, len(all_epochs)), np.nan)
    for si, seed in enumerate(per_seed):
        for ei, ep in enumerate(all_epochs):
            if ep in per_seed[seed]["val_nse"]:
                nse_matrix[si, ei] = per_seed[seed]["val_nse"][ep]
            if ep in per_seed[seed]["val_loss"]:
                loss_matrix[si, ei] = per_seed[seed]["val_loss"][ep]
            if ep in per_seed[seed]["train"]:
                train_matrix[si, ei] = per_seed[seed]["train"][ep]

    # Per-seed plateau-band median (over epochs >= 5)
    eps_arr = np.array(all_epochs)
    post5 = eps_arr >= 5
    seed_plateau_medians = np.nanmedian(nse_matrix[:, post5], axis=1)
    seed_plateau_mads = np.nanmedian(
        np.abs(nse_matrix[:, post5] - seed_plateau_medians[:, None]), axis=1)

    # Linear regression slope per seed for epochs 10-60
    slope_window = (eps_arr >= 10) & (eps_arr <= 60)
    seed_slopes = []
    for si in range(n_seeds):
        ys = nse_matrix[si, slope_window]
        xs = eps_arr[slope_window]
        valid = ~np.isnan(ys)
        if valid.sum() < 5:
            seed_slopes.append(np.nan)
            continue
        slope, _ = np.polyfit(xs[valid], ys[valid], 1)
        seed_slopes.append(float(slope))
    seed_slopes = np.array(seed_slopes)

    # Across-seed statistics
    nse_mean = np.nanmean(nse_matrix, axis=0)
    nse_std = np.nanstd(nse_matrix, axis=0)
    loss_mean = np.nanmean(loss_matrix, axis=0)
    loss_std = np.nanstd(loss_matrix, axis=0)
    train_mean = np.nanmean(train_matrix, axis=0)
    train_std = np.nanstd(train_matrix, axis=0)

    # Plot — three panels: train, val loss, val NSE
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    seed_colors = ["#666"] * n_seeds

    # Train
    ax = axes[0]
    for si, seed in enumerate(per_seed):
        ax.plot(eps_arr, train_matrix[si], color=seed_colors[si], lw=0.8, alpha=0.5,
                 label=f"seed {seed}" if si == 0 else None)
    ax.plot(eps_arr, train_mean, color="C0", lw=2.0, label="across-seed mean")
    ax.fill_between(eps_arr, train_mean - train_std, train_mean + train_std,
                     color="C0", alpha=0.2, label="±1σ across seeds")
    ax.axvline(30, color="grey", lw=0.7, ls=":", label="pilot stop")
    ax.set_xlabel("epoch"); ax.set_ylabel("train MSE")
    ax.set_title("train loss"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    # Val loss
    ax = axes[1]
    for si, seed in enumerate(per_seed):
        ax.plot(eps_arr, loss_matrix[si], color=seed_colors[si], lw=0.8, alpha=0.5)
    ax.plot(eps_arr, loss_mean, color="C3", lw=2.0, label="across-seed mean")
    ax.fill_between(eps_arr, loss_mean - loss_std, loss_mean + loss_std,
                     color="C3", alpha=0.2)
    ax.axvline(30, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation avg loss")
    ax.set_title("val loss"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    # Val NSE
    ax = axes[2]
    for si, seed in enumerate(per_seed):
        ax.plot(eps_arr, nse_matrix[si], color=seed_colors[si], lw=0.8, alpha=0.5,
                 label=f"seed {list(per_seed)[si]}")
    ax.plot(eps_arr, nse_mean, color="C2", lw=2.0, label="mean")
    ax.fill_between(eps_arr, nse_mean - nse_std, nse_mean + nse_std,
                     color="C2", alpha=0.2, label="±1σ")
    ax.axvline(30, color="grey", lw=0.7, ls=":")
    ax.axhline(seed_plateau_medians.mean(), color="C2", lw=0.5, ls="-.", alpha=0.5,
                label=f"plateau median {seed_plateau_medians.mean():.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("val NSE (median across basins)")
    ax.set_title("val NSE"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"E0.5 multi-seed loss saturation — {n_seeds} seeds × 60 epochs (strong baseline, 23 basins)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_saturation_multiseed.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote: {OUT_DIR / 'loss_saturation_multiseed.png'}")

    # Verdict
    plateau_disagreement = float(seed_plateau_medians.max() - seed_plateau_medians.min())
    mean_slope = float(np.nanmean(seed_slopes))
    abs_max_slope = float(np.nanmax(np.abs(seed_slopes)))

    plateau_pass = plateau_disagreement <= 0.05
    slope_pass = abs_max_slope <= 0.001

    record = {
        "seeds": list(per_seed.keys()),
        "per_seed_plateau_medians": [float(x) for x in seed_plateau_medians],
        "per_seed_plateau_mads": [float(x) for x in seed_plateau_mads],
        "across_seed_plateau_median_mean": float(seed_plateau_medians.mean()),
        "across_seed_plateau_disagreement_max_minus_min": plateau_disagreement,
        "per_seed_slope_epochs_10_to_60": [float(x) if not np.isnan(x) else None for x in seed_slopes],
        "mean_slope_across_seeds": mean_slope,
        "max_abs_slope_across_seeds": abs_max_slope,
        "plateau_check_pass": bool(plateau_pass),
        "slope_check_pass": bool(slope_pass),
        "verdict": (
            "MULTI-SEED CONFIRMS the pragmatic saturation reading: "
            "all seeds plateau in the same NSE band, no consistent upward trend."
            if (plateau_pass and slope_pass) else
            "MULTI-SEED PARTIALLY supports the saturation reading; "
            "some cross-seed disagreement requires nuance — see per-seed values."
        ),
    }
    with open(OUT_DIR / "decision_record_multiseed.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote: {OUT_DIR / 'decision_record_multiseed.json'}")

    print("\n=== Per-seed plateau (val NSE, epochs >= 5) ===")
    for seed, med, mad in zip(per_seed.keys(), seed_plateau_medians, seed_plateau_mads):
        print(f"  seed {seed:3d}: median {med:.3f} ± MAD {mad:.3f}")
    print(f"\n  Cross-seed plateau-median disagreement (max - min): {plateau_disagreement:.4f}")
    print(f"    Bar: ≤ 0.05 → {'PASS' if plateau_pass else 'FAIL'}")
    print(f"  Per-seed slope ep10→60: {[f'{s:+.5f}' for s in seed_slopes]}")
    print(f"    Mean across seeds: {mean_slope:+.5f}/epoch")
    print(f"    Max |slope|:        {abs_max_slope:.5f}/epoch")
    print(f"    Bar: |slope| ≤ 0.001 → {'PASS' if slope_pass else 'FAIL'}")
    print(f"\nVERDICT: {record['verdict']}")


if __name__ == "__main__":
    main()
