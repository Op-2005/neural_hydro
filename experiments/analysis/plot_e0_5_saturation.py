"""E0.5 — Plot the loss-saturation curve from a 60-epoch baseline retrain.

Reads NH's per-epoch training log + validation metrics, plots train and
validation loss vs. epoch, and prints the pre-registered saturation
verdict per `idea1.md` §E0.5:

  Hypothesis: validation loss is flat (≤ 1% relative change per 5 epochs)
              for ≥ 15 of the last 30 of the 60-epoch retrain.
  Falsification: validation loss still descending > 1% per 5 epochs at
                  epoch 60 → pilot was under-trained.

Usage:
    python experiments/analysis/plot_e0_5_saturation.py
        [--run-dir runs/lstm_study_network_strong_60ep_*]
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "experiments" / "analysis_outputs" / "dynamical_systems_probes" / "e0_5"


def find_latest_run():
    candidates = sorted(ROOT.glob("runs/lstm_study_network_strong_60ep_*"))
    return candidates[-1] if candidates else None


def parse_log(run_dir: Path):
    """Parse NH's output.log for per-epoch train loss + val loss + val NSE.

    NH log format observed in this version:
        Epoch N average loss: avg_loss: X.XXX, avg_total_loss: X.XXX
        Epoch N average validation loss: X.XXX -- Median validation metrics: avg_loss: X.XXX, NSE: X.XXX
    """
    log_path = run_dir / "output.log"
    epoch_train, epoch_val_loss, epoch_val_nse = {}, {}, {}
    if not log_path.exists():
        return epoch_train, epoch_val_loss, epoch_val_nse

    train_pat = re.compile(
        r"Epoch\s+(\d+)\s+average\s+loss:\s*avg_loss:\s*([0-9.]+)", re.IGNORECASE)
    val_pat = re.compile(
        r"Epoch\s+(\d+)\s+average\s+validation\s+loss:\s*([0-9.]+).*NSE:\s*([0-9.\-]+)",
        re.IGNORECASE)

    with open(log_path) as f:
        for line in f:
            m = train_pat.search(line)
            if m:
                ep = int(m.group(1))
                epoch_train[ep] = float(m.group(2))
                continue
            m = val_pat.search(line)
            if m:
                ep = int(m.group(1))
                epoch_val_loss[ep] = float(m.group(2))
                epoch_val_nse[ep] = float(m.group(3))
    return epoch_train, epoch_val_loss, epoch_val_nse


def saturation_verdict(val_nse_by_epoch: dict, n_tail_epochs: int = 30,
                        flat_threshold_per_5ep: float = 0.01,
                        min_flat_window: int = 15):
    """Verdict per idea1.md §E0.5:
       'flat (≤ 1% relative change per 5 epochs) for ≥ 15 of the last 30'
    """
    if not val_nse_by_epoch:
        return "no validation data — cannot compute"
    epochs = sorted(val_nse_by_epoch)
    if len(epochs) < n_tail_epochs:
        return f"only {len(epochs)} epochs of val data — need {n_tail_epochs}"

    tail = epochs[-n_tail_epochs:]
    # For each 5-epoch sliding window in the tail, compute relative change
    # of the median NSE. We want ≥ min_flat_window of these to be < 1%.
    flat_count = 0
    eligible = 0
    for i in range(0, len(tail) - 5):
        ep_a, ep_b = tail[i], tail[i + 5]
        v_a, v_b = val_nse_by_epoch[ep_a], val_nse_by_epoch[ep_b]
        if abs(v_a) < 1e-6:
            continue
        rel_change = abs(v_b - v_a) / abs(v_a)
        eligible += 1
        if rel_change < flat_threshold_per_5ep:
            flat_count += 1

    if eligible == 0:
        return "no eligible 5-epoch windows in tail"
    if flat_count >= min_flat_window:
        return f"PASS — saturated ({flat_count}/{eligible} windows flat in last {n_tail_epochs} epochs)"
    return f"FAIL — not saturated ({flat_count}/{eligible} windows flat in last {n_tail_epochs} epochs)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None,
                        help="Path to run dir; autodetect if omitted")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    if run_dir is None or not run_dir.exists():
        raise SystemExit("No 60-epoch run found.")
    print(f"Run: {run_dir}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loss, val_loss, val_nse = parse_log(run_dir)
    print(f"  Train-loss epochs:    {len(train_loss)}")
    print(f"  Val-loss / NSE epochs: {len(val_loss)}")

    # Plot — three panels: train loss, val loss, val NSE
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (data, label, color) in zip(axes, [
        (train_loss, "train MSE", "C0"),
        (val_loss, "validation avg loss", "C3"),
        (val_nse, "validation NSE (median across basins)", "C2"),
    ]):
        if data:
            eps = sorted(data)
            ax.plot(eps, [data[e] for e in eps], color=color, lw=1.6, label=label)
        ax.axvline(30, color="grey", lw=0.7, ls=":", label="pilot stop (ep 30)")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"E0.5 — Loss saturation curve, strong baseline (60-epoch retrain)\n"
                  f"run: {run_dir.name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_saturation.png", dpi=140)
    plt.close(fig)
    print(f"  Wrote: {OUT_DIR / 'loss_saturation.png'}")

    verdict = saturation_verdict(val_nse)
    print(f"\nSaturation verdict (val NSE): {verdict}")
    verdict_loss = saturation_verdict(val_loss)
    print(f"Saturation verdict (val loss): {verdict_loss}")

    record = {
        "run_dir": str(run_dir),
        "n_train_loss_epochs": len(train_loss),
        "n_val_loss_epochs": len(val_loss),
        "n_val_nse_epochs": len(val_nse),
        "train_loss_by_epoch": {str(k): v for k, v in train_loss.items()},
        "val_loss_by_epoch": {str(k): v for k, v in val_loss.items()},
        "val_nse_by_epoch": {str(k): v for k, v in val_nse.items()},
        "saturation_verdict_val_nse": verdict,
        "saturation_verdict_val_loss": verdict_loss,
        "pilot_stop_epoch": 30,
        "val_nse_at_pilot_stop": val_nse.get(30),
        "val_nse_at_final_epoch": val_nse.get(max(val_nse)) if val_nse else None,
    }
    with open(OUT_DIR / "decision_record.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Wrote: {OUT_DIR / 'decision_record.json'}")


if __name__ == "__main__":
    main()
