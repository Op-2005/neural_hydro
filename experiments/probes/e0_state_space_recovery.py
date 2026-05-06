"""E0 — State-space vs prediction-space recovery measurement.

Addresses the hostile-reviewer concern from JOURNAL.md 2026-04-24
(Reviewer Q1): "Probe A's prediction-space recovery might just mean the
head W is approximately orthogonal to the noise direction. You haven't
shown true state-space recovery."

For the run-05 strong baseline, this script computes BOTH:
  * prediction-space deviation:  |y_pert(t) - y_unpert(t)| / σ_y
  * state-space deviation:       ||h_pert(t) - h_unpert(t)|| / σ_h

after a hidden-state perturbation at t=15. If state-space recovery is
fast (decay rate similar to prediction-space), then the LSTM cell map is
genuinely contracting and the framing claim holds in state space too.
If state-space recovery is much slower, then prediction-space recovery
was head-orthogonality, not true state-space recovery — a weaker but
still useful finding.

Outputs:
  experiments/analysis_outputs/e0/state_space_recovery.png
  experiments/analysis_outputs/e0/state_space_recovery.csv
  experiments/analysis_outputs/e0/state_space_decision.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "training"))

from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.utils.config import Config

from train_graph_lstm import DirectedGraphLSTM, load_basin_data
from e0_self_stabilization import warm_start_from_baseline_ckpt, step_forward

BASELINE_DIR = ROOT / "runs" / "05_lstm_23basin_strong_baseline"
BASIN_FILE = ROOT / "experiments" / "basin_lists" / "study_network_basins.txt"
OUT_DIR = ROOT / "experiments" / "analysis_outputs" / "dynamical_systems_probes" / "e0"
HIDDEN_SIZE = 64
DROPOUT = 0.4
DEVICE = torch.device("cpu")
SEED = 42
N_WINDOWS = 200
PERTURB_T = 15
PERTURB_SIGMA_FRAC = 0.5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    basin_ids = [l.strip() for l in open(BASIN_FILE) if l.strip()]
    n_basins = len(basin_ids)

    cfg = Config(BASELINE_DIR / "config.yml")
    scaler = load_scaler(BASELINE_DIR)
    id_to_int = load_basin_id_encoding(BASELINE_DIR) if cfg.use_basin_id_encoding else {}

    print(f"Loading test data...")
    x_d_test, x_s, _ = load_basin_data(cfg, scaler, basin_ids, "test", id_to_int)
    n_dyn = x_d_test.shape[3]
    seq_len = x_d_test.shape[2]
    input_size = n_dyn + x_s.shape[1]

    model = DirectedGraphLSTM(
        input_size=input_size, hidden_size=HIDDEN_SIZE, edges=[],
        n_basins=n_basins, n_targets=1, dropout=DROPOUT,
        initial_forget_bias=cfg.initial_forget_bias,
        use_edge_features=False, use_diff_term=False,
        use_attention=False, use_sigmoid_gate=False,
    ).to(DEVICE).eval()
    base_ckpt = sorted(BASELINE_DIR.glob("model_epoch*.pt"))[-1]
    warm_start_from_baseline_ckpt(model, base_ckpt)

    rng = np.random.RandomState(SEED)
    sampled = np.sort(rng.choice(x_d_test.shape[0], N_WINDOWS, replace=False))
    x_s_dev = x_s.to(DEVICE)

    # Pass 1: collect unperturbed h and y trajectories
    print(f"Collecting unperturbed h + y trajectories on {N_WINDOWS} windows...")
    h_full_unpert = torch.zeros(len(sampled), seq_len, n_basins, HIDDEN_SIZE)
    y_full_unpert = torch.zeros(len(sampled), seq_len, n_basins)
    with torch.no_grad():
        for wi, w in enumerate(sampled):
            x_w = x_d_test[w].transpose(0, 1).to(DEVICE)
            h_traj, y_traj = step_forward(model, x_w, x_s_dev)
            h_full_unpert[wi] = h_traj.cpu()
            y_full_unpert[wi] = y_traj.cpu()

    # Reference scales: per-basin std of unperturbed h and y over post-PERTURB_T
    h_post_T = h_full_unpert[:, PERTURB_T:, :, :]
    y_post_T = y_full_unpert[:, PERTURB_T:, :]
    natural_h_std = h_post_T.std(dim=0).mean(dim=0)        # [n_basins, hidden]
    natural_y_std = y_post_T.std(dim=0).mean(dim=0)        # [n_basins]
    h_at_T_std = h_full_unpert[:, PERTURB_T, :, :].std(dim=0)   # [n_basins, hidden] — for noise

    # Pass 2: perturbed
    print(f"Running perturbed pass (σ={PERTURB_SIGMA_FRAC})...")
    h_full_pert = torch.zeros_like(h_full_unpert)
    y_full_pert = torch.zeros_like(y_full_unpert)
    noise_sigma = PERTURB_SIGMA_FRAC * h_at_T_std
    with torch.no_grad():
        for wi, w in enumerate(sampled):
            x_w = x_d_test[w].transpose(0, 1).to(DEVICE)
            torch.manual_seed(SEED + int(w))
            noise = torch.randn(n_basins, HIDDEN_SIZE) * noise_sigma
            h_traj, y_traj = step_forward(model, x_w, x_s_dev,
                                            perturb_at=PERTURB_T,
                                            perturb_noise=noise.to(DEVICE))
            h_full_pert[wi] = h_traj.cpu()
            y_full_pert[wi] = y_traj.cpu()

    # Compute |Δh| and |Δy| over time, normalized
    h_dev_norm_per_window = (h_full_pert - h_full_unpert).norm(dim=-1)
    h_natural_norm = natural_h_std.norm(dim=-1)
    h_dev_normalized = h_dev_norm_per_window / (h_natural_norm[None, None, :] + 1e-6)
    y_dev_normalized = (y_full_pert - y_full_unpert).abs() / (natural_y_std[None, None, :] + 1e-6)

    # Median across windows × basins for each timestep
    h_curve = h_dev_normalized.median(dim=0).values.median(dim=-1).values.numpy()
    y_curve = y_dev_normalized.median(dim=0).values.median(dim=-1).values.numpy()

    # Quantiles for IQR shading: per-timestep, across windows × basins.
    # Shape: [n_windows, seq_len, n_basins] -> permute to [seq_len, n_windows*n_basins]
    h_flat = h_dev_normalized.permute(1, 0, 2).flatten(1, 2)
    y_flat = y_dev_normalized.permute(1, 0, 2).flatten(1, 2)
    h_q25 = h_flat.quantile(0.25, dim=-1).numpy()
    h_q75 = h_flat.quantile(0.75, dim=-1).numpy()
    y_q25 = y_flat.quantile(0.25, dim=-1).numpy()
    y_q75 = y_flat.quantile(0.75, dim=-1).numpy()

    # Plot
    t_axis = np.arange(seq_len) - PERTURB_T
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(t_axis, h_curve, color="C0", lw=1.8, label="state-space ‖Δh‖ / ‖σ_h‖")
    ax.fill_between(t_axis, h_q25, h_q75, alpha=0.20, color="C0")
    ax.plot(t_axis, y_curve, color="C3", lw=1.8, ls="--", label="prediction-space |Δy| / σ_y")
    ax.fill_between(t_axis, y_q25, y_q75, alpha=0.20, color="C3")
    ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.axhline(0.10, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("timestep relative to perturbation")
    ax.set_ylabel("normalized deviation")
    ax.set_title(f"E0 state-space vs prediction-space recovery (run-05, σ={PERTURB_SIGMA_FRAC})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "state_space_recovery.png", dpi=140)
    plt.close(fig)
    print(f"Wrote figure: {OUT_DIR / 'state_space_recovery.png'}")

    # CSV
    df = pd.DataFrame({
        "t_relative": t_axis,
        "h_dev_normalized_median": h_curve,
        "y_dev_normalized_median": y_curve,
        "h_dev_q25": h_q25, "h_dev_q75": h_q75,
        "y_dev_q25": y_q25, "y_dev_q75": y_q75,
    })
    df.to_csv(OUT_DIR / "state_space_recovery.csv", index=False)

    # Decision record
    h_at_T = float(h_curve[PERTURB_T])
    y_at_T = float(y_curve[PERTURB_T])
    h_at_T_plus_1 = float(h_curve[PERTURB_T + 1]) if PERTURB_T + 1 < seq_len else None
    y_at_T_plus_1 = float(y_curve[PERTURB_T + 1]) if PERTURB_T + 1 < seq_len else None
    h_at_T_plus_5 = float(h_curve[PERTURB_T + 5]) if PERTURB_T + 5 < seq_len else None
    y_at_T_plus_5 = float(y_curve[PERTURB_T + 5]) if PERTURB_T + 5 < seq_len else None

    interpretation = (
        f"State-space recovery: ‖Δh‖/‖σ_h‖ goes from {h_at_T:.3f} at t=T to "
        f"{h_at_T_plus_5:.3f} at t=T+5. "
        f"Prediction-space recovery: |Δy|/σ_y goes from {y_at_T:.3f} at t=T to "
        f"{y_at_T_plus_5:.3f} at t=T+5. "
    )
    if h_at_T_plus_5 < 0.10 and y_at_T_plus_5 < 0.10:
        interpretation += "BOTH state-space and prediction-space recover within 5 steps to <10% of natural variance — true contracting dynamics, not just head-orthogonality."
    elif y_at_T_plus_5 < 0.10 and h_at_T_plus_5 >= 0.10:
        interpretation += "Prediction-space recovers but state-space DOES NOT — head-orthogonality / null-space rejection, not true contracting dynamics. The framing claim should be qualified to 'prediction-space self-stabilization' specifically."
    else:
        interpretation += "Mixed signal — recovery is partial in both spaces. Framing needs nuance."

    record = {
        "perturb_t": PERTURB_T,
        "perturb_sigma_frac": PERTURB_SIGMA_FRAC,
        "n_windows": N_WINDOWS,
        "h_dev_normalized_at_T": h_at_T,
        "h_dev_normalized_at_T_plus_1": h_at_T_plus_1,
        "h_dev_normalized_at_T_plus_5": h_at_T_plus_5,
        "y_dev_normalized_at_T": y_at_T,
        "y_dev_normalized_at_T_plus_1": y_at_T_plus_1,
        "y_dev_normalized_at_T_plus_5": y_at_T_plus_5,
        "h_recovers_below_10pct_within_5_steps": (h_at_T_plus_5 is not None
                                                     and h_at_T_plus_5 < 0.10),
        "y_recovers_below_10pct_within_5_steps": (y_at_T_plus_5 is not None
                                                     and y_at_T_plus_5 < 0.10),
        "interpretation": interpretation,
    }
    with open(OUT_DIR / "state_space_decision.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote decision record: {OUT_DIR / 'state_space_decision.json'}")
    print()
    print("INTERPRETATION:")
    print(interpretation)


if __name__ == "__main__":
    main()
