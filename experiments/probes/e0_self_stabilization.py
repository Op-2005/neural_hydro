"""E0 — Self-stabilization verification of a trained baseline LSTM.

Pre-registered in `idea1.md` (post-2026-04-21 reframing). The dynamical-
systems framing rests on the claim that the trained multi-basin LSTM exhibits
self-stabilizing dynamics: its rolled-out predictions are dominated by the
model's own hidden-state evolution, not by external forcings.

This script runs two complementary probes on a chosen baseline (default:
run-05 strong baseline), loaded into a step-by-step LSTMCell wrapper
(DirectedGraphLSTM with empty edges = pure LSTM, since `has_parents=0`
makes message=0 at every step).

CLI args:
  --baseline-dir       path to baseline run (defaults to run-05 strong)
  --sigma              hidden-state perturbation σ (× per-basin h-std)
  --probe-b-mode       t-1 (default) | zero | random-day
                       Controls the forcing-replacement variant for Probe B.
  --out-suffix         optional suffix appended to output filenames
                       (so multiple variants can coexist in the same dir)

Probe A — Perturbation recovery
    Roll out the LSTM on a test-period window. At a fixed midway timestep T,
    add Gaussian noise to the hidden state. Continue forward propagation.
    Measure prediction-space deviation between perturbed and unperturbed
    trajectories.
    Self-stabilization predicts: deviation peaks immediately, then decays
    monotonically, returning to within 10% of natural prediction variance
    in < 5 timesteps post-perturbation.

Probe B — Forcing replacement
    Roll out twice on the same window. In the second roll-out, at timestep
    T, replace the forcing with the t-1 forcing (i.e., "yesterday's
    weather"). Measure prediction-space deviation.
    Self-stabilization predicts: bounded effect on subsequent predictions
    — deviation is small relative to the natural day-to-day prediction
    variance of the unperturbed model.

Pre-registered success criterion (idea1.md §E0):
    Both probes show self-stabilization signatures on >= 50% of test basins.
    Specifically:
        Probe A: median post-perturbation prediction deviation drops below
                 10% of unperturbed prediction std within 5 timesteps.
        Probe B: median forcing-replacement prediction deviation stays
                 below 30% of unperturbed prediction std at all timesteps
                 after T.

Outputs:
    experiments/analysis_outputs/e0/probe_a_recovery.csv
    experiments/analysis_outputs/e0/probe_b_forcing.csv
    experiments/analysis_outputs/e0/probe_a_recovery.png
    experiments/analysis_outputs/e0/probe_b_forcing.png
    experiments/analysis_outputs/e0/decision_record.json

Usage:
    /Applications/anaconda3/envs/nh/bin/python experiments/probes/e0_self_stabilization.py
"""
import argparse
import json
import logging
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

DEFAULT_BASELINE_DIR = ROOT / "runs" / "05_lstm_23basin_strong_baseline"
BASIN_FILE = ROOT / "experiments" / "basin_lists" / "study_network_basins.txt"
OUT_DIR = ROOT / "experiments" / "analysis_outputs" / "e0"

HIDDEN_SIZE = 64
DROPOUT = 0.4
DEVICE = torch.device("cpu")
DEFAULT_SEED = 42

# Probe configuration
N_WINDOWS_PROBE = 200      # how many test windows to evaluate per probe
DEFAULT_PERTURB_T = 15     # timestep at which to perturb / replace forcing (out of 30)

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def warm_start_from_baseline_ckpt(model, ckpt_path):
    """Copy nn.LSTM weights from NH baseline into our LSTMCell-based model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mapping = {
        "lstm.weight_ih_l0": "lstm_cell.weight_ih",
        "lstm.weight_hh_l0": "lstm_cell.weight_hh",
        "lstm.bias_ih_l0": "lstm_cell.bias_ih",
        "lstm.bias_hh_l0": "lstm_cell.bias_hh",
        "head.net.0.weight": "head.weight",
        "head.net.0.bias": "head.bias",
    }
    own = model.state_dict()
    copied = 0
    for src, dst in mapping.items():
        if src in ckpt and dst in own and ckpt[src].shape == own[dst].shape:
            own[dst].copy_(ckpt[src])
            copied += 1
    LOGGER.info(f"  copied {copied} tensors from {ckpt_path.name}")


def step_forward(model, x_d_window, x_s, perturb_at=None, perturb_noise=None,
                 forcing_replace_at=None, forcing_replace_with=None):
    """Run the LSTM step-by-step, optionally perturbing hidden state or forcing.

    x_d_window : [seq_len, n_basins, n_dyn]
    x_s        : [n_basins, n_static_total]

    perturb_at         : int t ∈ [0, seq_len) — inject Gaussian noise into h
                         AFTER step t completes (so h at the start of t+1 is
                         perturbed)
    perturb_noise      : [n_basins, hidden] tensor of noise to add (must be
                         pre-generated for reproducibility across paired runs)
    forcing_replace_at : int t — at step t, replace x_d[t] with the
                         t-1-shifted forcing
    forcing_replace_with : [n_basins, n_dyn] tensor (typically x_d_window[t-1])

    Returns
    -------
    h_traj : [seq_len, n_basins, hidden]
    y_traj : [seq_len, n_basins] (single-target predictions; the head outputs
             a scalar per basin per timestep)
    """
    seq_len, n_basins, _ = x_d_window.shape
    h = torch.zeros(n_basins, model.hidden_size, device=x_d_window.device)
    c = torch.zeros(n_basins, model.hidden_size, device=x_d_window.device)
    h_traj = torch.zeros(seq_len, n_basins, model.hidden_size,
                          device=x_d_window.device)
    y_traj = torch.zeros(seq_len, n_basins, device=x_d_window.device)

    for t in range(seq_len):
        if forcing_replace_at is not None and t == forcing_replace_at:
            x_t_dyn = forcing_replace_with
        else:
            x_t_dyn = x_d_window[t]
        x_t = torch.cat([x_t_dyn, x_s], dim=-1)
        h, c = model.lstm_cell(x_t, (h, c))

        # Perturb AFTER the step — i.e., the state going into t+1 carries noise
        if perturb_at is not None and t == perturb_at and perturb_noise is not None:
            h = h + perturb_noise

        h_traj[t] = h
        # Apply head to current h (no dropout at eval) → scalar output per basin
        y_traj[t] = model.head(h).squeeze(-1)

    return h_traj, y_traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR),
                        help="Path to baseline run directory")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Probe A perturbation σ (× per-basin h-std)")
    parser.add_argument("--probe-b-mode", default="t-1",
                        choices=["t-1", "zero", "random-day"],
                        help="Probe B forcing-replacement mode")
    parser.add_argument("--perturb-t", type=int, default=DEFAULT_PERTURB_T,
                        help="Timestep at which to perturb (default 15 of 30)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for noise + window sampling")
    parser.add_argument("--measure-state-space", action="store_true",
                        help="Also measure state-space recovery ||h_pert - h_unpert||")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to output filenames")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    perturb_sigma_frac = args.sigma
    probe_b_mode = args.probe_b_mode
    out_suffix = args.out_suffix
    SEED = args.seed
    PERTURB_T = args.perturb_t
    measure_state_space = args.measure_state_space

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    basin_ids = [l.strip() for l in open(BASIN_FILE) if l.strip()]
    n_basins = len(basin_ids)

    LOGGER.info(f"Baseline: {baseline_dir}")
    LOGGER.info(f"Probe A σ = {perturb_sigma_frac};  Probe B mode = {probe_b_mode}")

    cfg = Config(baseline_dir / "config.yml")
    scaler = load_scaler(baseline_dir)
    id_to_int = load_basin_id_encoding(baseline_dir) if cfg.use_basin_id_encoding else {}

    LOGGER.info("Loading test-period data...")
    x_d_test, x_s, _ = load_basin_data(cfg, scaler, basin_ids, "test", id_to_int)
    LOGGER.info(f"x_d_test shape: {x_d_test.shape}   x_s shape: {x_s.shape}")
    n_dyn = x_d_test.shape[3]
    seq_len = x_d_test.shape[2]
    input_size = n_dyn + x_s.shape[1]

    # Build pure-LSTM model (DirectedGraphLSTM with empty edges)
    model = DirectedGraphLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        edges=[],
        n_basins=n_basins,
        n_targets=1,
        dropout=DROPOUT,
        initial_forget_bias=cfg.initial_forget_bias,
        use_edge_features=False,
        use_diff_term=False,
        use_attention=False,
        use_sigmoid_gate=False,
    ).to(DEVICE).eval()
    base_ckpt = sorted(baseline_dir.glob("model_epoch*.pt"))[-1]
    warm_start_from_baseline_ckpt(model, base_ckpt)

    # Sample a fixed set of windows (deterministic) for both probes
    n_windows_total = x_d_test.shape[0]
    rng = np.random.RandomState(SEED)
    if n_windows_total <= N_WINDOWS_PROBE:
        sampled = np.arange(n_windows_total)
    else:
        sampled = np.sort(rng.choice(n_windows_total, N_WINDOWS_PROBE, replace=False))
    LOGGER.info(f"Running probes on {len(sampled)} of {n_windows_total} test windows")

    x_s_dev = x_s.to(DEVICE)

    # First: collect the unperturbed reference trajectories AND establish the
    # natural prediction-variance scale per basin (std of clean predictions
    # across all sampled windows at all timesteps post-PERTURB_T).
    ref_y_post_T = []  # list of [n_basins, seq_len-PERTURB_T] arrays
    ref_h_at_T_std = torch.zeros(n_basins, HIDDEN_SIZE)  # std of clean h at T

    LOGGER.info("Collecting unperturbed reference trajectories...")
    h_at_T_collect = []
    y_full_collect = []
    with torch.no_grad():
        for w in sampled:
            x_w = x_d_test[w].transpose(0, 1).to(DEVICE)  # [seq_len, n_basins, n_dyn]
            h_traj, y_traj = step_forward(model, x_w, x_s_dev)
            h_at_T_collect.append(h_traj[PERTURB_T].cpu())  # [n_basins, hidden]
            y_full_collect.append(y_traj.cpu())             # [seq_len, n_basins]
    h_at_T_stack = torch.stack(h_at_T_collect, dim=0)  # [n_w, n_basins, hidden]
    y_full_stack = torch.stack(y_full_collect, dim=0)  # [n_w, seq_len, n_basins]
    ref_h_at_T_std = h_at_T_stack.std(dim=0)            # [n_basins, hidden]
    # Natural prediction std per basin: std of unperturbed y across windows,
    # averaged across post-T timesteps
    y_post_T = y_full_stack[:, PERTURB_T:, :]  # [n_w, seq_len-T, n_basins]
    natural_pred_std = y_post_T.std(dim=0).mean(dim=0)  # [n_basins]
    LOGGER.info(f"Natural per-basin prediction std (post-T): "
                 f"min={natural_pred_std.min():.3f} median="
                 f"{natural_pred_std.median():.3f} max={natural_pred_std.max():.3f}")

    # ============================================================
    # PROBE A: perturbation recovery
    # ============================================================
    LOGGER.info(f"Probe A: hidden-state perturbation at t={PERTURB_T} "
                 f"(noise σ = {perturb_sigma_frac} × per-window-std)")
    # For each window, generate one noise tensor per basin (scaled by per-basin
    # std of clean h at T, so the perturbation magnitude is comparable across
    # basins of different scale)
    noise_sigma = perturb_sigma_frac * ref_h_at_T_std  # [n_basins, hidden]

    deviations_a = np.zeros((len(sampled), seq_len, n_basins), dtype=np.float32)
    with torch.no_grad():
        for wi, w in enumerate(sampled):
            x_w = x_d_test[w].transpose(0, 1).to(DEVICE)
            ref_y = y_full_stack[wi]  # [seq_len, n_basins], already on cpu
            torch.manual_seed(SEED + int(w))
            noise = torch.randn(n_basins, HIDDEN_SIZE) * noise_sigma
            _, y_pert = step_forward(model, x_w, x_s_dev,
                                       perturb_at=PERTURB_T,
                                       perturb_noise=noise.to(DEVICE))
            deviations_a[wi] = (y_pert.cpu() - ref_y).numpy()

    # Aggregate: per-basin median absolute deviation by timestep relative to T
    abs_dev_a = np.abs(deviations_a)  # [n_w, seq_len, n_basins]
    median_abs_dev_a = np.median(abs_dev_a, axis=0)  # [seq_len, n_basins]
    # Normalize by per-basin natural prediction std
    normalized_dev_a = median_abs_dev_a / (natural_pred_std.numpy()[None, :] + 1e-6)
    # [seq_len, n_basins]; values close to 1 mean "deviation comparable to
    # natural prediction variance"; close to 0 means "back to baseline."

    # ============================================================
    # PROBE B: forcing replacement (mode controlled by --probe-b-mode)
    # ============================================================
    mode_descriptions = {
        "t-1": f"replace forcing at t={PERTURB_T} with t-1's forcing",
        "zero": f"zero-out forcing at t={PERTURB_T} (all dynamic inputs = 0)",
        "random-day": f"replace forcing at t={PERTURB_T} with a randomly-chosen historical day's forcing",
    }
    LOGGER.info(f"Probe B [{probe_b_mode}]: " + mode_descriptions[probe_b_mode])
    deviations_b = np.zeros((len(sampled), seq_len, n_basins), dtype=np.float32)
    rng_b = np.random.RandomState(SEED + 1)
    with torch.no_grad():
        for wi, w in enumerate(sampled):
            x_w = x_d_test[w].transpose(0, 1).to(DEVICE)
            ref_y = y_full_stack[wi]

            if probe_b_mode == "t-1":
                replace_with = x_w[PERTURB_T - 1]                     # [n_basins, n_dyn]
            elif probe_b_mode == "zero":
                replace_with = torch.zeros_like(x_w[PERTURB_T])
            elif probe_b_mode == "random-day":
                # Pick a random window and a random timestep inside it, then
                # use that window's forcing-row as the replacement.
                w_alt = int(rng_b.randint(0, n_windows_total))
                t_alt = int(rng_b.randint(0, seq_len))
                replace_with = x_d_test[w_alt, :, t_alt].to(DEVICE)
            else:
                raise ValueError(probe_b_mode)

            _, y_pert = step_forward(model, x_w, x_s_dev,
                                       forcing_replace_at=PERTURB_T,
                                       forcing_replace_with=replace_with)
            deviations_b[wi] = (y_pert.cpu() - ref_y).numpy()

    abs_dev_b = np.abs(deviations_b)
    median_abs_dev_b = np.median(abs_dev_b, axis=0)
    normalized_dev_b = median_abs_dev_b / (natural_pred_std.numpy()[None, :] + 1e-6)

    # ============================================================
    # Per-basin verdicts
    # ============================================================
    # Probe A success per basin: deviation drops below 10% of natural std
    # within 5 timesteps after T.
    a_success = np.zeros(n_basins, dtype=bool)
    a_recovery_step = np.full(n_basins, -1, dtype=int)
    for b in range(n_basins):
        # walk forward from PERTURB_T+1, find first step where deviation < 0.10
        for t in range(PERTURB_T + 1, min(PERTURB_T + 6, seq_len)):
            if normalized_dev_a[t, b] < 0.10:
                a_success[b] = True
                a_recovery_step[b] = t - PERTURB_T
                break

    # Probe B success per basin: max deviation in any post-T timestep stays
    # below 30% of natural std.
    b_max_post_T = normalized_dev_b[PERTURB_T:].max(axis=0)
    b_success = b_max_post_T < 0.30

    # Save per-basin results
    suffix = f"_{out_suffix}" if out_suffix else ""
    df_a = pd.DataFrame({
        "basin": basin_ids,
        "recovery_steps": a_recovery_step,
        "success_within_5_steps": a_success,
        "max_dev_post_T_normalized": normalized_dev_a[PERTURB_T:].max(axis=0),
    })
    df_a.to_csv(OUT_DIR / f"probe_a_recovery{suffix}.csv", index=False)

    df_b = pd.DataFrame({
        "basin": basin_ids,
        "max_dev_post_T_normalized": b_max_post_T,
        "success_below_30pct": b_success,
    })
    df_b.to_csv(OUT_DIR / f"probe_b_forcing{suffix}.csv", index=False)

    # ============================================================
    # Summary verdict
    # ============================================================
    a_pct = a_success.mean() * 100
    b_pct = b_success.mean() * 100
    LOGGER.info("=" * 70)
    LOGGER.info("E0 RESULTS (pre-registered bar: ≥ 50% of basins on each probe)")
    LOGGER.info("=" * 70)
    LOGGER.info(f"  Probe A — perturbation recovery within 5 steps: "
                 f"{a_success.sum()}/{n_basins} basins ({a_pct:.0f}%)  "
                 f"{'PASS' if a_pct >= 50 else 'FAIL'}")
    LOGGER.info(f"  Probe B — forcing-replacement deviation < 30%: "
                 f"{b_success.sum()}/{n_basins} basins ({b_pct:.0f}%)  "
                 f"{'PASS' if b_pct >= 50 else 'FAIL'}")

    overall_pass = (a_pct >= 50) and (b_pct >= 50)
    LOGGER.info(f"  OVERALL: {'PASS — framing alive' if overall_pass else 'FAIL — framing wounded'}")

    # Per-basin median recovery step for Probe A (only over basins that recovered)
    if a_success.any():
        median_recovery = int(np.median(a_recovery_step[a_success]))
        LOGGER.info(f"  Probe A median recovery step: {median_recovery}")
    LOGGER.info(f"  Probe B median max-deviation: {np.median(b_max_post_T):.3f}")

    # Plot: median normalized deviation by timestep, both probes
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    t_axis = np.arange(seq_len) - PERTURB_T

    # Probe A
    median_curve_a = np.median(normalized_dev_a, axis=1)
    q25_a = np.quantile(normalized_dev_a, 0.25, axis=1)
    q75_a = np.quantile(normalized_dev_a, 0.75, axis=1)
    axes[0].plot(t_axis, median_curve_a, color="C0", lw=1.8, label="median across basins")
    axes[0].fill_between(t_axis, q25_a, q75_a, alpha=0.25, color="C0", label="IQR")
    axes[0].axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[0].axhline(0.10, color="C2", lw=0.7, ls=":", label="10% of natural std")
    axes[0].set_title(f"Probe A — perturbation recovery\n"
                       f"PASS rate: {a_pct:.0f}% of basins")
    axes[0].set_xlabel("timestep relative to perturbation")
    axes[0].set_ylabel("median |Δprediction| / natural prediction std")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(bottom=-0.02)

    # Probe B
    median_curve_b = np.median(normalized_dev_b, axis=1)
    q25_b = np.quantile(normalized_dev_b, 0.25, axis=1)
    q75_b = np.quantile(normalized_dev_b, 0.75, axis=1)
    axes[1].plot(t_axis, median_curve_b, color="C3", lw=1.8, label="median across basins")
    axes[1].fill_between(t_axis, q25_b, q75_b, alpha=0.25, color="C3", label="IQR")
    axes[1].axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[1].axhline(0.30, color="C2", lw=0.7, ls=":", label="30% of natural std")
    axes[1].set_title(f"Probe B — forcing replacement\n"
                       f"PASS rate: {b_pct:.0f}% of basins")
    axes[1].set_xlabel("timestep relative to forcing replacement")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(bottom=-0.02)

    fig.suptitle(f"E0 — LSTM self-stabilization probes  ({baseline_dir.name}, "
                  f"σ={perturb_sigma_frac}, probe-B={probe_b_mode})",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"probe_a_recovery{suffix}.png", dpi=140)
    plt.close(fig)
    LOGGER.info(f"Wrote figure: {OUT_DIR / ('probe_a_recovery' + suffix + '.png')}")

    # Decision record
    decision = {
        "baseline_dir": str(baseline_dir),
        "probe_b_mode": probe_b_mode,
        "perturb_sigma_frac": perturb_sigma_frac,
        "probe_a_pass_pct": float(a_pct),
        "probe_b_pass_pct": float(b_pct),
        "probe_a_pass": bool(a_pct >= 50),
        "probe_b_pass": bool(b_pct >= 50),
        "overall_pass": bool(overall_pass),
        "n_basins": n_basins,
        "n_windows_evaluated": int(len(sampled)),
        "perturb_t": PERTURB_T,
        "probe_a_median_recovery_steps_among_recovered": (
            int(np.median(a_recovery_step[a_success])) if a_success.any() else None),
        "probe_b_median_max_deviation_normalized": float(np.median(b_max_post_T)),
        "interpretation": (
            "framing alive — proceed to E0.5 then forcing-comparison sub-experiment"
            if overall_pass else
            "framing wounded — revert idea1.md to pre-reframing version; "
            "the A/B/C ablation still stands as the empirical contribution"
        ),
    }
    out_record = OUT_DIR / f"decision_record{suffix}.json"
    with open(out_record, "w") as f:
        json.dump(decision, f, indent=2)
    LOGGER.info(f"Wrote decision record: {out_record}")


if __name__ == "__main__":
    main()
