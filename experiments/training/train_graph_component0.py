"""Scaled graph-LSTM trainer for Component 0 (183 basins).

This is the HYPOTHESIS.md §6 Step-3 script: train baseline-warm-started
DirectedGraph-LSTM variants on the 183-basin eastern-US network that has
meaningful n-per-depth-stratum (51 at depth 2, 16 at depth 3).

Intentionally parameterized via CLI so we can sweep seeds/variants without
editing code, which is the repo-hygiene fix flagged in the earlier review.

Three variants to run, each × multiple seeds:
  * --variant frozen     : graph+edges, LSTM+head frozen (clean test of clause i/iii)
  * --variant warm       : graph+edges, full finetune (for comparison to prior 23-basin result)
  * --variant gcn_lowpass: mean-aggregation only, no edge features, no direction —
                           the low-pass stand-in required by §4 item 3
                           (over-squashing control / Kirschstein-style null baseline)

Smoke-test mode (--epochs 2) reports wall-clock per epoch so we can set a
realistic compute budget for the full sweep.

Usage:
    /Applications/anaconda3/envs/nh/bin/python experiments/train_graph_component0.py \\
        --variant frozen --seed 42 --epochs 15
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Repo root is three levels up from experiments/training/<script>.py.
# Second insert keeps experiments/training on sys.path so we can import
# train_graph_lstm (same directory).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.utils.config import Config

from train_graph_lstm import (
    DirectedGraphLSTM,
    load_graph_with_features,
    load_basin_data,
    warm_start_from_baseline,
    train_epoch,
    evaluate,
    evaluate_with_predictions,
)

ROOT = Path(__file__).parent.parent.parent
BASIN_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt"
EDGE_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv"

HIDDEN_SIZE = 64
DROPOUT = 0.4
LR = 1e-3
SEQ_LENGTH = 30
BATCH_SIZE = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
LOGGER = logging.getLogger(__name__)
DEVICE = torch.device("cpu")


def find_component0_baseline():
    """Find the most recent successful Component 0 baseline run."""
    cands = sorted(ROOT.glob("runs/lstm_component0_baseline_*"))
    for d in reversed(cands):
        if list(d.glob("model_epoch*.pt")):
            return d
    return None


VARIANTS = {
    # Ablation dimensions: edge_feat, diff, frozen, attn, sigate, add_topology_features, use_full_edges
    # Conditions in the locked 5-condition factorial framework (idea1.md §"Revised Framework"):
    #   L = NH cudalstm baseline                             (no script needed; use lstm_*_baseline.yaml)
    #   G = DirectedGraphLSTM, empty edges, no topology      → variant `empty_graph`         [NEW]
    #   G+T = DirectedGraphLSTM, empty edges, + topology     → variant `topology_features`
    #   G+M = DirectedGraphLSTM, full edges, no topology     → variant `warm`
    #   G+T+M = DirectedGraphLSTM, full edges, + topology    → variant `full_graph_with_topology` [NEW]
    "warm":                       dict(edge_feat=True,  diff=False, frozen=False, attn=False, sigate=False, add_topology_features=False, use_full_edges=True),
    "frozen":                     dict(edge_feat=True,  diff=False, frozen=True,  attn=False, sigate=False, add_topology_features=False, use_full_edges=True),
    "gcn_lowpass":                dict(edge_feat=False, diff=False, frozen=False, attn=False, sigate=False, add_topology_features=False, use_full_edges=True),
    "topology_features":          dict(edge_feat=False, diff=False, frozen=False, attn=False, sigate=False, add_topology_features=True,  use_full_edges=False),
    # === New variants for the 5-condition factorial (added 2026-05-06) ===
    "empty_graph":                dict(edge_feat=False, diff=False, frozen=False, attn=False, sigate=False, add_topology_features=False, use_full_edges=False),
    "full_graph_with_topology":   dict(edge_feat=True,  diff=False, frozen=False, attn=False, sigate=False, add_topology_features=True,  use_full_edges=True),
}


def compute_topology_features(basin_ids, edge_file, topo_file):
    """Compute the 5 Condition-B topology scalars per basin, z-normalized.

    Returns: torch.Tensor [n_basins, 5] with columns:
      0  graph depth (longest path from any root to this basin)
      1  in-degree
      2  out-degree
      3  transitive upstream count / network size
      4  log of (sum-of-upstream-areas + own-area) / own-area, z-normalized
    """
    import networkx as nx
    edges_df = pd.read_csv(edge_file, dtype={"parent_id": str, "child_id": str})
    topo = pd.read_csv(topo_file, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")

    G = nx.DiGraph()
    for b in basin_ids:
        G.add_node(b)
    for _, row in edges_df.iterrows():
        if row["parent_id"] in basin_ids and row["child_id"] in basin_ids:
            G.add_edge(row["parent_id"], row["child_id"])

    n = len(basin_ids)
    feats = np.zeros((n, 5))
    roots = [m for m in G if G.in_degree(m) == 0]
    for i, b in enumerate(basin_ids):
        # Depth: longest path from any root to b
        max_depth = 0
        for r in roots:
            try:
                d = nx.shortest_path_length(G, r, b)
                max_depth = max(max_depth, d)
            except nx.NetworkXNoPath:
                continue
        feats[i, 0] = max_depth
        feats[i, 1] = G.in_degree(b)
        feats[i, 2] = G.out_degree(b)
        ancestors = nx.ancestors(G, b)
        feats[i, 3] = len(ancestors) / max(n, 1)
        own_area = float(topo.loc[b, "area_gages2"]) if b in topo.index else 1.0
        upstream_area = sum(float(topo.loc[a, "area_gages2"])
                              for a in ancestors if a in topo.index)
        feats[i, 4] = np.log((upstream_area + own_area) / max(own_area, 1.0) + 1e-6)

    # Z-normalize each column independently
    means = feats.mean(axis=0)
    stds = feats.std(axis=0) + 1e-8
    feats_norm = (feats - means) / stds

    return torch.tensor(feats_norm, dtype=torch.float32)


def warm_start_with_extra_input_dims(model, baseline_ckpt_path, n_extra_dims):
    """Warm-start when the target model has n_extra_dims more input columns
    than the baseline's W_ih. Baseline columns are copied; extra columns are
    zero-initialized so the augmented model starts identical to baseline on
    the original inputs.
    """
    ckpt = torch.load(baseline_ckpt_path, map_location="cpu", weights_only=True)
    own = model.state_dict()

    src_W_ih = ckpt.get("lstm.weight_ih_l0")
    dst_W_ih = own["lstm_cell.weight_ih"]
    if src_W_ih is not None and src_W_ih.shape[1] + n_extra_dims == dst_W_ih.shape[1]:
        with torch.no_grad():
            dst_W_ih[:, :src_W_ih.shape[1]].copy_(src_W_ih)
            dst_W_ih[:, src_W_ih.shape[1]:].zero_()
        LOGGER.info(f"  Partial warm-start: copied {src_W_ih.shape[1]} cols of W_ih, "
                     f"zero-init {n_extra_dims} new cols")
    elif src_W_ih is not None and src_W_ih.shape == dst_W_ih.shape:
        own["lstm_cell.weight_ih"].copy_(src_W_ih)
        LOGGER.info(f"  Warm-start: W_ih shapes match, full copy")
    else:
        LOGGER.warning(f"  W_ih shape mismatch ({src_W_ih.shape if src_W_ih is not None else 'missing'} "
                        f"vs {dst_W_ih.shape}); skipping W_ih warm-start")

    for src, dst in [("lstm.weight_hh_l0", "lstm_cell.weight_hh"),
                       ("lstm.bias_ih_l0", "lstm_cell.bias_ih"),
                       ("lstm.bias_hh_l0", "lstm_cell.bias_hh"),
                       ("head.net.0.weight", "head.weight"),
                       ("head.net.0.bias", "head.bias")]:
        if src in ckpt and dst in own and ckpt[src].shape == own[dst].shape:
            own[dst].copy_(ckpt[src])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--basin-file", default=str(BASIN_FILE))
    parser.add_argument("--edge-file", default=str(EDGE_FILE))
    parser.add_argument("--baseline-run", default=None,
                        help="Path to NH baseline run (auto-detected if omitted)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 2 epochs and report per-epoch wall-clock")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--tag", default="",
                        help="Extra tag for the run directory name")
    parser.add_argument("--run-dir", default=None,
                        help="Override the auto-generated run dir path. "
                              "Used by the 5cond_factorial sweep to write into runs/5cond_factorial/<cond>_seed<N>/.")
    parser.add_argument("--use-compile", action="store_true",
                        help="Wrap the DirectedGraphLSTM with torch.compile for ~2-3x training speedup. "
                              "Requires PyTorch >= 2.0. Falls back to uncompiled with a warning if unavailable.")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    variant = VARIANTS[args.variant]
    basin_file = Path(args.basin_file)
    edge_file = Path(args.edge_file)

    # Resolve baseline
    if args.baseline_run:
        baseline_run = Path(args.baseline_run)
    else:
        baseline_run = find_component0_baseline()
    if baseline_run is None:
        LOGGER.error("No Component 0 baseline run found. Train it first with "
                      "lstm_component0_baseline.yaml")
        sys.exit(1)
    LOGGER.info(f"Baseline run: {baseline_run}")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%d%m_%H%M%S")
        tag = f"c0_{args.variant}_seed{args.seed}"
        if args.smoke_test:
            tag += "_SMOKE"
        if args.tag:
            tag += f"_{args.tag}"
        run_dir = ROOT / f"runs/graph_{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {run_dir}")

    basin_ids = [l.strip() for l in open(basin_file) if l.strip()]
    n_basins = len(basin_ids)
    edges = load_graph_with_features(edge_file, basin_ids)
    LOGGER.info(f"Basins: {n_basins}   Edges: {len(edges)}")

    cfg = Config(baseline_run / "config.yml")
    scaler = load_scaler(baseline_run)

    id_to_int = load_basin_id_encoding(baseline_run) if cfg.use_basin_id_encoding else {}

    LOGGER.info("Loading train-period data...")
    x_d_train, x_s, y_train = load_basin_data(cfg, scaler, basin_ids, "train", id_to_int)
    LOGGER.info(f"Train tensor shape: {x_d_train.shape}   Static dim: {x_s.shape[1]}")
    LOGGER.info("Loading test-period data...")
    x_d_test, _, y_test = load_basin_data(cfg, scaler, basin_ids, "test", id_to_int)
    LOGGER.info(f"Test tensor shape: {x_d_test.shape}")

    # Condition B: append topology-derived static features.
    # We zero-pad them when ablating *just* the message passing — see comments.
    if variant["add_topology_features"]:
        topo_file = ROOT / "datasets/camels_us/camels_attributes_v2.0/camels_topo.txt"
        topo_feats = compute_topology_features(basin_ids, edge_file, topo_file)  # [n_basins, 5]
        LOGGER.info(f"Computed Condition-B topology features: {topo_feats.shape}; "
                     f"first row = {topo_feats[0].tolist()}")
        x_s = torch.cat([x_s, topo_feats], dim=-1)
        LOGGER.info(f"Augmented static dim: {x_s.shape[1]} (was {x_s.shape[1] - 5})")

    n_dyn = x_d_train.shape[3]
    input_size = n_dyn + x_s.shape[1]

    # Edges are present iff the variant explicitly says so (use_full_edges flag).
    # Conditions in the 5-condition factorial:
    #   G            (empty_graph)              -> edges=[]   (no graph)
    #   G+T          (topology_features)        -> edges=[]   (no graph; topology comes from static features)
    #   G+M          (warm)                     -> full edges (message passing)
    #   G+T+M        (full_graph_with_topology) -> full edges (message passing AND topology features)
    model_edges = edges if variant["use_full_edges"] else []

    model = DirectedGraphLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        edges=model_edges,
        n_basins=n_basins,
        n_targets=1,
        dropout=DROPOUT,
        initial_forget_bias=cfg.initial_forget_bias,
        use_edge_features=variant["edge_feat"],
        use_diff_term=variant["diff"],
        use_attention=variant["attn"],
        use_sigmoid_gate=variant["sigate"],
    ).to(DEVICE)
    LOGGER.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}   "
                 f"input_size: {input_size}   edges: {len(model_edges)}")

    if not args.no_warm_start:
        ckpts = sorted(baseline_run.glob("model_epoch*.pt"))
        if ckpts:
            LOGGER.info(f"Warm-starting from: {ckpts[-1]}")
            if variant["add_topology_features"]:
                # Augmented input dim — partial warm-start with zero-init for new dims.
                warm_start_with_extra_input_dims(model, ckpts[-1], n_extra_dims=5)
            else:
                warm_start_from_baseline(model, ckpts[-1])
        else:
            LOGGER.warning("No baseline checkpoint to warm-start from.")

    if variant["frozen"]:
        for p in model.lstm_cell.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = False
        LOGGER.info("Frozen LSTM + head; only W_msg_edge and W_out will train")

    trainable = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,} of "
                 f"{sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(trainable, lr=LR)

    # === Forward-pass optimization ===
    # Wrap the DirectedGraphLSTM with torch.compile if available and requested.
    # Locked decision (2026-05-06, idea1.md): the 5-condition factorial uses
    # torch.compile for ~2-3x training speedup. All 4 graph variants share the
    # same compiled architecture, preserving the architecture-matched control.
    if args.use_compile:
        try:
            torch_major = int(torch.__version__.split(".")[0])
            if torch_major < 2:
                LOGGER.warning(f"  torch.compile requires PyTorch >= 2.0 (have {torch.__version__}); "
                                f"falling back to uncompiled forward.")
            else:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                LOGGER.info("  Wrapped model with torch.compile(mode='reduce-overhead'). "
                             "First epoch will be slow (compilation); subsequent epochs ~2-3x faster.")
        except Exception as e:
            LOGGER.warning(f"  torch.compile failed ({e}); falling back to uncompiled forward.")

    # Pre-training eval (sanity: should ~match baseline if warm-start worked)
    LOGGER.info("Pre-training test NSE (sanity check):")
    pre = evaluate(model, x_d_test, x_s, y_test, basin_ids)
    pre_med = float(np.nanmedian(list(pre.values())))
    LOGGER.info(f"  median NSE = {pre_med:.3f}")

    epoch_times = []
    loss_history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, x_d_train, x_s, y_train, optimizer, BATCH_SIZE)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        loss_history.append(loss)
        LOGGER.info(f"Epoch {epoch}/{args.epochs}  loss={loss:.5f}  "
                     f"wall={elapsed:.1f}s")
        torch.save(model.state_dict(), run_dir / f"model_epoch{epoch:03d}.pt")

    # Final eval — capture raw obs/pred so the analysis script can compute
    # NSE / KGE / log-NSE consistently across all 5 conditions.
    final, obs_pred = evaluate_with_predictions(model, x_d_test, x_s, y_test, basin_ids)
    final_med = float(np.nanmedian(list(final.values())))
    final_mean = float(np.nanmean(list(final.values())))
    LOGGER.info(f"Final median NSE: {final_med:.3f}   mean: {final_mean:.3f}")

    pd.DataFrame([{"basin": b, "NSE": v} for b, v in final.items()]).to_csv(
        run_dir / "test_metrics.csv", index=False)

    # Long-format predictions: one row per (basin, timestep) for the test window.
    # ~275k rows for Component 0 (183 basins × ~1500 steps); ~10 MB CSV.
    pred_rows = []
    for basin, (obs, pred) in obs_pred.items():
        for i in range(len(obs)):
            pred_rows.append({"basin": basin, "step": i,
                               "obs": float(obs[i]), "pred": float(pred[i])})
    pd.DataFrame(pred_rows).to_csv(run_dir / "test_predictions.csv", index=False)
    LOGGER.info(f"Wrote {len(pred_rows)} prediction rows to test_predictions.csv")

    run_config = {
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "n_basins": n_basins,
        "n_edges": len(edges),
        "baseline_run": str(baseline_run),
        "smoke_test": bool(args.smoke_test),
        "pre_training_median_nse": pre_med,
        "final_median_nse": final_med,
        "final_mean_nse": final_mean,
        "loss_history": loss_history,
        "epoch_wall_seconds": epoch_times,
        "mean_epoch_wall_seconds": float(np.mean(epoch_times)),
        "input_size": input_size,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    LOGGER.info("=" * 70)
    LOGGER.info(f"DONE. variant={args.variant}  seed={args.seed}  median_nse={final_med:.3f}")
    LOGGER.info(f"Mean epoch wall-clock: {np.mean(epoch_times):.1f}s  "
                 f"({np.mean(epoch_times)/60:.1f} min)")
    if args.smoke_test:
        full_epochs = 15
        projected = full_epochs * np.mean(epoch_times)
        LOGGER.info(f"SMOKE-TEST PROJECTION: full run ({full_epochs} epochs) = "
                     f"{projected:.0f}s = {projected/60:.1f} min")
        LOGGER.info(f"  For 3 seeds × 3 variants = 9 runs: "
                     f"{9 * projected / 3600:.1f} hours total CPU")
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
