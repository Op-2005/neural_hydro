"""Local-subgraph sweep: train a small condition set on each local subgraph,
report the 3-seed loss-distribution invariant (mean +/- std NSE).

This is the Phase 2+3 driver from
experiments/5cond_factorial/analysis/post_meeting_plan.md, but implemented as a
self-contained batch following the professor's prescription:
  - small local subgraphs (built by build_local_subgraphs.py)
  - 3-seed mean +/- std as the tracked invariant
  - runs on the order of 5-15 min each so we can iterate

Conditions per subgraph (the minimal informative set):
  L      — NH cudalstm baseline (field standard)
  G      — DirectedGraphLSTM, empty edges (architecture-matched control)
  G+T+M  — DirectedGraphLSTM, full edges + topology features (the full model)

(G+T and G+M are dropped from the per-subgraph default to keep runs fast; they
can be added back per-subgraph via --full-factorial once a subgraph is chosen
for the deeper Phase-3 dive.)

Output dir: runs/local_subgraphs/<subgraph>/<cond>_seed<N>/

Usage (per subgraph, on Colab):
    python experiments/local_subgraphs/run_subgraph_sweep.py \
        --subgraph sg_northeast --seeds 11 13 17 \
        --conditions L G G_T_M --device cuda:0 --use-compile
"""
import argparse
import subprocess
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
BASIN_DIR = Path(__file__).parent / "basin_lists"
RUN_ROOT = ROOT / "runs" / "local_subgraphs"
CFG_DIR = Path(__file__).parent / "configs"
BASE_L_CONFIG = ROOT / "experiments" / "5cond_factorial" / "configs" / "L_seed11.yaml"
GRAPH_TRAINER = ROOT / "experiments" / "training" / "train_graph_component0.py"

# Condition -> (kind, variant_flag). kind is "nh" (cudalstm via nh_run) or
# "graph" (DirectedGraphLSTM via train_graph_component0.py).
CONDITIONS = {
    "L":     ("nh",    None),
    "G":     ("graph", "empty_graph"),
    "G_T":   ("graph", "topology_features"),
    "G_M":   ("graph", "warm"),
    "G_T_M": ("graph", "full_graph_with_topology"),
}


def write_L_config(subgraph, seed, device):
    """Build a cudalstm config pointed at the subgraph basin list."""
    with open(BASE_L_CONFIG) as f:
        cfg = yaml.safe_load(f)
    basin_file = f"experiments/local_subgraphs/basin_lists/{subgraph}_basins.txt"
    cfg["experiment_name"] = f"L_{subgraph}_seed{seed}"
    cfg["train_basin_file"] = basin_file
    cfg["validation_basin_file"] = basin_file
    cfg["test_basin_file"] = basin_file
    cfg["seed"] = seed
    cfg["device"] = device
    cfg["run_dir"] = f"runs/local_subgraphs/{subgraph}"
    out = CFG_DIR / f"L_{subgraph}_seed{seed}.yaml"
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def run(cmd, label):
    print(f"\n=== {label} ===")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print("\n".join(out.rstrip().splitlines()[-6:]))
    return r.returncode


def train_L(subgraph, seed, device):
    import glob, shutil, os
    canonical = RUN_ROOT / subgraph / f"L_seed{seed}"
    if (canonical / "test" / "model_epoch030" / "test_metrics.csv").exists():
        print(f"[skip] L {subgraph} seed={seed} done")
        return
    cfg = write_L_config(subgraph, seed, device)
    rc = run([sys.executable, "neuralhydrology/nh_run.py", "train",
              "--config-file", str(cfg)], f"L {subgraph} seed={seed} TRAIN")
    if rc != 0:
        return
    # NH writes runs/local_subgraphs/<sg>/L_<sg>_seed<N>_<ts>/; rename to canonical
    cands = sorted((RUN_ROOT / subgraph).glob(f"L_{subgraph}_seed{seed}_*"))
    cands = [c for c in cands if c.is_dir()]
    if cands:
        if canonical.exists():
            shutil.rmtree(canonical)
        cands[-1].rename(canonical)
    run([sys.executable, "neuralhydrology/nh_run.py", "evaluate",
         "--run-dir", str(canonical), "--epoch", "30"], f"L {subgraph} seed={seed} EVAL")


def train_graph(subgraph, seed, variant_flag, cond_name, device, use_compile):
    import os, shutil
    run_dir = RUN_ROOT / subgraph / f"{cond_name}_seed{seed}"
    if (run_dir / "test_metrics.csv").exists():
        print(f"[skip] {cond_name} {subgraph} seed={seed} done")
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    basin_file = BASIN_DIR / f"{subgraph}_basins.txt"
    edge_file = BASIN_DIR / f"{subgraph}_edges.csv"
    # Graph variants need a Component-0-scale baseline for cfg+scaler; use the
    # subgraph's own L run (same basins, so id_to_int covers them).
    baseline = RUN_ROOT / subgraph / f"L_seed{seed}"
    if not baseline.exists():
        # fall back to any completed L run for this subgraph
        import glob
        cands = sorted((RUN_ROOT / subgraph).glob("L_seed*"))
        cands = [c for c in cands if c.is_dir() and list(c.glob("train_data/*"))]
        if not cands:
            print(f"[abort] no L baseline for {subgraph}; train L first")
            return
        baseline = cands[0]
    cmd = [sys.executable, str(GRAPH_TRAINER),
           "--variant", variant_flag, "--seed", str(seed),
           "--no-warm-start", "--epochs", "30",
           "--run-dir", str(run_dir),
           "--basin-file", str(basin_file),
           "--edge-file", str(edge_file),
           "--baseline-run", str(baseline)]
    if use_compile:
        cmd.append("--use-compile")
    run(cmd, f"{cond_name} {subgraph} seed={seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgraph", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17])
    parser.add_argument("--conditions", nargs="+", default=["L", "G", "G_T_M"],
                        choices=list(CONDITIONS.keys()))
    parser.add_argument("--device", default="cpu",
                        help="cpu (default — these subgraphs are small enough to run "
                              "locally in minutes) or cuda:0 for the optional scale-up path.")
    parser.add_argument("--use-compile", action="store_true")
    args = parser.parse_args()

    (RUN_ROOT / args.subgraph).mkdir(parents=True, exist_ok=True)

    # Always train L first (graph variants depend on it for cfg+scaler)
    ordered = (["L"] if "L" in args.conditions else []) + \
              [c for c in args.conditions if c != "L"]

    for cond in ordered:
        kind, variant_flag = CONDITIONS[cond]
        for seed in args.seeds:
            if kind == "nh":
                train_L(args.subgraph, seed, args.device)
            else:
                train_graph(args.subgraph, seed, variant_flag, cond,
                            args.device, args.use_compile)

    print(f"\nDone with subgraph {args.subgraph}. "
           f"Run analyze_subgraphs.py for the invariant table.")


if __name__ == "__main__":
    main()
