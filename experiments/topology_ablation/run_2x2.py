"""Run the encoding x topology 2x2 on one or more networks (stock NH cudalstm).

For each network: generate the 4 configs, train + evaluate each (single seed by
default), writing to runs/topology_ablation/<network>/<cond>_<network>_seed<N>/.
Idempotent: skips conditions whose test_metrics.csv already exists.

Usage:
    python experiments/topology_ablation/run_2x2.py \
        --networks component0 sg_northeast sg_ohio \
        --seed 11 --device cuda:0
"""
import argparse
import subprocess
import sys
import glob
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
RUN_ROOT = ROOT / "runs" / "topology_ablation"
CFG_DIR = Path(__file__).parent / "configs"
MAKE = Path(__file__).parent / "make_configs.py"

# Where to find each network's basin file.
def basin_file_for(network):
    if network == "component0":
        return "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt"
    # subgraphs live in local_subgraphs/basin_lists
    return f"experiments/local_subgraphs/basin_lists/{network}_basins.txt"

CONDITIONS = ["L", "L_T", "L_noID", "L_noID_T"]


def run(cmd, label):
    """Stream the subprocess output live (so the cell never looks stalled), while
    keeping only the lines we care about (epoch loss / metrics / errors)."""
    import sys as _sys
    print(f"\n=== {label} ===", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    keep = ("Epoch", "average loss", "validation", "NSE", "KGE",
            "Stored", "Error", "error", "Traceback", "Wrote", "median")
    for line in proc.stdout:
        s = line.rstrip()
        if any(k in s for k in keep):
            print("   " + s, flush=True)
    proc.wait()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    for net in args.networks:
        bf = basin_file_for(net)
        assert (ROOT / bf).exists(), f"missing basin file {bf}"
        # (re)generate the 4 configs for this network/seed
        run([sys.executable, str(MAKE), "--network", net, "--basin-file", bf,
             "--seed", str(args.seed), "--device", args.device, "--epochs", str(args.epochs)],
            f"configs {net} seed={args.seed}")

        for cond in CONDITIONS:
            canonical = RUN_ROOT / net / f"{cond}_{net}_seed{args.seed}"
            if (canonical / "test" / f"model_epoch{args.epochs:03d}" / "test_metrics.csv").exists():
                print(f"[skip] {cond} {net} seed={args.seed} done")
                continue
            cfg = CFG_DIR / f"{cond}_{net}_seed{args.seed}.yaml"
            rc = run([sys.executable, "neuralhydrology/nh_run.py", "train", "--config-file", str(cfg)],
                     f"{cond} {net} seed={args.seed} TRAIN")
            if rc != 0:
                print(f"[abort] train failed: {cond} {net}")
                continue
            # NH writes <run_dir>/<experiment_name>_<ts>/; rename to canonical
            cands = sorted((RUN_ROOT / net).glob(f"{cond}_{net}_seed{args.seed}_*"))
            cands = [c for c in cands if c.is_dir()]
            if cands:
                if canonical.exists():
                    shutil.rmtree(canonical)
                cands[-1].rename(canonical)
            run([sys.executable, "neuralhydrology/nh_run.py", "evaluate",
                 "--run-dir", str(canonical), "--epoch", str(args.epochs)],
                f"{cond} {net} seed={args.seed} EVAL")

    print("\nDone. Analyze with: python experiments/topology_ablation/analyze_2x2.py")


if __name__ == "__main__":
    main()
