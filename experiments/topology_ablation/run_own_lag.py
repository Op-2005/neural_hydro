"""Train stock cudalstm with the own-lagged-discharge inputs (camera-ready reviewer request).

Generalises run_upstream_feature.py, which assumes exactly one extra column named `upstream_q`.
Here the appended inputs are named explicitly, so a condition can take `self_q`, `upstream_q`, or
both. Everything else — base config, static attributes, basin file, epochs, one-hot, metrics — is
byte-identical to every published condition.

Usage (see preregistrations/own_lag.md):
    python experiments/topology_ablation/run_own_lag.py --seed 11 --device cuda:0 \
        --cond-name L_selfQ \
        --extra-input self_q --feature-file experiments/topology_ablation/features/self_q_component0_lag1.p

    python experiments/topology_ablation/run_own_lag.py --seed 11 --device cuda:0 \
        --cond-name L_selfQ_upQ \
        --extra-input self_q     --feature-file .../self_q_component0_lag1.p \
        --extra-input upstream_q --feature-file .../upstream_q_component0_lag1.p
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
BASE = ROOT / "experiments" / "5cond_factorial" / "configs" / "L_seed11.yaml"
CFG_DIR = Path(__file__).parent / "configs"
RUN_ROOT = ROOT / "runs" / "topology_ablation"
BASIN_FILE = "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt"
BASE_STATIC = ["elev_mean", "area_gages2", "slope_mean", "p_mean", "pet_mean"]
DYN = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]


def run(cmd, label):
    print(f"\n=== {label} ===", flush=True)
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    keep = ("Epoch", "average loss", "validation", "NSE", "KGE", "Stored", "Error", "Traceback")
    for line in p.stdout:
        if any(k in line for k in keep):
            print("   " + line.rstrip(), flush=True)
    p.wait()
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--cond-name", required=True)
    ap.add_argument("--extra-input", action="append", required=True,
                    help="column name to append, repeatable (paired with --feature-file)")
    ap.add_argument("--feature-file", action="append", required=True,
                    help="feature file providing the matching --extra-input, repeatable")
    args = ap.parse_args()
    if len(args.extra_input) != len(args.feature_file):
        ap.error("--extra-input and --feature-file must be given the same number of times")

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(BASE))
    name = f"{args.cond_name}_component0_seed{args.seed}"
    cfg.update(experiment_name=name, run_dir="runs/topology_ablation/component0",
               train_basin_file=BASIN_FILE, validation_basin_file=BASIN_FILE,
               test_basin_file=BASIN_FILE,
               seed=args.seed, device=args.device, epochs=args.epochs, num_workers=2,
               static_attributes=BASE_STATIC, metrics=["NSE", "KGE"],
               use_basin_id_encoding=True,
               dynamic_inputs=DYN + list(args.extra_input),
               additional_feature_files=[str(Path(f).resolve()) for f in args.feature_file])
    out = CFG_DIR / f"{name}.yaml"
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    print(f"dynamic_inputs: {cfg['dynamic_inputs']}")

    canonical = RUN_ROOT / "component0" / name
    if (canonical / "test" / f"model_epoch{args.epochs:03d}" / "test_metrics.csv").exists():
        print(f"[skip] {name} done")
        return
    if run([sys.executable, "neuralhydrology/nh_run.py", "train", "--config-file", str(out)],
           f"{name} TRAIN") != 0:
        print(f"[abort] {name}")
        return
    cands = [c for c in sorted((RUN_ROOT / "component0").glob(f"{name}_*")) if c.is_dir()]
    if cands:
        if canonical.exists():
            shutil.rmtree(canonical)
        cands[-1].rename(canonical)
    run([sys.executable, "neuralhydrology/nh_run.py", "evaluate", "--run-dir",
         str(canonical), "--epoch", str(args.epochs)], f"{name} EVAL")
    print(f"DONE {name}")


if __name__ == "__main__":
    main()
