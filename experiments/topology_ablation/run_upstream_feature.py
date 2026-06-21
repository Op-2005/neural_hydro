"""Train stock cudalstm with one upstream-feature pickle as an extra dynamic input.
Generic runner for the post-oracle chain (shuffled-Q null, upstream-precip, lag sweep).

Usage:
    python experiments/topology_ablation/run_upstream_feature.py \
        --network component0 --seed 11 --device cpu \
        --feature-file experiments/topology_ablation/features/upstream_q_shuffled_component0_lag1.p \
        --cond-name L_upQshuf
"""
import argparse, subprocess, sys, shutil
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent.parent
BASE = ROOT / "experiments" / "5cond_factorial" / "configs" / "L_seed11.yaml"
CFG_DIR = Path(__file__).parent / "configs"
RUN_ROOT = ROOT / "runs" / "topology_ablation"
BASE_STATIC = ["elev_mean", "area_gages2", "slope_mean", "p_mean", "pet_mean"]
DYN = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]


def basin_file_for(net):
    if net == "component0":
        return "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt"
    return f"experiments/local_subgraphs/basin_lists/{net}_basins.txt"


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
    ap.add_argument("--network", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--feature-file", required=True)
    ap.add_argument("--cond-name", required=True, help="e.g. L_upQshuf, L_upPrecip, L_upQ_lag0")
    args = ap.parse_args()

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(BASE))
    name = f"{args.cond_name}_{args.network}_seed{args.seed}"
    bf = basin_file_for(args.network)
    cfg.update(experiment_name=name, run_dir=f"runs/topology_ablation/{args.network}",
               train_basin_file=bf, validation_basin_file=bf, test_basin_file=bf,
               seed=args.seed, device=args.device, epochs=args.epochs, num_workers=2,
               static_attributes=BASE_STATIC, metrics=["NSE", "KGE"],
               use_basin_id_encoding=True,
               dynamic_inputs=DYN + ["upstream_q"],
               additional_feature_files=[str(Path(args.feature_file).resolve())])
    out = CFG_DIR / f"{name}.yaml"
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)

    canonical = RUN_ROOT / args.network / name
    if (canonical / "test" / f"model_epoch{args.epochs:03d}" / "test_metrics.csv").exists():
        print(f"[skip] {name} done")
        return
    if run([sys.executable, "neuralhydrology/nh_run.py", "train", "--config-file", str(out)],
           f"{name} TRAIN") != 0:
        print(f"[abort] {name}"); return
    cands = [c for c in sorted((RUN_ROOT / args.network).glob(f"{name}_*")) if c.is_dir()]
    if cands:
        if canonical.exists():
            shutil.rmtree(canonical)
        cands[-1].rename(canonical)
    run([sys.executable, "neuralhydrology/nh_run.py", "evaluate", "--run-dir",
         str(canonical), "--epoch", str(args.epochs)], f"{name} EVAL")
    print(f"DONE {name}")


if __name__ == "__main__":
    main()
