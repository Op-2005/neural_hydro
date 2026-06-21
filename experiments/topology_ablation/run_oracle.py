"""EXP-0: upstream-discharge ORACLE upper bound (the bounding experiment).

Two stock-cudalstm conditions on a network:
  L       baseline (5 static attrs, one-hot ON)
  L_upQ   baseline + `upstream_q` dynamic input (area-weighted mean of upstream
          basins' lagged observed discharge), via NH's additional_feature_files

If L_upQ does not beat L, no learned message passing can — structure is
uninformative for next-day downstream flow. If it does, message passing is
justified and the failure of static topology features was a representation issue.

Usage:
    python experiments/topology_ablation/run_oracle.py --network component0 \
        --seed 11 --device cpu --lag-days 1
"""
import argparse, subprocess, sys, shutil
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent.parent
BASE = ROOT / "experiments" / "5cond_factorial" / "configs" / "L_seed11.yaml"
CFG_DIR = Path(__file__).parent / "configs"
FEAT_DIR = Path(__file__).parent / "features"
RUN_ROOT = ROOT / "runs" / "topology_ablation"
BASE_STATIC = ["elev_mean", "area_gages2", "slope_mean", "p_mean", "pet_mean"]
DYN = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]


def basin_file_for(net):
    if net == "component0":
        return "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt"
    return f"experiments/local_subgraphs/basin_lists/{net}_basins.txt"


def write_cfg(net, seed, device, epochs, lag, with_upq):
    cfg = yaml.safe_load(open(BASE))
    cond = "L_upQ" if with_upq else "L"
    name = f"{cond}_{net}_seed{seed}"
    bf = basin_file_for(net)
    cfg.update(experiment_name=name, run_dir=f"runs/topology_ablation/{net}",
               train_basin_file=bf, validation_basin_file=bf, test_basin_file=bf,
               seed=seed, device=device, epochs=epochs, num_workers=2,
               static_attributes=BASE_STATIC, metrics=["NSE", "KGE"],
               use_basin_id_encoding=True)
    if with_upq:
        cfg["dynamic_inputs"] = DYN + ["upstream_q"]
        cfg["additional_feature_files"] = [str(FEAT_DIR / f"upstream_q_{net}_lag{lag}.p")]
    out = CFG_DIR / f"{name}.yaml"
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    return cond, name, out


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
    ap.add_argument("--lag-days", type=int, default=1)
    args = ap.parse_args()

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    for with_upq in (False, True):
        cond, name, cfg = write_cfg(args.network, args.seed, args.device, args.epochs,
                                    args.lag_days, with_upq)
        canonical = RUN_ROOT / args.network / f"{cond}_{args.network}_seed{args.seed}"
        if (canonical / "test" / f"model_epoch{args.epochs:03d}" / "test_metrics.csv").exists():
            print(f"[skip] {cond} {args.network} done")
            continue
        if run([sys.executable, "neuralhydrology/nh_run.py", "train", "--config-file", str(cfg)],
               f"{name} TRAIN") != 0:
            print(f"[abort] {name}")
            continue
        cands = [c for c in sorted((RUN_ROOT / args.network).glob(f"{name}_*")) if c.is_dir()]
        if cands:
            if canonical.exists():
                shutil.rmtree(canonical)
            cands[-1].rename(canonical)
        run([sys.executable, "neuralhydrology/nh_run.py", "evaluate", "--run-dir",
             str(canonical), "--epoch", str(args.epochs)], f"{name} EVAL")

    # quick contrast
    import pandas as pd, numpy as np
    def nse(cond):
        p = RUN_ROOT / args.network / f"{cond}_{args.network}_seed{args.seed}" / "test" / f"model_epoch{args.epochs:03d}" / "test_metrics.csv"
        return pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"] if p.exists() else None
    L, U = nse("L"), nse("L_upQ")
    if L is not None and U is not None:
        b = L.index.intersection(U.index)
        d = (U.loc[b] - L.loc[b]).dropna().values
        print(f"\n=== ORACLE RESULT ({args.network}, seed {args.seed}) ===")
        print(f"  L     median NSE: {L.median():+.3f}")
        print(f"  L_upQ median NSE: {U.median():+.3f}")
        print(f"  upQ − L paired: median {np.median(d):+.4f}  mean {np.mean(d):+.4f}  "
              f"frac>0 {np.mean(d>0):.2f}  frac>+0.02 {np.mean(d>0.02):.2f}  (n={len(d)})")
        print(f"  PRE-REG: success if median Δ >= +0.02 & mostly positive; "
              f"falsify if <= +0.005")


if __name__ == "__main__":
    main()
