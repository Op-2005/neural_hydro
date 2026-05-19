"""Step 1 of the testing-framework-proposal.md ladder: matched-budget L control.

Trains NH `cudalstm` for EXACTLY 420 gradient updates (the total budget the
DirectedGraphLSTM trainer got over its 30-epoch sweep). Tests whether the
L − G NSE gap of +0.050 in the 5-condition factorial is a training-budget
confound or a real architecture confound.

Pre-registration: `experiments/5cond_factorial/preregistration_step1.md`.

Usage:
    python experiments/training/train_matched_budget_lstm.py \\
        --seed 11 13 17 \\
        --max-steps 420 \\
        --device cpu

Output:
    runs/5cond_factorial/L420_seed{N}/  with the standard NH layout
    (config.yml, model_epoch001.pt, test/model_epoch001/test_metrics.csv).
"""
import argparse
import subprocess
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
BASE_CONFIG = ROOT / "experiments" / "5cond_factorial" / "configs" / "L_seed11.yaml"
OUT_ROOT = ROOT / "runs" / "5cond_factorial"
CFG_DIR = ROOT / "experiments" / "5cond_factorial" / "configs"


def write_config(seed: int, max_steps: int, device: str) -> Path:
    """Build a 1-epoch, max_updates_per_epoch=`max_steps` cudalstm config."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment_name"] = f"L420_seed{seed}"
    cfg["epochs"] = 1
    cfg["max_updates_per_epoch"] = max_steps
    cfg["seed"] = seed
    cfg["device"] = device
    cfg["validate_every"] = 999  # disable mid-epoch validation
    out = CFG_DIR / f"L420_seed{seed}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def run_nh(cmd: list, log_prefix: str):
    """Run an NH command and stream its tail."""
    print(f"\n=== {log_prefix} ===")
    print(" ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    # Print last 8 lines of stderr+stdout for visibility
    out = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(out.rstrip().splitlines()[-8:])
    print(tail)
    if result.returncode != 0:
        print(f"!! NH command exited with code {result.returncode}")
    return result.returncode


def find_run_dir(experiment_name: str) -> Path:
    """NH creates runs/{run_dir}/{experiment_name}_{timestamp}/. Find the latest."""
    cands = sorted(OUT_ROOT.glob(f"{experiment_name}_*"))
    cands = [c for c in cands if c.is_dir()]
    if not cands:
        return None
    return cands[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs="+", default=[11, 13, 17])
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0", "mps"])
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for seed in args.seed:
        canonical = OUT_ROOT / f"L420_seed{seed}"

        # Skip if already complete
        if (canonical / "test" / "model_epoch001" / "test_metrics.csv").exists():
            print(f"[skip] L420_seed{seed} already complete")
            continue

        # Write config, train, then evaluate, then rename to canonical dir
        cfg_path = write_config(seed, args.max_steps, args.device)

        rc = run_nh(
            [sys.executable, "neuralhydrology/nh_run.py", "train",
             "--config-file", str(cfg_path)],
            f"TRAIN L420 seed={seed} max_steps={args.max_steps}",
        )
        if rc != 0:
            print(f"[abort] training failed for seed {seed}")
            continue

        # Find NH's timestamped folder and rename to canonical
        run_dir = find_run_dir(f"L420_seed{seed}")
        if run_dir is None:
            print(f"[abort] could not find produced run dir for seed {seed}")
            continue
        if canonical.exists():
            import shutil
            shutil.rmtree(canonical)
        run_dir.rename(canonical)
        print(f"  moved {run_dir.name} -> L420_seed{seed}/")

        rc = run_nh(
            [sys.executable, "neuralhydrology/nh_run.py", "evaluate",
             "--run-dir", str(canonical), "--epoch", "1"],
            f"EVAL L420 seed={seed}",
        )
        if rc != 0:
            print(f"[warn] evaluation failed for seed {seed}")

    # Quick headline summary
    print("\n\n========== L420 RESULTS ==========")
    import pandas as pd
    import numpy as np
    for seed in args.seed:
        f = OUT_ROOT / f"L420_seed{seed}" / "test" / "model_epoch001" / "test_metrics.csv"
        if f.exists():
            df = pd.read_csv(f, dtype={"basin": str})
            print(f"  L420_seed{seed}: n={len(df)}  "
                  f"median NSE={df['NSE'].median():.4f}  "
                  f"mean NSE={df['NSE'].mean():.4f}  "
                  f"median KGE={df.get('KGE', pd.Series([np.nan])).median():.4f}")
        else:
            print(f"  L420_seed{seed}: NO test_metrics.csv yet")


if __name__ == "__main__":
    main()
