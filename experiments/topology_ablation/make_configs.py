"""Generate the encoding x topology 2x2 configs for a given basin set.

The 2x2 (all STOCK NH cudalstm — identical trainer, only config differs):

                       topology OFF        topology ON
  basin one-hot ON     L                   L_T
  basin one-hot OFF    L_noID              L_noID_T

Key contrasts:
  (L_T  - L)           does topology help the standard (identity-encoded) model?
                       Predict ~0: the 671-dim one-hot already memorizes basin identity,
                       so 5 topology scalars are redundant.
  (L_noID_T - L_noID)  does topology help when the model CANNOT memorize identity?
                       Predict > 0: this is the regime where network position carries
                       non-redundant signal. THE headline contrast.
  interaction          (L_T - L) - (L_noID_T - L_noID): is the topology benefit
                       modulated by the encoding? Predict negative (topology helps more
                       when one-hot is off). Connects to GNN theory (Kipf-Welling):
                       structure helps most in the can't-memorize regime.

Usage:
    python experiments/topology_ablation/make_configs.py \
        --network component0 \
        --basin-file topology_analysis/phase1_network_discovery/outputs/component0_basins.txt \
        --seed 11 --device cuda:0
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
BASE = ROOT / "experiments" / "5cond_factorial" / "configs" / "L_seed11.yaml"
CFG_DIR = Path(__file__).parent / "configs"

BASE_STATIC = ["elev_mean", "area_gages2", "slope_mean", "p_mean", "pet_mean"]
TOPO_STATIC = ["graph_depth", "n_upstream", "total_upstream_area", "in_degree", "frac_upstream_area"]

# (condition, use_basin_id_encoding, include_topology)
CONDITIONS = [
    ("L",        True,  False),
    ("L_T",      True,  True),
    ("L_noID",   False, False),
    ("L_noID_T", False, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, help="label, e.g. component0 or sg_northeast")
    ap.add_argument("--basin-file", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(BASE))

    written = []
    for cond, use_oh, use_topo in CONDITIONS:
        cfg = dict(base)
        name = f"{cond}_{args.network}_seed{args.seed}"
        cfg["experiment_name"] = name
        cfg["run_dir"] = f"runs/topology_ablation/{args.network}"
        cfg["train_basin_file"] = args.basin_file
        cfg["validation_basin_file"] = args.basin_file
        cfg["test_basin_file"] = args.basin_file
        cfg["seed"] = args.seed
        cfg["device"] = args.device
        cfg["epochs"] = args.epochs
        cfg["use_basin_id_encoding"] = use_oh
        cfg["static_attributes"] = BASE_STATIC + (TOPO_STATIC if use_topo else [])
        cfg["metrics"] = ["NSE", "KGE"]
        out = CFG_DIR / f"{name}.yaml"
        yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
        written.append(out.name)

    print(f"Wrote {len(written)} configs for network={args.network} seed={args.seed}:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
