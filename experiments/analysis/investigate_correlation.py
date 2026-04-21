"""Investigate: WHY does the graph help basins the baseline already predicts well?

Two hypotheses to test:
  H1: The downstream basin's own baseline quality matters (the correlation we already saw)
  H2: The upstream parents' baseline quality matters (graph messages are more useful when
      the upstream model is good)

These are not mutually exclusive. If H2 is also strong, it suggests the graph mechanism
is "clean upstream signal helps", which is physically meaningful. If H2 is weak, then the
correlation with downstream NSE is more about the LOCAL learnability of the basin.

Also: check train vs test NSE to detect overfitting.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is three levels up from experiments/analysis/<script>.py.
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ROOT = Path(__file__).parent.parent.parent

# Load per-basin data
df = pd.read_csv(ROOT / "experiments" / "analysis_outputs" / "per_basin_analysis.csv",
                  dtype={"basin": str})
edges = pd.read_csv(ROOT / "topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv",
                     dtype={"parent_id": str, "child_id": str})

# Build parent mapping
parents_of = {}
for _, row in edges.iterrows():
    parents_of.setdefault(row["child_id"], []).append(row["parent_id"])

# For each basin with parents, compute:
#   - own baseline NSE
#   - avg parents' baseline NSE
#   - min parents' baseline NSE (weakest link)
#   - max parents' baseline NSE (strongest upstream signal)
basin_to_nse = {row["basin"]: row["nse_baseline"] for _, row in df.iterrows()}

analysis = []
for _, row in df.iterrows():
    bid = row["basin"]
    if bid not in parents_of:
        continue
    parent_nses = [basin_to_nse[p] for p in parents_of[bid] if p in basin_to_nse]
    if not parent_nses:
        continue
    analysis.append({
        "basin": bid,
        "depth": row["depth"],
        "n_upstream": row["n_upstream"],
        "own_baseline_nse": row["nse_baseline"],
        "delta_graph": row["delta"],
        "delta_frozen": row["delta_frozen"],
        "parents_mean_nse": np.mean(parent_nses),
        "parents_min_nse": np.min(parent_nses),
        "parents_max_nse": np.max(parent_nses),
        "parents_range_nse": np.max(parent_nses) - np.min(parent_nses),
    })

adf = pd.DataFrame(analysis)

print("=" * 75)
print("HYPOTHESIS TEST: Does graph benefit depend on upstream parent quality?")
print("=" * 75)
print()

for delta_col in ["delta_graph", "delta_frozen"]:
    print(f"\n--- {delta_col} (Graph - Baseline per basin) ---")
    print("Correlations:")
    for prop in ["own_baseline_nse", "parents_mean_nse", "parents_min_nse",
                  "parents_max_nse", "parents_range_nse", "depth", "n_upstream"]:
        r = adf[delta_col].corr(adf[prop])
        marker = "  <-- strong" if abs(r) > 0.5 else ""
        print(f"  {prop:25s}: r = {r:+.3f}{marker}")

# Per-basin table
print("\n\n" + "=" * 75)
print("Per-basin detail (sorted by delta_graph):")
print("=" * 75)
cols = ["basin", "depth", "n_upstream", "own_baseline_nse",
         "parents_mean_nse", "parents_min_nse", "delta_graph", "delta_frozen"]
print(adf[cols].sort_values("delta_graph", ascending=False).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# Multivariate: can we predict delta from own_NSE alone, vs both own + parents?
from numpy.linalg import lstsq

X1 = adf[["own_baseline_nse"]].values
X2 = adf[["own_baseline_nse", "parents_mean_nse"]].values
y = adf["delta_graph"].values

def rsq(X, y):
    X_aug = np.column_stack([X, np.ones(len(X))])
    coef, _, _, _ = lstsq(X_aug, y, rcond=None)
    pred = X_aug @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, coef

r2_own, coef_own = rsq(X1, y)
r2_both, coef_both = rsq(X2, y)
print(f"\n\nLinear regression R²:")
print(f"  delta_graph ~ own_baseline_NSE:         R² = {r2_own:.3f}")
print(f"    coef: own={coef_own[0]:+.3f}  intercept={coef_own[1]:+.3f}")
print(f"  delta_graph ~ own_NSE + parents_mean:   R² = {r2_both:.3f}")
print(f"    coef: own={coef_both[0]:+.3f}  parents_mean={coef_both[1]:+.3f}  intercept={coef_both[2]:+.3f}")
print(f"  Marginal R² contribution of parents: +{r2_both - r2_own:.3f}")
