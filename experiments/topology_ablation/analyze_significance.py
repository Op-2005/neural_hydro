"""Step A of the hardening chain — statistical significance of the deployable gain.

Zero training compute. Reads stored per-basin NSE (test_metrics.csv) for
L / L_upQpred / L_upQshuf across seeds [11,13,17], Component 0.

Tests (pre-registered in preregistration_hardening_chain.md):
  - realizable-vs-L: paired Wilcoxon signed-rank on per-basin (upQpred - L), pooled seeds.
  - realizable-vs-null: paired Wilcoxon on per-basin ((upQpred-L) - (upQshuf-L)) = (upQpred-upQshuf).
  - effect size: median Δ, fraction positive, bootstrap 95% CI on the median.
  - per-seed Wilcoxon (n=183 each) as robustness.

Writes analysis/SIGNIFICANCE.md.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
OUT = ROOT / "experiments" / "topology_ablation" / "analysis" / "SIGNIFICANCE.md"
SEEDS = [11, 13, 17]


def nse(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    df = pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"]
    return df


def boot_ci_median(x, n_boot=10000, seed=0):
    # deterministic bootstrap CI on the median (Date.now/random unavailable-safe: fixed seed)
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = np.array([np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def main():
    # pooled paired deltas across seeds, basin-aligned per seed
    real_minus_L = []   # upQpred - L
    null_minus_L = []   # upQshuf - L
    real_minus_null = []  # upQpred - upQshuf
    per_seed = {}
    for s in SEEDS:
        L = nse("L", s)
        P = nse("L_upQpred", s)
        N = nse("L_upQshuf", s)
        common = L.index.intersection(P.index).intersection(N.index)
        dRL = (P.loc[common] - L.loc[common]).values
        dNL = (N.loc[common] - L.loc[common]).values
        dRN = (P.loc[common] - N.loc[common]).values
        real_minus_L.extend(dRL)
        null_minus_L.extend(dNL)
        real_minus_null.extend(dRN)
        # per-seed Wilcoxon
        w_rl = wilcoxon(dRL, alternative="greater")
        w_rn = wilcoxon(dRN, alternative="greater")
        per_seed[s] = dict(n=len(common),
                           med_RL=float(np.median(dRL)), p_RL=float(w_rl.pvalue),
                           med_RN=float(np.median(dRN)), p_RN=float(w_rn.pvalue))

    real_minus_L = np.array(real_minus_L)
    null_minus_L = np.array(null_minus_L)
    real_minus_null = np.array(real_minus_null)

    # pooled Wilcoxon (one-sided: realizable improves)
    w_RL = wilcoxon(real_minus_L, alternative="greater")
    w_RN = wilcoxon(real_minus_null, alternative="greater")

    ci_RL = boot_ci_median(real_minus_L)
    ci_RN = boot_ci_median(real_minus_null)

    frac_pos_RL = float((real_minus_L > 0).mean())
    frac_pos_RN = float((real_minus_null > 0).mean())

    md = []
    md.append("# Step A — Statistical Significance of the Deployable Gain")
    md.append("")
    md.append("Zero training. Paired per-basin NSE deltas, pooled seeds [11,13,17], Component 0. "
              "Wilcoxon signed-rank, one-sided (H1: realizable improves). "
              "Pre-reg: `preregistration_hardening_chain.md`.")
    md.append("")
    md.append("## Pooled tests (n={} basin×seed)".format(len(real_minus_L)))
    md.append("")
    md.append("| Contrast | median Δ | frac basins + | Wilcoxon p (1-sided) | bootstrap 95% CI (median) |")
    md.append("|---|---|---|---|---|")
    md.append(f"| realizable − L (upQpred−L) | {np.median(real_minus_L):+.4f} | {frac_pos_RL:.2%} | "
              f"{w_RL.pvalue:.2e} | [{ci_RL[0]:+.4f}, {ci_RL[1]:+.4f}] |")
    md.append(f"| null − L (upQshuf−L) | {np.median(null_minus_L):+.4f} | {(null_minus_L>0).mean():.2%} | "
              f"— | — |")
    md.append(f"| **realizable − null (upQpred−upQshuf)** | {np.median(real_minus_null):+.4f} | "
              f"{frac_pos_RN:.2%} | **{w_RN.pvalue:.2e}** | [{ci_RN[0]:+.4f}, {ci_RN[1]:+.4f}] |")
    md.append("")

    pass_RL = w_RL.pvalue < 0.01
    pass_RN = w_RN.pvalue < 0.05
    md.append("## Pre-registered verdict")
    md.append("")
    md.append(f"- realizable-vs-L p < 0.01: **{pass_RL}** (p={w_RL.pvalue:.2e})")
    md.append(f"- realizable-vs-null p < 0.05: **{pass_RN}** (p={w_RN.pvalue:.2e})")
    md.append("")
    verdict = "PASS — deployable gain is significant AND separable from added-input capacity" \
        if (pass_RL and pass_RN) else \
        "PARTIAL/FAIL — see criteria; chain stops if realizable-vs-null not significant"
    md.append(f"**{verdict}**")
    md.append("")
    md.append("## Per-seed robustness (n=183 each; effect should hold within seeds, not only pooled)")
    md.append("")
    md.append("| seed | median (real−L) | p (real−L) | median (real−null) | p (real−null) |")
    md.append("|---|---|---|---|---|")
    for s in SEEDS:
        r = per_seed[s]
        md.append(f"| {s} | {r['med_RL']:+.4f} | {r['p_RL']:.2e} | {r['med_RN']:+.4f} | {r['p_RN']:.2e} |")

    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> wrote {OUT}")


if __name__ == "__main__":
    main()
