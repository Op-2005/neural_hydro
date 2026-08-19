# TOST — distance control vs true network (equivalence)

Regenerates the equivalence numbers quoted in the Results and Limitations sections from
stored per-basin test metrics. Paired per-basin Delta NSE, connected basins, seeds 11/13/17.

- n (basin-seed pairs): **450**
- paired median (distance control - true network): **-0.0018**
- bootstrap 95% CI on the median (2000 resamples, seeded): **[-0.0066, +0.0050]**
- TOST against a +/-0.01 NSE margin: lower p = **2.53e-04**, upper p = **1.35e-03**
- equivalent at alpha=0.05: **True**

Reading: both one-sided tests reject a difference larger than the margin, so the two
conditions are statistically equivalent within +/-0.01 NSE rather than merely
indistinguishable. The margin is pre-specified at half the deployable effect size.
