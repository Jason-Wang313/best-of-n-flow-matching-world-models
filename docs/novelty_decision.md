# Novelty Decision

## Decision

Use the architecture-specific diagnostic angle, not a broad algorithmic novelty claim.

The final paper angle is:

> Best-of-N inference over conditional rectified-flow trajectory generators can convert small proxy-score errors into off-manifold trajectory selection. A lightweight uncertainty penalty, calibrated from training trajectory features, reduces the effect in a synthetic world-model setting.

## Why This Angle

The literature review weakens any claim that Best-of-N reward hacking is new. It also weakens any claim that flow matching alone creates a unique pathology. What remains defensible is the measurement interface:

- flow/rectified-flow models generate whole continuous trajectories by integrating a learned vector field;
- Best-of-N reranking evaluates many complete futures under a proxy;
- the selected futures can be physically invalid, low-diversity, or far from the expert trajectory manifold even when their proxy scores improve.

## Theory Inclusion

Include only a small proposition:

If the absolute proxy error for each candidate is bounded by beta times an uncertainty score, then choosing the candidate that maximizes proxy minus beta times uncertainty gives a true-return lower bound controlled by the uncertainty of the selected and best candidates. This is a standard robust-selection argument, not a novel theorem.

## Repair Inclusion

Include the uncertainty-penalized scorer because the smoke experiment supports it directionally. Do not claim it is optimal. It is a sanity-check baseline that makes the diagnostic actionable.

## Next Experiments Needed

- Compare against diffusion trajectory models with the same proxy and candidate budget.
- Replace hand-built uncertainty with ensembles or flow-consistency residuals.
- Evaluate on D4RL-style offline control or simple MuJoCo tasks.
- Sweep proxy quality and candidate count to estimate the overoptimization curve.
