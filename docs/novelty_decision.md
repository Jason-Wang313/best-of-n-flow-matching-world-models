# Novelty Decision

## Decision

Use the rectified-flow tail-audit angle, not a broad large-candidate selection claim.

The final paper angle is:

> Tail-value audits reveal when proxy-selected upper-tail futures from a conditional rectified-flow trajectory generator move off the training manifold and lose realized return. A simple feature-space calibration penalty is a diagnostic baseline that reduces the failure in a five-seed synthetic audit.

## Why This Angle Survives

Prior reward-model overoptimization work already explains why proxy maximization can fail. Flow matching alone also does not create a unique pathology. What remains defensible is the measurement interface:

- the generator produces continuous trajectories by integrating a learned vector field;
- evaluation focuses on the selected upper tail rather than average sample quality;
- diagnostics connect proxy-realized gap, physical validity, training-manifold distance, and mode entropy;
- an Euler-step sweep separates proxy exploitation from coarse numerical integration artifacts;
- the claim is supported by a repeated-seed audit and a machine-checkable claim gate.

## Theory Inclusion

Include only a small calibration sanity check:

If proxy error is bounded by a feature-distance uncertainty score, then selecting by proxy minus the matching uncertainty penalty gives a true-return comparison controlled by the selected and comparison candidates' uncertainty. This is a standard robust-selection lens, not a novel theorem.

## Repair Inclusion

Include the calibrated selector because the five-seed smoke audit supports it directionally. Do not claim it is optimal, architecture-complete, or sufficient for robotics-scale planning. It is a baseline that makes the diagnostic actionable.

## Next Experiments Needed

- Compare against diffusion trajectory generators under the same proxy.
- Replace hand-built feature distance with ensembles, conformal scores, or flow-consistency residuals.
- Evaluate on D4RL-style offline-control tasks.
- Sweep proxy quality and candidate count to estimate an overoptimization curve.
