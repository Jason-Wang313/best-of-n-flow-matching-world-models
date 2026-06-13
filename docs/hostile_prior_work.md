# Hostile Prior-Work Review

This document argues against the project as if reviewing it skeptically.

## Most Damaging Prior Work

**Reward-model overoptimization already explains the broad effect.** If the paper only says "a proxy can be overoptimized," it is not novel. The manuscript must keep the contribution tied to rectified-flow trajectory audits and selected-future diagnostics.

**Regularized reranking already suggests the repair.** A penalty on proxy scores is not a new algorithmic idea. The paper should treat feature-space calibration as a sanity-check baseline, not as the central method.

**Diffusion planners and trajectory generators are close neighbors.** Diffuser, Decision Diffuser, and related trajectory-generation methods already sample or guide trajectories. The paper must avoid claiming a general generative-planning result without a diffusion baseline.

**World-model RL has stronger baselines.** DreamerV3, TD-MPC2, EfficientZero, and MuZero-style systems are much more mature than this synthetic audit. The paper should not present itself as a competitive model-based RL algorithm.

## Claims To Avoid

- Avoid: "Rectified-flow world models fail under large candidate budgets."
- Avoid: "Feature-space calibration solves proxy exploitation."
- Avoid: "This is a new planning algorithm."
- Avoid: "The point-mass evidence predicts robotics, video, or pixel-control behavior."

## Claims That Survive

- Proxy-selected upper-tail futures can be worse than budget-invariant first candidates in a conditional rectified-flow trajectory generator.
- The failure is measurable through held-out true return, proxy-realized gap, feature-space OOD distance, selection bias, and mode entropy.
- In the five-seed synthetic audit, the calibrated tail improves true return and reduces gap and OOD distance relative to the proxy tail.
- The artifact is a reproducible tail-value audit that future work can scale to learned uncertainty and standard control benchmarks.

## Reviewer Questions

1. Is the proxy model too artificial?
   Yes, by design. It isolates omitted-feature proxy exploitation. The paper scopes claims accordingly and calls for learned value models in future work.

2. Is the repair just regularized reranking?
   Abstractly yes. The paper presents it only as a diagnostic baseline tied to trajectory-manifold distance.

3. Is the flow model doing anything a diffusion model would not?
   Not proven. The experiment supports a rectified-flow instance of a broader generative-reranking risk. The v3 Euler-step sweep does add a flow-specific negative control: proxy-tail harm persists when increasing integration steps, and residual-only reranking lowers step sensitivity without solving true return.

4. Is the paper submission-ready?
   The rewritten version has a distinct framing, repeated-seed evidence, tests, a claim audit, and explicit limitations. The remaining weakness is scope: it is still synthetic and should not overclaim beyond the audit setting.
