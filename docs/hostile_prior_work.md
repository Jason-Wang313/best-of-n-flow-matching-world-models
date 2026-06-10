# Hostile Prior-Work Review

This document argues against the project as if reviewing it skeptically.

## Most Damaging Prior Work

**Reward model overoptimization already explains the main effect.** Gao, Schulman, and Hilton explicitly study Best-of-N sampling against a proxy reward model. If this repo only shows that selecting the maximum proxy score can hurt true reward, the contribution is not new.

**Regularized Best-of-N already suggests the repair.** Work on regularized Best-of-N and minimum Bayes risk reranking shows that adding a penalty to reranking can mitigate reward hacking. An uncertainty penalty in trajectory-feature space is best described as a domain adaptation of this idea.

**Diffusion planners already optimize generated trajectories.** Diffuser and Decision Diffuser show that generative trajectory models can be sampled, guided, or conditioned. A claim about "generative world models" must separate reranking-specific pathologies from ordinary planning errors.

**World-model RL has stronger baselines.** DreamerV3, TD-MPC2, EfficientZero, and MuZero-style methods are more mature than the toy setup here. The repo cannot claim algorithmic superiority over model-based RL.

## Claims To Avoid

- Avoid: "Best-of-N breaks flow-matching world models."
- Avoid: "Uncertainty-penalized reranking solves reward hacking."
- Avoid: "Rectified flow has a unique Goodhart failure."
- Avoid: "The synthetic point-mass experiment predicts robotics or video-model behavior."

## Claims That Survive

- Best-of-N can amplify proxy error in a conditional rectified-flow trajectory generator.
- The amplification is measurable with true-vs-predicted return gap, feature-space OOD distance, and selected-mode entropy.
- In this synthetic setting, an uncertainty penalty reduces proxy exploitation relative to naive proxy-only selection.
- The right next experiment is to replace hand-built features with learned uncertainty or model-consistency diagnostics on a standard offline-control benchmark.

## Reviewer Questions

1. Is the proxy model too artificial?
   Yes. It is intentionally misspecified to isolate the mechanism. A stronger paper needs learned value functions with held-out true returns.

2. Is the repair just regularized Best-of-N?
   Mostly yes. The contribution is not the abstract penalty; it is the domain-specific diagnostic that connects penalty to trajectory-manifold distance.

3. Is the flow model doing anything a diffusion model would not?
   Not yet. The experiment supports a flow-matching instance of a broader generative-reranking risk. It does not prove uniqueness.

4. Is the paper ICLR-ready?
   As a first pass, no. It is a runnable pilot and paper draft. It needs larger benchmarks, ablations, and at least one non-synthetic setting.
