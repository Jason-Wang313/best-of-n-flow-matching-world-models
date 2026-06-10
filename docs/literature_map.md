# Literature Map

This map was written before choosing the final paper angle. The purpose is to identify what would make the project non-novel or overclaimed.

## Flow Matching And Rectified Flow

Flow Matching trains continuous normalizing flows by regressing vector fields along probability paths rather than simulating a diffusion process during training. Lipman et al. introduced the framework and showed that optimal-transport paths can improve training and sampling efficiency. Rectified Flow, from Liu, Gong, and Liu, is especially relevant because it learns straight transports between base and data distributions and can be sampled with a small number of ODE steps.

The immediate novelty pressure is that many failure modes of deterministic ODE samplers are not unique to rectified flow. Stochastic interpolants, probability-flow ODEs, consistency models, and diffusion guidance all blur the boundary between "flow-specific" and "generic generative model" claims. A defensible paper should therefore focus on the interface between flow-generated trajectories and inference-time selection, not on broad claims about flow matching itself.

Key sources:

- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023: https://arxiv.org/abs/2210.02747
- Liu et al., "Flow Straight and Fast", ICLR 2023: https://arxiv.org/abs/2209.03003
- Albergo and Vanden-Eijnden, "Building Normalizing Flows with Stochastic Interpolants", ICLR 2023: https://arxiv.org/abs/2209.15571
- Song et al., "Consistency Models", ICML 2023: https://arxiv.org/abs/2303.01469

## World Models And Model-Based Control

World-model work already has mature answers for planning under learned dynamics: PlaNet, Dreamer, DreamerV2/V3, TD-MPC, TD-MPC2, EfficientZero, and MuZero-like search systems. These systems do not merely sample many futures and pick the largest proxy score; they learn latent dynamics, value functions, policies, and planning procedures jointly or iteratively.

This project is therefore not positioned as a competitive model-based RL algorithm. It is a diagnostic artifact for a narrower pattern: whole-trajectory generative sampling plus post-hoc value reranking.

Key sources:

- Ha and Schmidhuber, "World Models": https://arxiv.org/abs/1803.10122
- Hafner et al., "Learning Latent Dynamics for Planning from Pixels": https://arxiv.org/abs/1811.04551
- Hafner et al., "Dream to Control": https://arxiv.org/abs/1912.01603
- Hafner et al., "Mastering Diverse Domains through World Models": https://arxiv.org/abs/2301.04104
- Hansen et al., "TD-MPC2": https://arxiv.org/abs/2310.16828

## Diffusion And Trajectory Generators

Diffuser, Decision Diffuser, trajectory diffusion, and flow-matching policies already cast decision making as conditional trajectory generation. These papers are the closest neighbors. They establish that generative samplers can be used for planning and policy generation, but they also make clear that sampling, guidance, conditioning, and reranking are design choices with different failure surfaces.

The final angle should avoid saying "generative planning fails under Best-of-N" as if that were new. The more precise question is whether proxy reranking exposes an off-manifold candidate-selection issue that is easy to miss when evaluating only average sample quality.

Key sources:

- Janner et al., "Diffuser": https://arxiv.org/abs/2205.09991
- Ajay et al., "Decision Diffuser": https://arxiv.org/abs/2211.15657
- Zhu et al., "Diffusion Models for Reinforcement Learning: A Survey": https://arxiv.org/abs/2311.01223
- PolyGRAD, "World Models via Policy-Guided Trajectory Diffusion": https://openreview.net/forum?id=9CcgO0LhKG

## Best-of-N, Reward Hacking, And Goodhart Effects

Best-of-N sampling is already known to overoptimize imperfect reward models. Gao, Schulman, and Hilton study this directly for language models. Later work on regularized Best-of-N and minimum Bayes risk reranking also makes a repair-style contribution risky unless the repair is clearly scoped.

The paper should therefore cite Best-of-N overoptimization as the parent phenomenon and ask what the world-model setting adds: continuous trajectories, physical validity, manifold distance, diversity collapse, and true-vs-proxy return gaps.

Key sources:

- Gao et al., "Scaling Laws for Reward Model Overoptimization": https://arxiv.org/abs/2210.10760
- Jinnai et al., "Regularized Best-of-N Sampling to Mitigate Reward Hacking": https://arxiv.org/abs/2404.01054

## Resulting Gap

The literature leaves room for a careful, narrow paper:

1. Train or use a flow-matching style conditional trajectory generator.
2. Apply Best-of-N inference with an imperfect learned value or proxy.
3. Measure whether higher proxy score comes from better trajectories or from off-manifold score exploitation.
4. Test a repair that penalizes candidates far from the training trajectory manifold.

The repo implements exactly this small claim. It should be treated as a falsifiable pilot, not as a finished ICLR-strength empirical study.
