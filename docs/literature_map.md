# Literature Map

This map identifies what would make the project non-novel or overclaimed.

## Flow Matching And Rectified Flow

Flow Matching trains continuous normalizing flows by regressing vector fields along probability paths rather than simulating a diffusion process during training. Rectified Flow specializes this view around straight transports between base and data distributions and can often be sampled with few ODE steps.

The novelty pressure is that many sampler and guidance failures are not unique to rectified flow. Stochastic interpolants, probability-flow ODEs, consistency models, and diffusion guidance all blur the boundary between "flow-specific" and "generic generative model" claims. The paper therefore focuses on the selected upper tail of flow-generated trajectories rather than claiming a unique rectified-flow pathology.

Key sources:

- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023: https://arxiv.org/abs/2210.02747
- Liu et al., "Flow Straight and Fast", ICLR 2023: https://arxiv.org/abs/2209.03003
- Albergo and Vanden-Eijnden, "Building Normalizing Flows with Stochastic Interpolants", ICLR 2023: https://arxiv.org/abs/2209.15571
- Song et al., "Consistency Models", ICML 2023: https://arxiv.org/abs/2303.01469

## World Models And Model-Based Control

World-model work already has mature planning systems: PlaNet, Dreamer, DreamerV2/V3, TD-MPC, TD-MPC2, EfficientZero, and MuZero-like search systems. These methods learn latent dynamics, value functions, policies, and planning procedures jointly or iteratively. This repo cannot claim algorithmic superiority over model-based RL.

The defensible position is narrower: when a conditional trajectory generator and a learned proxy are used as a post-hoc selection interface, the selected future should be audited for physical validity, manifold distance, and realized return.

Key sources:

- Ha and Schmidhuber, "World Models": https://arxiv.org/abs/1803.10122
- Hafner et al., "Learning Latent Dynamics for Planning from Pixels": https://arxiv.org/abs/1811.04551
- Hafner et al., "Dream to Control": https://arxiv.org/abs/1912.01603
- Hafner et al., "Mastering Diverse Domains through World Models": https://arxiv.org/abs/2301.04104
- Hansen et al., "TD-MPC2": https://arxiv.org/abs/2310.16828

## Diffusion And Trajectory Generators

Diffuser, Decision Diffuser, trajectory diffusion, and flow-matching policies already cast behavior as conditional trajectory generation. They establish that generative samplers can support planning and policy generation, but sampling, guidance, conditioning, and reranking create different failure surfaces.

The paper should therefore avoid saying "generative planning fails" as if that were new. The precise gap is whether a selected high-proxy trajectory from a rectified-flow candidate pool is still executable and close to the training trajectory manifold.

Key sources:

- Janner et al., "Diffuser": https://arxiv.org/abs/2205.09991
- Ajay et al., "Decision Diffuser": https://arxiv.org/abs/2211.15657
- Zhu et al., "Diffusion Models for Reinforcement Learning: A Survey": https://arxiv.org/abs/2311.01223
- PolyGRAD, "World Models via Policy-Guided Trajectory Diffusion": https://openreview.net/forum?id=9CcgO0LhKG

## Reward-Model Overoptimization And Reranking

Reward-model overoptimization is already known: a learned reward or preference model can be exploited by selecting or optimizing strongly against it. Later work on regularized reranking and minimum Bayes risk decoding also makes repair-style novelty risky unless the repair is clearly scoped.

The paper should cite overoptimization as the parent phenomenon and ask what the trajectory world-model setting adds: continuous paths, physical-validity features, manifold distance, diversity collapse, and held-out true returns.

Key sources:

- Gao et al., "Scaling Laws for Reward Model Overoptimization": https://arxiv.org/abs/2210.10760
- Jinnai et al., "Regularized Best-of-N Sampling to Mitigate Reward Hacking": https://arxiv.org/abs/2404.01054

## Resulting Gap

The literature leaves room for a careful, narrow paper:

1. Train or use a flow-matching style conditional trajectory generator.
2. Apply upper-tail selection with an imperfect learned value or proxy.
3. Measure whether higher proxy score comes from better trajectories or from off-manifold score exploitation.
4. Test whether a training-manifold calibration penalty reduces the selected-tail failure.
5. Keep claims scoped to the synthetic audit until standard control benchmarks and diffusion baselines are added.
