# Best-of-N Flow-Matching World Models

This repository is a first-pass research artifact for the question:

> What goes wrong when a flow-matching or rectified-flow world model is sampled many times and reranked by an imperfect value model?

The code trains a small conditional rectified-flow trajectory generator on a synthetic point-mass world, samples `N` candidate futures, and compares naive Best-of-N reranking against an uncertainty-penalized repair. The paper and docs are written in an intentionally skeptical style: the repo prioritizes runnable evidence and clear failure modes over claiming a breakthrough.

## Quick Start

```powershell
python -m pip install -e .[dev]
python -m pytest
python experiments/run_synthetic.py --preset smoke --output results/smoke
python scripts/build_paper.py
```

The paper build writes:

- `paper/final/iclr_submission.pdf`
- `C:\Users\wangz\Downloads\iclr_submission_flow_matching_world_models.pdf`

## Repository Layout

- `src/flow_matching_bon/`: synthetic task, rectified-flow model, samplers, metrics, diagnostics, and plotting.
- `experiments/`: runnable synthetic experiments.
- `tests/`: sampler, metric, diagnostic, and reproducibility coverage.
- `docs/`: literature map, hostile novelty review, related-work matrix, novelty decision, and final audit.
- `paper/`: anonymous ICLR-style LaTeX source and generated final PDF.
- `scripts/`: smoke experiment, literature matrix, and paper build helpers.

## Research Claim

The current evidence supports a modest claim:

Naive Best-of-N selection can amplify proxy-score errors in conditional rectified-flow trajectory generators by selecting candidates that are high-scoring under the proxy but far from the training trajectory manifold. A simple feature-space uncertainty penalty reduces that exploitation in the synthetic setting, but it is not yet a general solution.

## Minimal Experiment

The synthetic world is a 2D point mass moving from `(0, 0)` to a context-dependent goal while avoiding an obstacle. Expert trajectories use two modes around the obstacle. A conditional rectified-flow model learns to generate whole trajectories. A learned linear proxy value model is trained on limited features, so it can overrate short or off-manifold trajectories. Diagnostics measure:

- true return vs. predicted return,
- selected-candidate exploitation gap,
- candidate and selected trajectory diversity,
- feature-space out-of-distribution distance,
- selection bias as `N` increases.

## Reproducibility

Experiments are seeded and CPU-friendly. The smoke preset should finish quickly on a standard laptop. The larger preset is still intentionally small enough for iteration:

```powershell
python experiments/run_synthetic.py --preset full --output results/full
```

## Limitations

This is not a claim that all flow-matching world models fail under Best-of-N, nor that uncertainty penalties are sufficient in high-dimensional robotic or video domains. The artifact is meant to make the failure mode concrete, testable, and easy to falsify.
