# Rectified-Flow Tail-Value Audits

This repository studies a narrow failure mode in rectified-flow trajectory world models:

> When many generated futures are available, does the upper tail chosen by a learned proxy reflect better executable behavior, or merely proxy-seeking trajectory artifacts?

The code trains a small conditional rectified-flow trajectory generator on a synthetic two-mode point-mass world, evaluates large candidate pools under an intentionally incomplete value proxy, and measures how the selected upper-tail futures differ from the training trajectory manifold. The project is framed as an audit harness, not a new planning algorithm.

## Quick Start

```powershell
python -m pip install -e .[dev]
python -m pytest
python experiments/run_synthetic.py --preset smoke --output results/smoke
python experiments/run_multiseed.py --preset smoke --seeds 0,1,2,3,4 --output results/multiseed
python experiments/run_step_sweep.py --preset smoke --seeds 50,51,52,53,54 --output results/step_sweep
python experiments/run_gym_control.py --preset full --seeds 70,71,72,73,74 --output results/gym_control
python scripts/run_claim_audit.py
python scripts/build_paper.py --desktop-copy "C:\Users\wangz\OneDrive\Desktop\best of n flow matching world models-v4.pdf"
```

The paper build writes:

- `paper/final/best of n flow matching world models-v4.pdf`
- optional visible Desktop copy when `--desktop-copy` is supplied.

## Repository Layout

- `src/flow_tail_audit/`: synthetic task, rectified-flow model, tail selectors, metrics, diagnostics, and plotting.
- `experiments/`: single-seed and multi-seed synthetic audits plus Gymnasium classic-control stress tests.
- `tests/`: sampler, metric, diagnostic, and reproducibility coverage.
- `docs/`: prior-work pressure tests, novelty decision, related-work matrix, and final audit.
- `paper/`: anonymous LaTeX source, generated figures, and final PDF.
- `scripts/`: claim audit, paper build, and literature-matrix helpers.

## Research Claim

The current evidence supports a scoped claim:

In a controlled rectified-flow trajectory generator, proxy-tail selection can increase apparent value while worsening realized return and moving selected trajectories farther from the training manifold. A feature-space calibration penalty reduces the gap and OOD distance across a five-seed synthetic audit and shows strict repair on held-out CartPole. MountainCar and Acrobot are reported as harder stress cases where calibration reduces OOD/gap but only weakly improves executed utility.

## Minimal Experiment

The synthetic world is a 2D point mass moving from `(0, 0)` to a context-dependent goal while avoiding an obstacle. Expert trajectories use two modes around the obstacle. A conditional rectified-flow model learns to generate whole trajectories. A learned linear proxy value model is trained on limited features, so it can overrate short or off-manifold trajectories. Diagnostics measure:

- true return vs. predicted return,
- selected-tail exploitation gap,
- candidate and selected trajectory diversity,
- feature-space out-of-distribution distance,
- rectified-flow Euler-step consistency,
- selection bias as candidate budget `N` increases.

## Standard Benchmark Tier

`experiments/run_gym_control.py` evaluates rectified-flow action-sequence sampling on Gymnasium `CartPole-v1`, `MountainCar-v0`, and `Acrobot-v1`. The full frozen benchmark uses seeds `70,71,72,73,74`, six contexts per task/seed, candidate budgets through `N=32`, and five selectors: first candidate, proxy tail, feature-calibrated tail, flow-residual control, and an oracle-in-pool upper bound.

## Reproducibility

Experiments are seeded and CPU-friendly. The five-seed smoke audit is intentionally small enough to rerun during review. The larger synthetic preset is available for a slower stress pass:

```powershell
python experiments/run_synthetic.py --preset full --output results/full
```

## Limitations

This is not evidence that all flow-matching world models fail under large candidate budgets, nor that feature-space penalties are sufficient in high-dimensional robotics or video domains. The artifact makes one tail-selection failure mode concrete, measurable, and easy to falsify, with Gymnasium stress tests used to expose both repair success and repair boundaries.
