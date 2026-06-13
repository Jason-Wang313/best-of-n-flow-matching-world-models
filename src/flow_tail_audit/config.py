from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the synthetic rectified-flow tail-audit experiment."""

    seed: int = 0
    horizon: int = 24
    train_size: int = 512
    eval_contexts: int = 64
    train_steps: int = 250
    batch_size: int = 128
    hidden_dim: int = 96
    lr: float = 2e-3
    ode_steps: int = 16
    max_candidates: int = 32
    candidate_counts: tuple[int, ...] = field(default_factory=lambda: (1, 2, 4, 8, 16, 32))
    penalty_weight: float = 0.35
    proxy_noise: float = 0.08
    device: str = "cpu"

    @classmethod
    def preset(cls, name: str, seed: int = 0) -> "ExperimentConfig":
        if name == "smoke":
            return cls(
                seed=seed,
                horizon=18,
                train_size=256,
                eval_contexts=32,
                train_steps=100,
                batch_size=96,
                hidden_dim=72,
                ode_steps=10,
                max_candidates=16,
                candidate_counts=(1, 2, 4, 8, 16),
            )
        if name == "full":
            return cls(seed=seed)
        if name == "test":
            return cls(
                seed=seed,
                horizon=12,
                train_size=64,
                eval_contexts=8,
                train_steps=12,
                batch_size=32,
                hidden_dim=32,
                ode_steps=4,
                max_candidates=4,
                candidate_counts=(1, 4),
            )
        raise ValueError(f"Unknown preset: {name}")
