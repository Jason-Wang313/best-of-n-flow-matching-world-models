from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticBatch:
    contexts: np.ndarray
    trajectories: np.ndarray
    modes: np.ndarray


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_contexts(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample goal/obstacle/wind contexts for the point-mass world."""

    goal_y = rng.uniform(-0.35, 0.35, size=n)
    radius = rng.uniform(0.13, 0.22, size=n)
    wind = rng.uniform(-0.20, 0.20, size=n)
    return np.stack([goal_y, radius, wind], axis=1).astype(np.float32)


def generate_expert_trajectories(
    contexts: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    noise_scale: float = 0.018,
) -> SyntheticBatch:
    """Generate two-mode expert trajectories around an obstacle."""

    n = contexts.shape[0]
    tau = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    modes = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    trajectories = np.zeros((n, horizon, 2), dtype=np.float32)

    for i, (goal_y, radius, wind) in enumerate(contexts):
        mode = modes[i]
        x = tau + 0.035 * wind * np.sin(2.0 * np.pi * tau)
        clearance_bump = (0.38 + 0.75 * radius) * np.sin(np.pi * tau)
        y = goal_y * tau + mode * clearance_bump
        y += 0.035 * wind * np.sin(3.0 * np.pi * tau)
        noise = rng.normal(0.0, noise_scale, size=(horizon, 2)).astype(np.float32)
        noise[0] = 0.0
        noise[-1] = 0.0
        trajectories[i, :, 0] = x
        trajectories[i, :, 1] = y
        trajectories[i] += smooth_noise(noise)

    trajectories[:, 0, :] = np.array([0.0, 0.0], dtype=np.float32)
    trajectories[:, -1, 0] = 1.0
    trajectories[:, -1, 1] = contexts[:, 0]
    return SyntheticBatch(contexts=contexts.astype(np.float32), trajectories=trajectories, modes=modes)


def smooth_noise(noise: np.ndarray) -> np.ndarray:
    if len(noise) < 5:
        return noise
    kernel = np.array([0.15, 0.25, 0.20, 0.25, 0.15], dtype=np.float32)
    padded = np.pad(noise, ((2, 2), (0, 0)), mode="edge")
    out = np.zeros_like(noise)
    for i in range(len(noise)):
        out[i] = (padded[i : i + 5] * kernel[:, None]).sum(axis=0)
    return out


def make_dataset(n: int, horizon: int, seed: int) -> SyntheticBatch:
    rng = make_rng(seed)
    contexts = generate_contexts(n, rng)
    return generate_expert_trajectories(contexts, horizon, rng)


def flatten_trajectories(trajectories: np.ndarray) -> np.ndarray:
    return trajectories.reshape(trajectories.shape[0], -1).astype(np.float32)


def unflatten_trajectories(flat: np.ndarray, horizon: int) -> np.ndarray:
    return flat.reshape(flat.shape[0], horizon, 2).astype(np.float32)
