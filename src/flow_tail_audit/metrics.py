from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = (
    "endpoint_score",
    "clearance",
    "collision_free",
    "negative_smoothness",
    "negative_path_length",
    "mode_height",
    "negative_backtrack",
)

PROXY_FEATURES = (0, 5)


def _as_contexts(contexts: np.ndarray, n: int) -> np.ndarray:
    if contexts.ndim == 1:
        return np.repeat(contexts[None, :], n, axis=0)
    return contexts


def trajectory_features(trajectories: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    """Return hand-built trajectory features for diagnostics and proxy fitting."""

    original_shape = trajectories.shape
    flat = trajectories.reshape(-1, original_shape[-2], 2)
    ctx = _as_contexts(contexts, flat.shape[0]).reshape(flat.shape[0], -1)
    goal = np.stack([np.ones(flat.shape[0]), ctx[:, 0]], axis=1)
    endpoint_error = np.linalg.norm(flat[:, -1, :] - goal, axis=1)
    endpoint_score = -endpoint_error

    obstacle = np.array([0.5, 0.0], dtype=np.float32)
    dist = np.linalg.norm(flat - obstacle[None, None, :], axis=2)
    clearance = np.min(dist - ctx[:, None, 1], axis=1)
    collision_depth = np.maximum(ctx[:, None, 1] - dist, 0.0)
    collision_free = -np.mean(collision_depth**2, axis=1)

    diffs = np.diff(flat, axis=1)
    path_length = np.sum(np.linalg.norm(diffs, axis=2), axis=1)
    second = np.diff(flat, n=2, axis=1)
    smoothness = np.mean(np.sum(second**2, axis=2), axis=1)
    mode_height = np.max(np.abs(flat[:, :, 1]), axis=1)
    backtrack = np.mean(np.maximum(-diffs[:, :, 0], 0.0), axis=1)

    features = np.stack(
        [
            endpoint_score,
            clearance,
            collision_free,
            -smoothness,
            -path_length,
            mode_height,
            -backtrack,
        ],
        axis=1,
    )
    return features.astype(np.float32)


def true_return(trajectories: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    """Evaluate the true synthetic task return."""

    f = trajectory_features(trajectories, contexts)
    excessive_height = np.maximum(f[:, 5] - 0.72, 0.0)
    reward = (
        4.0 * f[:, 0]
        + 8.0 * np.minimum(f[:, 1], 0.0)
        + 70.0 * f[:, 2]
        + 8.0 * f[:, 3]
        + 0.45 * f[:, 4]
        + 1.8 * f[:, 6]
        - 9.0 * excessive_height**2
    )
    return reward.astype(np.float32)


@dataclass(frozen=True)
class ProxyValueModel:
    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    full_feature_mean: np.ndarray
    full_feature_scale: np.ndarray

    def score(self, trajectories: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        features = trajectory_features(trajectories, contexts)
        x = (features[:, PROXY_FEATURES] - self.feature_mean) / self.feature_scale
        return (x @ self.weights + self.bias).astype(np.float32)

    def uncertainty(self, trajectories: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        features = trajectory_features(trajectories, contexts)
        z = (features - self.full_feature_mean) / self.full_feature_scale
        return np.sqrt(np.mean(z**2, axis=1)).astype(np.float32)


def fit_proxy_value(
    trajectories: np.ndarray,
    contexts: np.ndarray,
    seed: int,
    label_noise: float = 0.08,
    ridge: float = 1e-3,
) -> ProxyValueModel:
    """Fit an intentionally imperfect proxy value model."""

    rng = np.random.default_rng(seed + 303)
    features = trajectory_features(trajectories, contexts)
    y = true_return(trajectories, contexts)
    y = y + rng.normal(0.0, label_noise * max(float(np.std(y)), 1e-6), size=y.shape)
    x = features[:, PROXY_FEATURES]
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6
    xz = (x - mean) / scale
    design = np.concatenate([xz, np.ones((xz.shape[0], 1), dtype=np.float32)], axis=1)
    reg = ridge * np.eye(design.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    params = np.linalg.solve(design.T @ design + reg, design.T @ y)
    weights = params[:-1].astype(np.float32)
    if len(weights) > 1:
        weights[1] = abs(float(weights[1])) + 2.5
    min_scale = np.array([0.05, 0.04, 0.002, 0.01, 0.05, 0.04, 0.005], dtype=np.float32)
    full_mean = features.mean(axis=0)
    full_scale = np.maximum(features.std(axis=0), min_scale) + 1e-6
    return ProxyValueModel(
        weights=weights,
        bias=float(params[-1]),
        feature_mean=mean.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        full_feature_mean=full_mean.astype(np.float32),
        full_feature_scale=full_scale.astype(np.float32),
    )


def mean_pairwise_distance(trajectories: np.ndarray) -> float:
    flat = trajectories.reshape(trajectories.shape[0], -1)
    if flat.shape[0] < 2:
        return 0.0
    diffs = flat[:, None, :] - flat[None, :, :]
    dist = np.sqrt(np.mean(diffs**2, axis=2))
    tri = np.triu_indices(flat.shape[0], k=1)
    return float(np.mean(dist[tri]))


def mode_entropy(trajectories: np.ndarray) -> float:
    mid = trajectories.shape[-2] // 2
    signs = trajectories[:, mid, 1] >= 0.0
    p = float(np.mean(signs))
    if p <= 1e-8 or p >= 1.0 - 1e-8:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))
