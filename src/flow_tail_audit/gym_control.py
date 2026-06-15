from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch

from flow_tail_audit.model import ConditionalVectorField, set_torch_seed


@dataclass(frozen=True)
class ControlSpec:
    name: str
    env_id: str
    horizon: int
    actions: tuple[int, ...]
    action_scale: tuple[float, ...]


CONTROL_BENCHMARKS = {
    "CartPole-v1": ControlSpec("CartPole-v1", "CartPole-v1", 80, (0, 1), (-1.0, 1.0)),
    "MountainCar-v0": ControlSpec("MountainCar-v0", "MountainCar-v0", 70, (0, 1, 2), (-1.0, 0.0, 1.0)),
    "Acrobot-v1": ControlSpec("Acrobot-v1", "Acrobot-v1", 70, (0, 1, 2), (-1.0, 0.0, 1.0)),
}


def start_state(env_id: str, seed: int) -> np.ndarray:
    env = gym.make(env_id)
    try:
        env.reset(seed=int(seed))
        return np.asarray(env.unwrapped.state, dtype=np.float32).copy()
    finally:
        env.close()


def context_vector(task_index: int, state: np.ndarray) -> np.ndarray:
    one_hot = np.zeros(3, dtype=np.float32)
    one_hot[int(task_index)] = 1.0
    padded = np.zeros(6, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    padded[: min(len(state), len(padded))] = state[: min(len(state), len(padded))]
    return np.concatenate([one_hot, padded]).astype(np.float32)


def _set_state(env, state: np.ndarray) -> None:
    env.unwrapped.state = np.asarray(state, dtype=np.float64).copy()


def discretize_actions(spec: ControlSpec, continuous: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(continuous), dtype=float)
    if len(spec.actions) == 2:
        return np.where(arr >= 0.0, spec.actions[1], spec.actions[0]).astype(int)
    out = np.full(arr.shape, spec.actions[1], dtype=int)
    out[arr < -0.33] = spec.actions[0]
    out[arr > 0.33] = spec.actions[2]
    return out


def valid_action_sequence(spec: ControlSpec, actions: Iterable[int]) -> bool:
    allowed = set(int(a) for a in spec.actions)
    return all(int(a) in allowed for a in actions)


def _cartpole_margin(state: np.ndarray) -> float:
    x, _, theta, _ = np.asarray(state, dtype=float)
    x_margin = max(0.0, 1.0 - abs(float(x)) / 2.4)
    theta_margin = max(0.0, 1.0 - abs(float(theta)) / 0.2095)
    return 0.5 * x_margin + 0.5 * theta_margin


def _mountaincar_progress(states: list[np.ndarray], final_state: np.ndarray) -> float:
    max_position = max([float(s[0]) for s in states] + [float(final_state[0])])
    final_position = float(final_state[0])
    final_velocity = float(final_state[1])
    return 35.0 * (max_position + 0.5) + 15.0 * max(0.0, final_position + 0.5) + 8.0 * final_velocity


def _acrobot_tip_height(state: np.ndarray) -> float:
    theta1, theta2 = float(state[0]), float(state[1])
    return -np.cos(theta1) - np.cos(theta1 + theta2)


def _acrobot_progress(states: list[np.ndarray], final_state: np.ndarray) -> float:
    heights = [_acrobot_tip_height(s) for s in states] or [_acrobot_tip_height(final_state)]
    return 14.0 * max(heights) + 8.0 * _acrobot_tip_height(final_state)


def rollout_utility(spec: ControlSpec, initial_state: np.ndarray, actions: Iterable[int]) -> tuple[float, dict[str, float]]:
    env = gym.make(spec.env_id)
    try:
        env.reset(seed=0)
        _set_state(env, initial_state)
        total_reward = 0.0
        final_state = np.asarray(initial_state, dtype=float).copy()
        states: list[np.ndarray] = []
        terminated_at = spec.horizon
        for t, action in enumerate(actions):
            _, reward, terminated, truncated, _ = env.step(int(action))
            final_state = np.asarray(env.unwrapped.state, dtype=float).copy()
            states.append(final_state)
            total_reward += (0.99**t) * float(reward)
            if terminated or truncated:
                terminated_at = t + 1
                break
    finally:
        env.close()

    if spec.name == "CartPole-v1":
        utility = total_reward + 8.0 * _cartpole_margin(final_state)
        success = float(terminated_at >= spec.horizon)
    elif spec.name == "MountainCar-v0":
        utility = total_reward + _mountaincar_progress(states, final_state)
        success = float(float(final_state[0]) >= 0.5)
    elif spec.name == "Acrobot-v1":
        utility = total_reward + _acrobot_progress(states, final_state)
        success = float(_acrobot_tip_height(final_state) > 1.0)
    else:
        raise ValueError(spec.name)
    return float(utility), {
        "discounted_env_return": float(total_reward),
        "terminated_at": float(terminated_at),
        "success": success,
        "final_state_norm": float(np.linalg.norm(final_state)),
    }


def expert_action_sequence(spec: ControlSpec, initial_state: np.ndarray, seed: int, noise: float = 0.08) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    env = gym.make(spec.env_id)
    scale = {int(a): float(v) for a, v in zip(spec.actions, spec.action_scale)}
    values: list[float] = []
    try:
        env.reset(seed=0)
        _set_state(env, initial_state)
        for t in range(spec.horizon):
            state = np.asarray(env.unwrapped.state, dtype=float)
            if spec.name == "CartPole-v1":
                score = state[2] + 0.40 * state[3] + 0.04 * state[0] + 0.02 * state[1]
                action = 1 if score >= 0.0 else 0
            elif spec.name == "MountainCar-v0":
                action = 2 if state[1] >= -0.002 else 0
                if t % 17 in {0, 1}:
                    action = 0
            elif spec.name == "Acrobot-v1":
                score = state[2] + state[3] + 0.5 * np.sin(state[0] + state[1])
                action = 2 if score >= 0.0 else 0
                if abs(score) < 0.08:
                    action = 1
            else:
                raise ValueError(spec.name)
            if rng.random() < noise:
                action = int(rng.choice(spec.actions))
            values.append(scale[int(action)])
            _, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                for _ in range(t + 1, spec.horizon):
                    values.append(scale[int(action)])
                break
    finally:
        env.close()
    return np.asarray(values[: spec.horizon], dtype=np.float32).reshape(spec.horizon, 1)


def action_features(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=float).reshape(-1)
    diffs = np.diff(arr)
    switches = np.abs(diffs) > 1e-6
    switch_rate = float(np.mean(switches)) if switches.size else 0.0
    runs: list[int] = []
    run_len = 1
    for changed in switches:
        if changed:
            runs.append(run_len)
            run_len = 1
        else:
            run_len += 1
    runs.append(run_len)
    max_run = float(max(runs) / max(1, len(arr)))
    roughness = float(np.mean(diffs**2)) if diffs.size else 0.0
    segments = np.array_split(arr, 4)
    segment_means = [float(np.mean(s)) if len(s) else 0.0 for s in segments]
    lag1 = float(np.mean(arr[:-1] * arr[1:])) if len(arr) > 1 else 0.0
    early_late = float(segment_means[0] - segment_means[-1])
    return np.asarray(
        [
            float(np.mean(arr)),
            float(abs(np.mean(arr))),
            float(np.mean(np.abs(arr))),
            switch_rate,
            max_run,
            roughness,
            float(np.mean(arr[: max(1, len(arr) // 4)])),
            *segment_means,
            lag1,
            early_late,
        ],
        dtype=np.float32,
    )


def optimistic_proxy_score(spec: ControlSpec, actions: np.ndarray, initial_state: np.ndarray) -> float:
    f = action_features(actions)
    mean, bias, energy, switch_rate, max_run, roughness, early = [float(x) for x in f[:7]]
    if spec.name == "CartPole-v1":
        theta = abs(float(initial_state[2]))
        return 35.0 + 48.0 * bias + 18.0 * energy + 30.0 * max_run - 16.0 * switch_rate - 10.0 * theta
    if spec.name == "MountainCar-v0":
        pos = float(initial_state[0])
        return -45.0 + 44.0 * max(0.0, -mean) + 18.0 * bias + 16.0 * max_run + 18.0 * (pos + 0.5)
    if spec.name == "Acrobot-v1":
        return -52.0 + 42.0 * bias + 22.0 * energy + 28.0 * max_run - 10.0 * switch_rate - 4.0 * roughness + 2.0 * abs(early)
    raise ValueError(spec.name)


def feature_stats(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feats = np.asarray([action_features(a) for a in actions], dtype=np.float32)
    return feats.mean(axis=0), np.maximum(feats.std(axis=0), 0.04)


def feature_ood(actions: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    feats = np.asarray([action_features(a) for a in actions], dtype=np.float32)
    z = (feats - mean[None, :]) / scale[None, :]
    return np.sqrt(np.mean(z**2, axis=1)).astype(np.float32)


def train_action_flow(
    actions: np.ndarray,
    contexts: np.ndarray,
    seed: int,
    steps: int = 140,
    hidden_dim: int = 72,
    batch_size: int = 96,
    lr: float = 2e-3,
) -> ConditionalVectorField:
    set_torch_seed(seed)
    y = torch.tensor(actions.reshape(actions.shape[0], -1), dtype=torch.float32)
    c = torch.tensor(contexts, dtype=torch.float32)
    model = ConditionalVectorField(y.shape[1], c.shape[1], hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 17)
    for _ in range(int(steps)):
        idx = torch.randint(0, y.shape[0], (min(batch_size, y.shape[0]),), generator=gen)
        x1 = y[idx]
        ctx = c[idx]
        x0 = torch.randn(x1.shape, generator=gen)
        t = torch.rand((x1.shape[0], 1), generator=gen)
        x_t = (1.0 - t) * x0 + t * x1
        pred = model(x_t, t, ctx)
        loss = torch.mean((pred - (x1 - x0)) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


@torch.no_grad()
def sample_action_flow(
    model: ConditionalVectorField,
    contexts: np.ndarray,
    horizon: int,
    n_candidates: int,
    ode_steps: int,
    seed: int,
    noise_scale: float = 1.15,
) -> np.ndarray:
    set_torch_seed(seed)
    model.eval()
    ctx = torch.tensor(contexts, dtype=torch.float32)
    repeated = ctx.repeat_interleave(int(n_candidates), dim=0)
    gen = torch.Generator().manual_seed(seed + 1009)
    x = noise_scale * torch.randn((contexts.shape[0] * int(n_candidates), int(horizon)), generator=gen)
    dt = 1.0 / float(ode_steps)
    for i in range(int(ode_steps)):
        t = torch.full((x.shape[0], 1), i / float(ode_steps), dtype=torch.float32)
        x = x + dt * model(x, t, repeated)
    arr = np.clip(x.detach().cpu().numpy().astype(np.float32), -1.5, 1.5)
    return arr.reshape(contexts.shape[0], int(n_candidates), int(horizon), 1)
