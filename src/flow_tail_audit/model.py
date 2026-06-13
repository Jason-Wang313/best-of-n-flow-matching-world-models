from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from flow_tail_audit.config import ExperimentConfig
from flow_tail_audit.data import flatten_trajectories, unflatten_trajectories


class ConditionalVectorField(nn.Module):
    def __init__(self, trajectory_dim: int, context_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trajectory_dim + context_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, trajectory_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x_t, t, context], dim=1))


@dataclass(frozen=True)
class TrainingResult:
    losses: list[float]


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def train_rectified_flow(
    trajectories: np.ndarray,
    contexts: np.ndarray,
    config: ExperimentConfig,
) -> tuple[ConditionalVectorField, TrainingResult]:
    """Train a simulation-free rectified-flow vector field."""

    set_torch_seed(config.seed)
    device = torch.device(config.device)
    y = torch.tensor(flatten_trajectories(trajectories), dtype=torch.float32, device=device)
    c = torch.tensor(contexts, dtype=torch.float32, device=device)
    model = ConditionalVectorField(y.shape[1], c.shape[1], config.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(config.seed + 17)
    losses: list[float] = []

    for _ in range(config.train_steps):
        idx = torch.randint(0, y.shape[0], (config.batch_size,), generator=generator, device=device)
        x1 = y[idx]
        ctx = c[idx]
        x0 = torch.randn(x1.shape, generator=generator, device=device)
        t = torch.rand((x1.shape[0], 1), generator=generator, device=device)
        x_t = (1.0 - t) * x0 + t * x1
        target = x1 - x0
        pred = model(x_t, t, ctx)
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))

    return model, TrainingResult(losses=losses)


@torch.no_grad()
def sample_rectified_flow(
    model: ConditionalVectorField,
    contexts: np.ndarray,
    n_candidates: int,
    horizon: int,
    ode_steps: int,
    seed: int,
    noise_scale: float = 1.0,
    device: str = "cpu",
) -> np.ndarray:
    """Sample candidate trajectories by Euler-integrating the learned flow."""

    set_torch_seed(seed)
    model.eval()
    ctx = torch.tensor(contexts, dtype=torch.float32, device=device)
    batch = contexts.shape[0]
    repeated_ctx = ctx.repeat_interleave(n_candidates, dim=0)
    dim = horizon * 2
    generator = torch.Generator(device=device).manual_seed(seed + 1009)
    x = noise_scale * torch.randn((batch * n_candidates, dim), generator=generator, device=device)
    dt = 1.0 / float(ode_steps)
    for i in range(ode_steps):
        t_value = torch.full((x.shape[0], 1), i / float(ode_steps), dtype=torch.float32, device=device)
        x = x + dt * model(x, t_value, repeated_ctx)
    arr = x.detach().cpu().numpy().astype(np.float32)
    arr = unflatten_trajectories(arr, horizon)
    arr = arr.reshape(batch, n_candidates, horizon, 2)
    arr[:, :, 0, :] = np.array([0.0, 0.0], dtype=np.float32)
    arr[:, :, -1, 0] = 1.0
    arr[:, :, -1, 1] = contexts[:, None, 0]
    return arr
