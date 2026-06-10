import numpy as np

from flow_matching_bon.data import generate_contexts, generate_expert_trajectories, make_rng
from flow_matching_bon.metrics import fit_proxy_value, trajectory_features, true_return


def test_true_return_penalizes_obstacle_collision():
    horizon = 16
    context = np.array([[0.0, 0.2, 0.0]], dtype=np.float32)
    tau = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    colliding = np.stack([tau, np.zeros_like(tau)], axis=1)[None, :, :]
    safe = np.stack([tau, 0.45 * np.sin(np.pi * tau)], axis=1)[None, :, :]

    assert true_return(safe, context)[0] > true_return(colliding, context)[0]


def test_proxy_value_shapes_and_uncertainty():
    rng = make_rng(42)
    contexts = generate_contexts(32, rng)
    batch = generate_expert_trajectories(contexts, horizon=12, rng=rng)
    proxy = fit_proxy_value(batch.trajectories, batch.contexts, seed=42)

    scores = proxy.score(batch.trajectories, batch.contexts)
    uncertainty = proxy.uncertainty(batch.trajectories, batch.contexts)
    features = trajectory_features(batch.trajectories, batch.contexts)

    assert scores.shape == (32,)
    assert uncertainty.shape == (32,)
    assert features.shape == (32, 7)
    assert np.isfinite(scores).all()
    assert np.isfinite(uncertainty).all()
