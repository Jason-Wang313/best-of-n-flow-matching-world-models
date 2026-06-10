import numpy as np

from flow_matching_bon.data import generate_contexts, generate_expert_trajectories, make_rng
from flow_matching_bon.diagnostics import evaluate_best_of_n
from flow_matching_bon.metrics import fit_proxy_value


def test_evaluate_best_of_n_returns_all_methods():
    rng = make_rng(7)
    contexts = generate_contexts(8, rng)
    batch = generate_expert_trajectories(contexts, horizon=10, rng=rng)
    proxy = fit_proxy_value(batch.trajectories, batch.contexts, seed=7)
    candidates = np.repeat(batch.trajectories[:, None, :, :], 4, axis=1)
    candidates[:, 1:, :, 1] += rng.normal(0.0, 0.05, size=candidates[:, 1:, :, 1].shape)

    result = evaluate_best_of_n(candidates, contexts, proxy, candidate_counts=(1, 4), penalty_weight=0.2)
    frame = result.frame

    assert set(frame["method"]) == {"random", "bon_proxy", "bon_uncertainty"}
    assert set(frame["n"]) == {1, 4}
    assert frame["true_return"].notna().all()
    assert "bon_proxy_true_return_at_max_n" in result.summary()
