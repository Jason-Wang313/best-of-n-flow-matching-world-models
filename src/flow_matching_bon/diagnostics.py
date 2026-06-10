from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flow_matching_bon.metrics import ProxyValueModel, mean_pairwise_distance, mode_entropy, true_return
from flow_matching_bon.samplers import best_of_n_indices, gather_candidates, uncertainty_penalized_indices


@dataclass(frozen=True)
class DiagnosticResult:
    frame: pd.DataFrame

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for method in sorted(self.frame["method"].unique()):
            subset = self.frame[self.frame["method"] == method]
            last = subset.sort_values("n").iloc[-1]
            out[f"{method}_true_return_at_max_n"] = float(last["true_return"])
            out[f"{method}_pred_true_gap_at_max_n"] = float(last["pred_true_gap"])
            out[f"{method}_ood_at_max_n"] = float(last["ood"])
        return out


def evaluate_best_of_n(
    candidates: np.ndarray,
    contexts: np.ndarray,
    proxy: ProxyValueModel,
    candidate_counts: tuple[int, ...],
    penalty_weight: float,
) -> DiagnosticResult:
    """Evaluate naive and repaired Best-of-N selection."""

    rows: list[dict[str, float | int | str]] = []
    n_contexts, max_candidates, horizon, _ = candidates.shape
    flat_contexts = np.repeat(contexts, max_candidates, axis=0)
    flat_candidates = candidates.reshape(n_contexts * max_candidates, horizon, 2)
    pred_all = proxy.score(flat_candidates, flat_contexts).reshape(n_contexts, max_candidates)
    true_all = true_return(flat_candidates, flat_contexts).reshape(n_contexts, max_candidates)
    ood_all = proxy.uncertainty(flat_candidates, flat_contexts).reshape(n_contexts, max_candidates)

    for n in candidate_counts:
        if n > max_candidates:
            raise ValueError(f"candidate count {n} exceeds sampled count {max_candidates}")
        pred = pred_all[:, :n]
        true = true_all[:, :n]
        ood = ood_all[:, :n]
        cand = candidates[:, :n]

        candidate_diversity = float(np.mean([mean_pairwise_distance(cand[i]) for i in range(n_contexts)]))
        candidate_true_mean = float(np.mean(true))
        candidate_pred_mean = float(np.mean(pred))

        for method, idx in {
            "random": np.zeros(n_contexts, dtype=int),
            "bon_proxy": best_of_n_indices(pred),
            "bon_uncertainty": uncertainty_penalized_indices(pred, ood, penalty_weight),
        }.items():
            selected = gather_candidates(cand, idx)
            selected_pred = pred[np.arange(n_contexts), idx]
            selected_true = true[np.arange(n_contexts), idx]
            selected_ood = ood[np.arange(n_contexts), idx]
            rows.append(
                {
                    "n": int(n),
                    "method": method,
                    "true_return": float(np.mean(selected_true)),
                    "predicted_return": float(np.mean(selected_pred)),
                    "pred_true_gap": float(np.mean(selected_pred - selected_true)),
                    "ood": float(np.mean(selected_ood)),
                    "selection_bias": float(np.mean(selected_pred) - candidate_pred_mean),
                    "candidate_true_mean": candidate_true_mean,
                    "candidate_pred_mean": candidate_pred_mean,
                    "candidate_diversity": candidate_diversity,
                    "selected_mode_entropy": mode_entropy(selected),
                }
            )

    return DiagnosticResult(frame=pd.DataFrame(rows))
