from __future__ import annotations

import numpy as np


def proxy_tail_indices(scores: np.ndarray) -> np.ndarray:
    """Select the upper-tail proxy candidate for each context."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [contexts, candidates]")
    return np.argmax(scores, axis=1)


def calibrated_tail_indices(scores: np.ndarray, uncertainty: np.ndarray, weight: float) -> np.ndarray:
    """Select upper-tail candidates after subtracting a calibration penalty."""

    if scores.shape != uncertainty.shape:
        raise ValueError("scores and uncertainty must have the same shape")
    return np.argmax(scores - weight * uncertainty, axis=1)


def gather_candidates(candidates: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if candidates.ndim < 3:
        raise ValueError("candidates must have shape [contexts, candidates, ...]")
    rows = np.arange(candidates.shape[0])
    return candidates[rows, indices]
