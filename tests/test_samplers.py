import numpy as np

from flow_matching_bon.samplers import best_of_n_indices, gather_candidates, uncertainty_penalized_indices


def test_best_of_n_indices_select_rowwise_maximum():
    scores = np.array([[1.0, 3.0, 2.0], [-1.0, -2.0, 0.0]])
    np.testing.assert_array_equal(best_of_n_indices(scores), np.array([1, 2]))


def test_uncertainty_penalty_can_change_selection():
    scores = np.array([[1.0, 2.0]])
    uncertainty = np.array([[0.0, 5.0]])
    np.testing.assert_array_equal(uncertainty_penalized_indices(scores, uncertainty, weight=0.5), np.array([0]))


def test_gather_candidates_uses_context_rows():
    candidates = np.array([[[1], [2], [3]], [[4], [5], [6]]])
    gathered = gather_candidates(candidates, np.array([2, 0]))
    np.testing.assert_array_equal(gathered, np.array([[3], [4]]))
