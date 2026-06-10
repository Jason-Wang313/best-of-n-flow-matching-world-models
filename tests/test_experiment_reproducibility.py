from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_synthetic import run_experiment
from flow_matching_bon.config import ExperimentConfig


def test_tiny_experiment_is_reproducible(tmp_path: Path):
    config = ExperimentConfig.preset("test", seed=123)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    summary_a = run_experiment(config, out_a)
    summary_b = run_experiment(config, out_b)
    metrics_a = pd.read_csv(out_a / "metrics.csv")
    metrics_b = pd.read_csv(out_b / "metrics.csv")

    assert summary_a.keys() == summary_b.keys()
    for key in summary_a:
        assert np.isclose(summary_a[key], summary_b[key])
    numeric_cols = metrics_a.select_dtypes(include="number").columns
    assert np.allclose(metrics_a[numeric_cols], metrics_b[numeric_cols])
