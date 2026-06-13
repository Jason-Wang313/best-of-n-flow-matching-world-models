from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_synthetic import run_experiment
from experiments.run_step_sweep import summarize
from flow_tail_audit.config import ExperimentConfig


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


def test_step_sweep_summary_tracks_persistent_proxy_harm():
    rows = []
    for seed in [0, 1]:
        for steps in [4, 8, 16]:
            rows.extend(
                [
                    {
                        "seed": seed,
                        "ode_steps": steps,
                        "method": "first_candidate",
                        "true_return": -10.0,
                        "flow_step_residual": 0.1,
                    },
                    {
                        "seed": seed,
                        "ode_steps": steps,
                        "method": "proxy_tail",
                        "true_return": -30.0,
                        "flow_step_residual": 0.2,
                    },
                    {
                        "seed": seed,
                        "ode_steps": steps,
                        "method": "feature_calibrated_tail",
                        "true_return": -5.0,
                        "flow_step_residual": 0.12,
                    },
                    {
                        "seed": seed,
                        "ode_steps": steps,
                        "method": "flow_residual_tail",
                        "true_return": -28.0,
                        "flow_step_residual": 0.05,
                    },
                ]
            )
    aggregate = summarize(rows)
    assert aggregate["all_steps_proxy_minus_first_true_return_max"] == -20.0
    assert aggregate["all_steps_feature_minus_proxy_true_return_min"] == 25.0
    assert aggregate["all_steps_flow_residual_minus_proxy_residual_mean"] < 0.0
