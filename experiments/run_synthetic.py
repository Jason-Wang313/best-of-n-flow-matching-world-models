from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flow_tail_audit.config import ExperimentConfig
from flow_tail_audit.data import generate_contexts, make_dataset, make_rng
from flow_tail_audit.diagnostics import evaluate_tail_selection
from flow_tail_audit.metrics import fit_proxy_value
from flow_tail_audit.model import sample_rectified_flow, train_rectified_flow
from flow_tail_audit.plotting import save_diagnostic_plots


def run_experiment(
    config: ExperimentConfig,
    output: str | Path,
    paper_figures: str | Path | None = None,
) -> dict[str, float]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    train = make_dataset(config.train_size, config.horizon, config.seed)
    model, train_result = train_rectified_flow(train.trajectories, train.contexts, config)
    proxy = fit_proxy_value(train.trajectories, train.contexts, config.seed, label_noise=config.proxy_noise)

    rng = make_rng(config.seed + 909)
    eval_contexts = generate_contexts(config.eval_contexts, rng)
    candidates = sample_rectified_flow(
        model,
        eval_contexts,
        n_candidates=config.max_candidates,
        horizon=config.horizon,
        ode_steps=config.ode_steps,
        seed=config.seed + 123,
        device=config.device,
    )

    diagnostics = evaluate_tail_selection(
        candidates,
        eval_contexts,
        proxy,
        candidate_counts=config.candidate_counts,
        penalty_weight=config.penalty_weight,
    )

    metrics_path = output / "metrics.csv"
    diagnostics.frame.to_csv(metrics_path, index=False)
    pd.DataFrame({"step": np.arange(len(train_result.losses)), "loss": train_result.losses}).to_csv(
        output / "train_loss.csv", index=False
    )
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    figures = save_diagnostic_plots(diagnostics.frame, output / "figures")
    if paper_figures is not None:
        paper_dir = Path(paper_figures)
        paper_dir.mkdir(parents=True, exist_ok=True)
        for path in figures.values():
            shutil.copy2(path, paper_dir / path.name)

    summary = diagnostics.summary()
    summary["final_train_loss"] = float(train_result.losses[-1])
    summary["metrics_rows"] = float(len(diagnostics.frame))
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic rectified-flow tail-audit experiment.")
    parser.add_argument("--preset", choices=["smoke", "full", "test"], default="smoke")
    parser.add_argument("--output", default="results/smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-contexts", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--candidate-counts", default=None, help="Comma-separated candidate counts, e.g. 1,4,16")
    parser.add_argument("--paper-figures", default="paper/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.preset(args.preset, seed=args.seed)
    updates = {}
    if args.train_steps is not None:
        updates["train_steps"] = args.train_steps
    if args.eval_contexts is not None:
        updates["eval_contexts"] = args.eval_contexts
    if args.max_candidates is not None:
        updates["max_candidates"] = args.max_candidates
    if args.candidate_counts is not None:
        updates["candidate_counts"] = tuple(int(x.strip()) for x in args.candidate_counts.split(",") if x.strip())
    if updates:
        config = replace(config, **updates)
    summary = run_experiment(config, args.output, paper_figures=args.paper_figures)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
