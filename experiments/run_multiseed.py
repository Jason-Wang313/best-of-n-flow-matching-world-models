from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_synthetic import run_experiment
from flow_tail_audit.config import ExperimentConfig


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def summarize_multiseed(output: Path, seeds: list[int]) -> dict[str, float]:
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        frame = pd.read_csv(output / f"seed_{seed}" / "metrics.csv")
        max_n = int(frame["n"].max())
        tail = frame[frame["n"] == max_n].copy()
        tail.insert(0, "seed", seed)
        rows.append(tail)

    summary = pd.concat(rows, ignore_index=True)
    summary_path = output / "summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate_rows: list[dict[str, float | str]] = []
    aggregate: dict[str, float] = {"seeds": float(len(seeds))}
    metrics = ["true_return", "pred_true_gap", "ood", "selection_bias", "selected_mode_entropy"]
    for method, subset in summary.groupby("method"):
        for metric in metrics:
            key = f"{method}_{metric}"
            aggregate[f"{key}_mean"] = float(subset[metric].mean())
            aggregate[f"{key}_std"] = float(subset[metric].std(ddof=1)) if len(subset) > 1 else 0.0
            aggregate_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": aggregate[f"{key}_mean"],
                    "std": aggregate[f"{key}_std"],
                }
            )

    aggregate["calibrated_minus_proxy_true_return_mean"] = (
        aggregate["calibrated_tail_true_return_mean"] - aggregate["proxy_tail_true_return_mean"]
    )
    aggregate["proxy_minus_calibrated_gap_mean"] = (
        aggregate["proxy_tail_pred_true_gap_mean"] - aggregate["calibrated_tail_pred_true_gap_mean"]
    )
    aggregate["proxy_minus_calibrated_ood_mean"] = (
        aggregate["proxy_tail_ood_mean"] - aggregate["calibrated_tail_ood_mean"]
    )

    pd.DataFrame(aggregate_rows).to_csv(output / "aggregate.csv", index=False)
    with (output / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated rectified-flow tail-audit experiments.")
    parser.add_argument("--preset", choices=["smoke", "full", "test"], default="smoke")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output", default="results/multiseed")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-contexts", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--candidate-counts", default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        config = ExperimentConfig.preset(args.preset, seed=seed)
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
        seed_output = output / f"seed_{seed}"
        run_experiment(config, seed_output)
        with (seed_output / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)

    aggregate = summarize_multiseed(output, seeds)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
