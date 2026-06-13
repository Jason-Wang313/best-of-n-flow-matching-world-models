from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flow_tail_audit.config import ExperimentConfig
from flow_tail_audit.data import generate_contexts, make_dataset, make_rng
from flow_tail_audit.metrics import fit_proxy_value, true_return
from flow_tail_audit.model import sample_rectified_flow, train_rectified_flow
from flow_tail_audit.samplers import calibrated_tail_indices, proxy_tail_indices


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _select_rows(
    seed: int,
    steps: int,
    pred: np.ndarray,
    ret: np.ndarray,
    ood: np.ndarray,
    residual: np.ndarray,
    penalty_weight: float,
) -> list[dict[str, float | int | str]]:
    n_contexts = pred.shape[0]
    if float(np.std(residual)) > 0.0:
        residual_z = (residual - float(np.mean(residual))) / (float(np.std(residual)) + 1e-6)
    else:
        residual_z = np.zeros_like(residual)
    methods = {
        "first_candidate": np.zeros(n_contexts, dtype=int),
        "proxy_tail": proxy_tail_indices(pred),
        "feature_calibrated_tail": calibrated_tail_indices(pred, ood, penalty_weight),
        "flow_residual_tail": calibrated_tail_indices(pred, residual_z, 8.0),
        "hybrid_feature_flow_tail": calibrated_tail_indices(pred, ood + 20.0 * residual, penalty_weight),
    }
    rows: list[dict[str, float | int | str]] = []
    for method, idx in methods.items():
        selected_pred = pred[np.arange(n_contexts), idx]
        selected_true = ret[np.arange(n_contexts), idx]
        selected_ood = ood[np.arange(n_contexts), idx]
        selected_residual = residual[np.arange(n_contexts), idx]
        rows.append(
            {
                "seed": seed,
                "ode_steps": steps,
                "method": method,
                "true_return": float(np.mean(selected_true)),
                "pred_true_gap": float(np.mean(selected_pred - selected_true)),
                "ood": float(np.mean(selected_ood)),
                "flow_step_residual": float(np.mean(selected_residual)),
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    aggregate: dict[str, float] = {"seeds": float(df["seed"].nunique())}
    for steps, step_df in df.groupby("ode_steps"):
        pivot = step_df.pivot(index="seed", columns="method", values="true_return")
        aggregate[f"steps_{steps}_proxy_minus_first_true_return_mean"] = float(
            (pivot["proxy_tail"] - pivot["first_candidate"]).mean()
        )
        aggregate[f"steps_{steps}_feature_minus_proxy_true_return_mean"] = float(
            (pivot["feature_calibrated_tail"] - pivot["proxy_tail"]).mean()
        )
        aggregate[f"steps_{steps}_flow_residual_minus_proxy_true_return_mean"] = float(
            (pivot["flow_residual_tail"] - pivot["proxy_tail"]).mean()
        )
        residual_pivot = step_df.pivot(index="seed", columns="method", values="flow_step_residual")
        aggregate[f"steps_{steps}_flow_residual_minus_proxy_residual_mean"] = float(
            (residual_pivot["flow_residual_tail"] - residual_pivot["proxy_tail"]).mean()
        )
    proxy_harm = []
    feature_gain = []
    flow_residual_change = []
    for steps, step_df in df.groupby("ode_steps"):
        pivot = step_df.pivot(index="seed", columns="method", values="true_return")
        proxy_harm.extend((pivot["proxy_tail"] - pivot["first_candidate"]).tolist())
        feature_gain.extend((pivot["feature_calibrated_tail"] - pivot["proxy_tail"]).tolist())
        residual_pivot = step_df.pivot(index="seed", columns="method", values="flow_step_residual")
        flow_residual_change.extend((residual_pivot["flow_residual_tail"] - residual_pivot["proxy_tail"]).tolist())
    aggregate["all_steps_proxy_minus_first_true_return_mean"] = float(np.mean(proxy_harm))
    aggregate["all_steps_proxy_minus_first_true_return_max"] = float(np.max(proxy_harm))
    aggregate["all_steps_feature_minus_proxy_true_return_mean"] = float(np.mean(feature_gain))
    aggregate["all_steps_feature_minus_proxy_true_return_min"] = float(np.min(feature_gain))
    aggregate["all_steps_flow_residual_minus_proxy_residual_mean"] = float(np.mean(flow_residual_change))
    aggregate["all_steps_flow_residual_minus_proxy_residual_max"] = float(np.max(flow_residual_change))
    return aggregate


def save_plot(rows: list[dict[str, float | int | str]], output: Path) -> None:
    df = pd.DataFrame(rows)
    methods = ["first_candidate", "proxy_tail", "feature_calibrated_tail", "flow_residual_tail"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), dpi=150)
    for method in methods:
        sub = df[df["method"] == method]
        grouped = sub.groupby("ode_steps", as_index=False).mean(numeric_only=True)
        axes[0].plot(grouped["ode_steps"], grouped["true_return"], marker="o", linewidth=1.8, label=method)
        axes[1].plot(grouped["ode_steps"], grouped["flow_step_residual"], marker="o", linewidth=1.8, label=method)
    axes[0].set_xlabel("Euler steps")
    axes[0].set_ylabel("Selected true return")
    axes[0].set_title("Return is not rescued by more steps")
    axes[1].set_xlabel("Euler steps")
    axes[1].set_ylabel("Fine-vs-coarse trajectory residual")
    axes[1].set_title("Residual-only control")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run_step_sweep(
    preset: str,
    seeds: list[int],
    output: str | Path,
    paper_figures: str | Path | None = None,
) -> dict[str, float]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    configs: dict[str, dict[str, object]] = {}
    for seed in seeds:
        config = ExperimentConfig.preset(preset, seed=seed)
        if preset == "smoke":
            config = replace(config, train_steps=80, eval_contexts=32, max_candidates=16, ode_steps=16)
        train = make_dataset(config.train_size, config.horizon, config.seed)
        model, _ = train_rectified_flow(train.trajectories, train.contexts, config)
        proxy = fit_proxy_value(train.trajectories, train.contexts, config.seed, label_noise=config.proxy_noise)
        contexts = generate_contexts(config.eval_contexts, make_rng(config.seed + 909))
        fine = sample_rectified_flow(
            model,
            contexts,
            n_candidates=config.max_candidates,
            horizon=config.horizon,
            ode_steps=16,
            seed=config.seed + 123,
            device=config.device,
        )
        for steps in (4, 8, 16):
            candidates = sample_rectified_flow(
                model,
                contexts,
                n_candidates=config.max_candidates,
                horizon=config.horizon,
                ode_steps=steps,
                seed=config.seed + 123,
                device=config.device,
            )
            residual = np.sqrt(np.mean((candidates - fine) ** 2, axis=(2, 3)))
            n_contexts, n_candidates, horizon, _ = candidates.shape
            flat_contexts = np.repeat(contexts, n_candidates, axis=0)
            flat_candidates = candidates.reshape(n_contexts * n_candidates, horizon, 2)
            pred = proxy.score(flat_candidates, flat_contexts).reshape(n_contexts, n_candidates)
            ret = true_return(flat_candidates, flat_contexts).reshape(n_contexts, n_candidates)
            ood = proxy.uncertainty(flat_candidates, flat_contexts).reshape(n_contexts, n_candidates)
            rows.extend(_select_rows(seed, steps, pred, ret, ood, residual, config.penalty_weight))
        configs[str(seed)] = asdict(config)

    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"preset": preset, "seeds": seeds, "configs": configs}, f, indent=2, sort_keys=True)
    aggregate = summarize(rows)
    with (output / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    plot_path = output / "figures" / "flow_step_sweep.png"
    save_plot(rows, plot_path)
    if paper_figures is not None:
        paper_dir = Path(paper_figures)
        paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot_path, paper_dir / plot_path.name)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rectified-flow Euler-step consistency stress audit.")
    parser.add_argument("--preset", choices=["smoke", "test"], default="smoke")
    parser.add_argument("--seeds", default="50,51,52,53,54")
    parser.add_argument("--output", default="results/step_sweep")
    parser.add_argument("--paper-figures", default="paper/figures")
    args = parser.parse_args()
    aggregate = run_step_sweep(args.preset, parse_seeds(args.seeds), args.output, args.paper_figures)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
