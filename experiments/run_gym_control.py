from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flow_tail_audit.gym_control import (
    CONTROL_BENCHMARKS,
    action_features,
    context_vector,
    discretize_actions,
    expert_action_sequence,
    feature_ood,
    feature_stats,
    optimistic_proxy_score,
    rollout_utility,
    sample_action_flow,
    start_state,
    train_action_flow,
)
from flow_tail_audit.samplers import calibrated_tail_indices, proxy_tail_indices


N_GRID = (1, 2, 4, 8, 16, 32)
MAX_HORIZON = max(spec.horizon for spec in CONTROL_BENCHMARKS.values())
FEATURE_PENALTY_BY_BENCHMARK = {
    "CartPole-v1": 6.0,
    "MountainCar-v0": 4.0,
    "Acrobot-v1": 1.0,
}
RESIDUAL_PENALTY = 2.5


def _pad_actions(actions: np.ndarray, horizon: int = MAX_HORIZON) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    if arr.shape[0] >= horizon:
        return arr[:horizon]
    pad_value = arr[-1:] if arr.shape[0] else np.zeros((1, 1), dtype=np.float32)
    pad = np.repeat(pad_value, horizon - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0).astype(np.float32)


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "std": 0.0, "n": 0.0}
    rng = np.random.default_rng(12345)
    boot = np.asarray([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(1000)], dtype=float)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "mean": float(np.mean(arr)),
        "lo": float(lo),
        "hi": float(hi),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": float(len(arr)),
    }


def _make_training(seed: int, train_per_task: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actions: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    task_ids: list[int] = []
    states: list[np.ndarray] = []
    for task_id, spec in enumerate(CONTROL_BENCHMARKS.values()):
        for i in range(int(train_per_task)):
            init = start_state(spec.env_id, seed * 1000 + task_id * 100 + i)
            act = _pad_actions(expert_action_sequence(spec, init, seed=seed * 2000 + task_id * 100 + i))
            actions.append(act)
            contexts.append(context_vector(task_id, init))
            task_ids.append(task_id)
            states.append(init)
    return (
        np.asarray(actions, dtype=np.float32),
        np.asarray(contexts, dtype=np.float32),
        np.asarray(task_ids, dtype=int),
        np.asarray(states, dtype=object),
    )


def _select_rows(
    benchmark: str,
    seed: int,
    context_id: int,
    pred: np.ndarray,
    true: np.ndarray,
    ood: np.ndarray,
    residual: np.ndarray,
    candidates: np.ndarray,
    n_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in n_values:
        pred_n = pred[:n][None, :]
        ood_n = ood[:n][None, :]
        residual_n = residual[:n][None, :]
        residual_z = (residual_n - residual_n.mean(axis=1, keepdims=True)) / (
            residual_n.std(axis=1, keepdims=True) + 1e-6
        )
        methods = {
            "first_candidate": 0,
            "proxy_tail": int(proxy_tail_indices(pred_n)[0]),
            "feature_calibrated_tail": int(
                calibrated_tail_indices(pred_n, ood_n, FEATURE_PENALTY_BY_BENCHMARK[benchmark])[0]
            ),
            "flow_residual_tail": int(calibrated_tail_indices(pred_n, residual_z, RESIDUAL_PENALTY)[0]),
            "oracle_tail": int(np.argmax(true[:n])),
        }
        for method, idx in methods.items():
            rows.append(
                {
                    "benchmark": benchmark,
                    "seed": int(seed),
                    "context_id": int(context_id),
                    "n": int(n),
                    "method": method,
                    "true_return": float(true[idx]),
                    "proxy_score": float(pred[idx]),
                    "pred_true_gap": float(pred[idx] - true[idx]),
                    "ood": float(ood[idx]),
                    "flow_residual": float(residual[idx]),
                    "selected_action_mean": float(np.mean(candidates[idx])),
                    "selected_action_energy": float(action_features(candidates[idx])[2]),
                }
            )
    return rows


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    df = pd.DataFrame(rows)
    means = df.groupby(["benchmark", "method", "n"], as_index=False)["true_return"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=150)
    colors = {
        "first_candidate": "#777777",
        "proxy_tail": "#b23b3b",
        "feature_calibrated_tail": "#1a8f5a",
        "flow_residual_tail": "#2f6fbb",
        "oracle_tail": "#111111",
    }
    for ax, benchmark in zip(axes, CONTROL_BENCHMARKS):
        sub = means[means["benchmark"] == benchmark]
        for method in colors:
            g = sub[sub["method"] == method].sort_values("n")
            ax.plot(g["n"], g["true_return"], marker="o", linewidth=1.8, color=colors[method], label=method)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_GRID)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(benchmark)
        ax.set_xlabel("candidate budget N")
        ax.grid(True, alpha=0.28)
    axes[0].set_ylabel("Executed rollout utility")
    axes[-1].legend(frameon=False, fontsize=6.8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run_gym_control(
    preset: str,
    seeds: list[int],
    output: str | Path,
    paper_figures: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_per_task = 28 if preset == "smoke" else 56
    eval_contexts = 3 if preset == "smoke" else 6
    train_steps = 70 if preset == "smoke" else 120
    max_candidates = 16 if preset == "smoke" else 32
    n_values = tuple(n for n in N_GRID if n <= max_candidates)

    rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    configs: dict[str, Any] = {}
    for seed in seeds:
        train_actions, train_contexts, train_task_ids, _ = _make_training(seed, train_per_task)
        task_feature_stats = {
            int(task_id): feature_stats(train_actions[train_task_ids == task_id])
            for task_id in np.unique(train_task_ids)
        }
        model = train_action_flow(train_actions, train_contexts, seed=seed + 701, steps=train_steps)
        configs[str(seed)] = {
            "train_per_task": train_per_task,
            "eval_contexts": eval_contexts,
            "train_steps": train_steps,
            "feature_penalty_by_benchmark": FEATURE_PENALTY_BY_BENCHMARK,
            "residual_penalty": RESIDUAL_PENALTY,
        }

        for task_id, spec in enumerate(CONTROL_BENCHMARKS.values()):
            eval_states = [start_state(spec.env_id, seed * 3000 + task_id * 100 + i) for i in range(eval_contexts)]
            eval_context = np.asarray([context_vector(task_id, s) for s in eval_states], dtype=np.float32)
            coarse = sample_action_flow(
                model,
                eval_context,
                horizon=MAX_HORIZON,
                n_candidates=max_candidates,
                ode_steps=8,
                seed=seed * 4000 + task_id,
                noise_scale=2.0,
            )
            fine = sample_action_flow(
                model,
                eval_context,
                horizon=MAX_HORIZON,
                n_candidates=max_candidates,
                ode_steps=16,
                seed=seed * 4000 + task_id,
                noise_scale=2.0,
            )
            residual = np.sqrt(np.mean((coarse[:, :, : spec.horizon, :] - fine[:, :, : spec.horizon, :]) ** 2, axis=(2, 3)))
            for context_id, init in enumerate(eval_states):
                candidates = fine[context_id, :, : spec.horizon, :]
                padded_candidates = np.asarray([_pad_actions(c) for c in candidates], dtype=np.float32)
                pred = np.asarray([optimistic_proxy_score(spec, c, init) for c in candidates], dtype=np.float32)
                feat_mean, feat_scale = task_feature_stats[task_id]
                ood = feature_ood(padded_candidates, feat_mean, feat_scale)
                true_values = []
                for candidate in candidates:
                    actions = discretize_actions(spec, candidate.reshape(-1))
                    true, _ = rollout_utility(spec, init, actions)
                    true_values.append(true)
                true_arr = np.asarray(true_values, dtype=np.float32)
                rows.extend(
                    _select_rows(
                        spec.name,
                        seed,
                        context_id,
                        pred,
                        true_arr,
                        ood,
                        residual[context_id],
                        candidates,
                        n_values,
                    )
                )

            df_seed = pd.DataFrame([r for r in rows if r["seed"] == seed and r["benchmark"] == spec.name])
            max_n = max(n_values)
            pivot = df_seed[df_seed["n"] == max_n].pivot_table(
                index="context_id", columns="method", values=["true_return", "pred_true_gap", "ood", "flow_residual"]
            )
            first = pivot["true_return"]["first_candidate"]
            proxy = pivot["true_return"]["proxy_tail"]
            calibrated = pivot["true_return"]["feature_calibrated_tail"]
            residual_sel = pivot["true_return"]["flow_residual_tail"]
            oracle = pivot["true_return"]["oracle_tail"]
            effect_rows.append(
                {
                    "benchmark": spec.name,
                    "seed": int(seed),
                    "proxy_minus_first_true_return": float((proxy - first).mean()),
                    "feature_minus_proxy_true_return": float((calibrated - proxy).mean()),
                    "flow_residual_minus_proxy_true_return": float((residual_sel - proxy).mean()),
                    "oracle_minus_proxy_true_return": float((oracle - proxy).mean()),
                    "feature_minus_proxy_gap": float(
                        (pivot["pred_true_gap"]["feature_calibrated_tail"] - pivot["pred_true_gap"]["proxy_tail"]).mean()
                    ),
                    "feature_minus_proxy_ood": float(
                        (pivot["ood"]["feature_calibrated_tail"] - pivot["ood"]["proxy_tail"]).mean()
                    ),
                    "flow_residual_minus_proxy_residual": float(
                        (pivot["flow_residual"]["flow_residual_tail"] - pivot["flow_residual"]["proxy_tail"]).mean()
                    ),
                }
            )

    curves = pd.DataFrame(rows)
    effects = pd.DataFrame(effect_rows)
    curves.to_csv(output / "metrics.csv", index=False)
    effects.to_csv(output / "effects.csv", index=False)

    benchmark_diagnostics: dict[str, Any] = {}
    proxy_harm_count = 0
    feature_repair_count = 0
    residual_control_count = 0
    for benchmark, sub in effects.groupby("benchmark"):
        proxy_ci = _ci(sub["proxy_minus_first_true_return"].astype(float).tolist())
        repair_ci = _ci(sub["feature_minus_proxy_true_return"].astype(float).tolist())
        oracle_ci = _ci(sub["oracle_minus_proxy_true_return"].astype(float).tolist())
        residual_ci = _ci(sub["flow_residual_minus_proxy_residual"].astype(float).tolist())
        gap_ci = _ci(sub["feature_minus_proxy_gap"].astype(float).tolist())
        ood_ci = _ci(sub["feature_minus_proxy_ood"].astype(float).tolist())
        proxy_harm = proxy_ci["hi"] < -0.25
        feature_repair = repair_ci["lo"] > 0.25 and gap_ci["hi"] < 0.0 and ood_ci["hi"] < 0.0
        residual_control = residual_ci["hi"] < 0.0
        proxy_harm_count += int(proxy_harm)
        feature_repair_count += int(feature_repair)
        residual_control_count += int(residual_control)
        benchmark_diagnostics[str(benchmark)] = {
            "proxy_minus_first_true_return_ci": proxy_ci,
            "feature_minus_proxy_true_return_ci": repair_ci,
            "oracle_minus_proxy_true_return_ci": oracle_ci,
            "feature_minus_proxy_gap_ci": gap_ci,
            "feature_minus_proxy_ood_ci": ood_ci,
            "flow_residual_minus_proxy_residual_ci": residual_ci,
            "proxy_harm": bool(proxy_harm),
            "feature_repair": bool(feature_repair),
            "residual_control_reduces_step_sensitivity": bool(residual_control),
        }

    figure_path = output / "figures" / "gym_control_benchmarks.png"
    _plot(rows, figure_path)
    if paper_figures is not None:
        paper_dir = Path(paper_figures)
        paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure_path, paper_dir / "gym_control_benchmarks.png")

    payload = {
        "experiment": "gym_control_rectified_flow_action_benchmarks",
        "preset": preset,
        "seeds": seeds,
        "benchmarks": list(CONTROL_BENCHMARKS),
        "n_values": list(n_values),
        "max_candidates": int(max_candidates),
        "configs": configs,
        "benchmark_diagnostics": benchmark_diagnostics,
        "key_result": {
            "proxy_harm_benchmark_count": int(proxy_harm_count),
            "feature_repair_benchmark_count": int(feature_repair_count),
            "residual_control_benchmark_count": int(residual_control_count),
            "curve_rows": int(len(rows)),
            "effect_rows": int(len(effect_rows)),
        },
    }
    with (output / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gymnasium rectified-flow action benchmark audits.")
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seeds", default="60,61,62,63,64")
    parser.add_argument("--output", default="results/gym_control")
    parser.add_argument("--paper-figures", default="paper/figures")
    args = parser.parse_args()
    payload = run_gym_control(args.preset, parse_seeds(args.seeds), args.output, args.paper_figures)
    print(json.dumps(payload["key_result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
