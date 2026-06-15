from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "multiseed" / "aggregate.json"
STEP_SWEEP = ROOT / "results" / "step_sweep" / "aggregate.json"
GYM_CONTROL = ROOT / "results" / "gym_control" / "aggregate.json"


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    if not AGGREGATE.exists():
        raise SystemExit(f"missing multiseed aggregate: {AGGREGATE}")
    with AGGREGATE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    step_data = None
    if STEP_SWEEP.exists():
        with STEP_SWEEP.open("r", encoding="utf-8") as f:
            step_data = json.load(f)
    gym_data = None
    if GYM_CONTROL.exists():
        with GYM_CONTROL.open("r", encoding="utf-8") as f:
            gym_data = json.load(f)

    failures: list[str] = []
    for method in ["first_candidate", "proxy_tail", "calibrated_tail"]:
        for metric in ["true_return", "pred_true_gap", "ood"]:
            key = f"{method}_{metric}_mean"
            require(key in data, f"missing aggregate metric: {key}", failures)

    if not failures:
        require(data["seeds"] >= 5, "multiseed audit must cover at least five seeds", failures)
        require(
            data["calibrated_tail_true_return_mean"] > data["proxy_tail_true_return_mean"],
            "calibrated tail selection must improve mean true return over proxy tail selection",
            failures,
        )
        require(
            data["calibrated_tail_pred_true_gap_mean"] < data["proxy_tail_pred_true_gap_mean"],
            "calibrated tail selection must reduce mean predicted-vs-true gap",
            failures,
        )
        require(
            data["calibrated_tail_ood_mean"] < data["proxy_tail_ood_mean"],
            "calibrated tail selection must reduce mean feature-space OOD distance",
            failures,
        )
        require(step_data is not None, f"missing flow step-sweep aggregate: {STEP_SWEEP}", failures)
        if step_data is not None:
            require(step_data["seeds"] >= 5, "flow step-sweep audit must cover at least five seeds", failures)
            require(
                step_data["all_steps_proxy_minus_first_true_return_max"] < -5.0,
                "proxy-tail harm must persist across Euler step counts",
                failures,
            )
            require(
                step_data["all_steps_feature_minus_proxy_true_return_min"] > 10.0,
                "feature-calibrated tail must recover true return across Euler step counts",
                failures,
            )
            require(
                step_data["all_steps_flow_residual_minus_proxy_residual_mean"] < 0.0,
                "flow-residual-only control must reduce selected step residual on average",
                failures,
            )
        require(gym_data is not None, f"missing Gymnasium control aggregate: {GYM_CONTROL}", failures)
        if gym_data is not None:
            key = gym_data["key_result"]
            diagnostics = gym_data["benchmark_diagnostics"]
            benchmarks = gym_data["benchmarks"]
            require(gym_data["preset"] == "full", "Gymnasium benchmark must use the full preset", failures)
            require(len(gym_data["seeds"]) >= 5, "Gymnasium benchmark must cover at least five held-out seeds", failures)
            require(gym_data["max_candidates"] >= 32, "Gymnasium benchmark must audit at least 32 candidates", failures)
            require(key["curve_rows"] >= 2700, "Gymnasium benchmark must include all task/seed/context/method rows", failures)
            require(key["effect_rows"] >= 15, "Gymnasium benchmark must include all task-by-seed effects", failures)
            require(
                key["proxy_harm_benchmark_count"] >= 1,
                "at least one standard-control benchmark must show statistically strict proxy-tail harm",
                failures,
            )
            require(
                key["feature_repair_benchmark_count"] >= 1,
                "at least one standard-control benchmark must show statistically strict feature-calibrated repair",
                failures,
            )
            require(
                key["residual_control_benchmark_count"] == len(benchmarks),
                "flow-residual control must reduce selected step residual on every Gymnasium benchmark",
                failures,
            )
            cartpole = diagnostics["CartPole-v1"]
            require(cartpole["proxy_harm"], "CartPole must retain strict proxy-tail harm", failures)
            require(cartpole["feature_repair"], "CartPole must retain strict feature-calibrated repair", failures)
            for benchmark in benchmarks:
                diag = diagnostics[benchmark]
                require(
                    diag["oracle_minus_proxy_true_return_ci"]["lo"] > 0.0,
                    f"{benchmark} must have a positive oracle-vs-proxy candidate-pool gap",
                    failures,
                )
                require(
                    diag["feature_minus_proxy_true_return_ci"]["mean"] >= 0.0,
                    f"{benchmark} feature-calibrated selector must not reduce mean executed utility",
                    failures,
                )
                require(
                    diag["feature_minus_proxy_ood_ci"]["hi"] <= 0.0,
                    f"{benchmark} feature-calibrated selector must not increase selected OOD distance",
                    failures,
                )
                require(
                    diag["feature_minus_proxy_gap_ci"]["mean"] <= 0.0,
                    f"{benchmark} feature-calibrated selector must not increase mean proxy-realized gap",
                    failures,
                )

    report = {
        "aggregate": str(AGGREGATE),
        "gym_control": str(GYM_CONTROL),
        "status": "fail" if failures else "pass",
        "failures": failures,
        "checked_claims": [
            "five-seed aggregate exists",
            "calibrated tail improves true return over proxy tail",
            "calibrated tail reduces predicted-vs-true gap",
            "calibrated tail reduces feature-space OOD distance",
            "proxy-tail harm persists across Euler step counts",
            "flow-residual-only control reduces step residual but is not promoted as the main repair",
            "held-out Gymnasium benchmark uses full preset and five seeds",
            "CartPole shows strict proxy-tail harm and strict feature-calibrated repair",
            "Gymnasium candidate pools contain oracle headroom above the proxy tail",
            "Gymnasium feature calibration does not increase mean gap or selected OOD distance",
            "Gymnasium flow-residual control reduces step sensitivity on every benchmark",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
