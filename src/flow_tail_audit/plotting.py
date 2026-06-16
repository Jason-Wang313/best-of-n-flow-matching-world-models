from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


def save_diagnostic_plots(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    specs = [
        ("return_vs_n", "true_return", "True return"),
        ("gap_vs_n", "pred_true_gap", "Predicted minus true return"),
        ("ood_vs_n", "ood", "Feature-space OOD distance"),
        ("diversity_vs_n", "selected_mode_entropy", "Selected mode entropy"),
    ]
    for name, column, ylabel in specs:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=140)
        for method, subset in frame.groupby("method"):
            subset = subset.sort_values("n")
            ax.plot(subset["n"], subset[column], marker="o", linewidth=1.8, label=method)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Candidate budget N")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        paths[name] = path
    return paths
