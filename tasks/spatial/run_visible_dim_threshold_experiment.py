import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep matplotlib fully headless and in writable cache locations.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .config import TrainConfig
from .run_kd_threshold_experiment import (
    _align_series_to_grid,
    _compute_near_best_threshold,
    _expected_unique_frequency_vectors,
    _extract_metric_series,
    _find_first_crossing,
    _geometric_mean_objective,
    _last_finite_metric,
    _load_metrics_rows,
    _make_axes,
    _normalized_auc,
    _parse_float_list,
    _parse_int_list,
    _prediction_for_k,
    _read_json,
    _run_artifacts_exist,
    _save_figure,
    _trailing_running_average,
    _value_tag,
    _write_json,
)
from .train import run_training


def _build_hidden_dim_grid(
    *,
    basis_complexity: int,
    visible_dims: list[int],
    explicit_hidden_dims: list[int] | None,
    hidden_dim_ratios: list[float],
    max_hidden_dim: int | None,
) -> tuple[dict[int, list[int]], dict[int, dict[str, Any]]]:
    grid: dict[int, list[int]] = {}
    predictions: dict[int, dict[str, Any]] = {}
    for visible_dim in visible_dims:
        prediction = _prediction_for_k(int(basis_complexity), int(visible_dim))
        predictions[int(visible_dim)] = prediction
        if explicit_hidden_dims is not None:
            dims = [int(value) for value in explicit_hidden_dims]
        else:
            dims = []
            threshold_exact = float(prediction["predicted_threshold_exact"])
            for ratio in hidden_dim_ratios:
                value = max(1, int(round(threshold_exact * float(ratio))))
                if max_hidden_dim is not None and value > int(max_hidden_dim):
                    continue
                dims.append(value)
        deduped = sorted(set(int(value) for value in dims if int(value) >= 1))
        if not deduped:
            raise ValueError(
                f"No hidden dimensions remain for visible_dim={visible_dim}. "
                "Increase --max_hidden_dim or provide --hidden_dims explicitly."
            )
        grid[int(visible_dim)] = deduped
    return grid, predictions


def _aggregate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    runs_dict = manifest.get("runs", {})
    if not isinstance(runs_dict, dict):
        raise ValueError("manifest['runs'] must be a dict")

    metrics_cache: dict[str, list[dict[str, Any]]] = {}
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs_dict.values():
        grouped[int(run["visible_dim"])][int(run["hidden_dim"])].append(run)

    predictions_raw = manifest.get("predictions", {})
    predictions: dict[int, dict[str, Any]] = {
        int(key): value for key, value in predictions_raw.items()
    }

    aggregate: dict[str, Any] = {
        "version": 1,
        "created_from_manifest": str(manifest.get("manifest_path", "")),
        "basis_complexity": int(manifest.get("basis_complexity", 2)),
        "success_rate_threshold": float(manifest.get("success_rate_threshold", 0.8)),
        "objective_threshold_tolerance": float(manifest.get("objective_threshold_tolerance", 0.1)),
        "objective_log_epsilon": float(manifest.get("objective_log_epsilon", 1e-6)),
        "cells": {},
        "by_visible_dim": {},
    }

    for visible_dim in sorted(grouped.keys()):
        prediction = predictions[int(visible_dim)]
        by_hidden = grouped[int(visible_dim)]
        dim_entry: dict[str, Any] = {
            **prediction,
            "cells": {},
            "observed_threshold_hidden_dim": None,
            "observed_threshold_ratio": None,
        }
        observed_threshold_hidden_dim: int | None = None

        for hidden_dim in sorted(by_hidden.keys()):
            runs = sorted(by_hidden[int(hidden_dim)], key=lambda item: int(item["seed"]))
            success_series: list[tuple[np.ndarray, np.ndarray]] = []
            objective_series: list[tuple[np.ndarray, np.ndarray]] = []
            objective_running_avg_series: list[tuple[np.ndarray, np.ndarray]] = []
            final_success_values: list[float] = []
            final_objective_values: list[float] = []

            for run in runs:
                metrics_csv = Path(str(run["metrics_csv"]))
                metrics_jsonl = Path(str(run["metrics_jsonl"]))
                cache_key = str(metrics_csv if metrics_csv.exists() else metrics_jsonl)
                if cache_key not in metrics_cache:
                    metrics_cache[cache_key] = _load_metrics_rows(metrics_csv, metrics_jsonl)
                rows = metrics_cache[cache_key]

                success_ep, success_vals = _extract_metric_series(rows, "success_rate")
                objective_ep, objective_vals = _extract_metric_series(rows, "avg_final_objective")
                if success_ep.size > 0:
                    success_series.append((success_ep, success_vals))
                if objective_ep.size > 0:
                    objective_series.append((objective_ep, objective_vals))
                    objective_running_avg_series.append(
                        (objective_ep, _trailing_running_average(objective_vals, window=100))
                    )

                final_success = _last_finite_metric(rows, "success_rate")
                final_objective = _last_finite_metric(rows, "avg_final_objective")
                if final_success is not None:
                    final_success_values.append(final_success)
                if final_objective is not None:
                    final_objective_values.append(final_objective)

            success_grid, success_mean, success_std = _align_series_to_grid(success_series)
            objective_grid, objective_mean, objective_std = _align_series_to_grid(objective_series)
            objective_ra_grid, objective_ra_mean, objective_ra_std = _align_series_to_grid(
                objective_running_avg_series
            )
            objective_ra_final = (
                float(objective_ra_mean[-1]) if objective_ra_mean.size > 0 else None
            )
            objective_ra_auc = _normalized_auc(objective_ra_grid, objective_ra_mean)
            objective_ra_geom = _geometric_mean_objective(
                objective_ra_grid,
                objective_ra_mean,
                epsilon=float(aggregate["objective_log_epsilon"]),
            )
            crossing_episode = _find_first_crossing(
                success_grid,
                success_mean,
                threshold=float(aggregate["success_rate_threshold"]),
            )
            if crossing_episode is not None and observed_threshold_hidden_dim is None:
                observed_threshold_hidden_dim = int(hidden_dim)

            num_frequency_vectors = int(prediction["num_frequency_vectors"])
            expected_unique = _expected_unique_frequency_vectors(int(hidden_dim), num_frequency_vectors)
            coverage_fraction = float(expected_unique / max(1.0, float(num_frequency_vectors)))
            threshold_ratio = float(hidden_dim / float(prediction["predicted_threshold_exact"]))

            cell_key = f"vis{int(visible_dim)}_D{int(hidden_dim)}"
            cell = {
                "visible_dim": int(visible_dim),
                "basis_complexity": int(prediction["basis_complexity"]),
                "hidden_dim": int(hidden_dim),
                "num_seeds": int(len(runs)),
                "threshold_ratio": float(threshold_ratio),
                "expected_unique_frequency_vectors": float(expected_unique),
                "expected_coverage_fraction": float(coverage_fraction),
                "final_success_rate_mean": (
                    float(np.mean(final_success_values)) if final_success_values else None
                ),
                "final_success_rate_std": (
                    float(np.std(final_success_values)) if final_success_values else None
                ),
                "final_objective_mean": (
                    float(np.mean(final_objective_values)) if final_objective_values else None
                ),
                "final_objective_std": (
                    float(np.std(final_objective_values)) if final_objective_values else None
                ),
                "episodes_to_success_rate_threshold": (
                    int(crossing_episode) if crossing_episode is not None else None
                ),
                "aggregate_success_rate_curve": {
                    "episodes": success_grid.tolist(),
                    "mean": success_mean.tolist(),
                    "std": success_std.tolist(),
                },
                "aggregate_objective_curve": {
                    "episodes": objective_grid.tolist(),
                    "mean": objective_mean.tolist(),
                    "std": objective_std.tolist(),
                },
                "aggregate_objective_curve_running_avg_100": {
                    "episodes": objective_ra_grid.tolist(),
                    "mean": objective_ra_mean.tolist(),
                    "std": objective_ra_std.tolist(),
                    "window_episodes": 100,
                },
                "final_objective_running_avg_100": objective_ra_final,
                "objective_auc_running_avg_100": objective_ra_auc,
                "geometric_mean_objective_running_avg_100": objective_ra_geom,
                "run_keys": [str(run["key"]) for run in runs],
            }
            aggregate["cells"][cell_key] = cell
            dim_entry["cells"][str(hidden_dim)] = cell

        final_threshold = _compute_near_best_threshold(
            dim_entry["cells"],
            metric_key="final_objective_running_avg_100",
            tolerance_fraction=float(aggregate["objective_threshold_tolerance"]),
        )
        geom_threshold = _compute_near_best_threshold(
            dim_entry["cells"],
            metric_key="geometric_mean_objective_running_avg_100",
            tolerance_fraction=float(aggregate["objective_threshold_tolerance"]),
        )
        predicted_threshold_exact = float(prediction["predicted_threshold_exact"])
        final_threshold["threshold_d_over_d_star"] = (
            float(final_threshold["threshold_hidden_dim"] / predicted_threshold_exact)
            if final_threshold["threshold_hidden_dim"] is not None
            else None
        )
        geom_threshold["threshold_d_over_d_star"] = (
            float(geom_threshold["threshold_hidden_dim"] / predicted_threshold_exact)
            if geom_threshold["threshold_hidden_dim"] is not None
            else None
        )
        dim_entry["objective_thresholds"] = {
            "final_objective_running_avg_100_near_best": final_threshold,
            "geometric_mean_objective_running_avg_100_near_best": geom_threshold,
        }
        dim_entry["observed_threshold_hidden_dim"] = observed_threshold_hidden_dim
        dim_entry["observed_threshold_ratio"] = (
            float(observed_threshold_hidden_dim / float(prediction["predicted_threshold_exact"]))
            if observed_threshold_hidden_dim is not None
            else None
        )
        aggregate["by_visible_dim"][str(visible_dim)] = dim_entry

    return aggregate


def _plot_curves_by_visible_dim(
    *,
    aggregate: dict[str, Any],
    plots_root: Path,
) -> dict[str, list[str]]:
    curve_plot_paths: dict[str, list[str]] = {
        "success_rate": [],
        "objective": [],
        "objective_running_avg_100": [],
    }
    by_visible_dim = aggregate.get("by_visible_dim", {})

    for key in sorted(by_visible_dim.keys(), key=lambda item: int(item)):
        dim_entry = by_visible_dim[key]
        visible_dim = int(dim_entry["visible_dim"])
        basis_complexity = int(dim_entry["basis_complexity"])
        predicted_threshold = float(dim_entry["predicted_threshold_exact"])
        num_frequency_vectors = int(dim_entry["num_frequency_vectors"])
        cell_items = sorted(
            ((int(hidden_dim), cell) for hidden_dim, cell in dim_entry.get("cells", {}).items()),
            key=lambda item: item[0],
        )
        if not cell_items:
            continue

        colors = plt.cm.viridis(np.linspace(0.10, 0.90, max(1, len(cell_items))))

        fig, ax = _make_axes()
        max_episode = 1.0
        for color, (hidden_dim, cell) in zip(colors, cell_items):
            curve = cell["aggregate_success_rate_curve"]
            episodes = np.asarray(curve["episodes"], dtype=np.float64)
            mean = np.asarray(curve["mean"], dtype=np.float64)
            std = np.asarray(curve["std"], dtype=np.float64)
            if episodes.size == 0:
                continue
            max_episode = max(max_episode, float(np.max(episodes)))
            ax.plot(
                episodes,
                mean,
                color=color,
                linewidth=2.0,
                label=f"D={hidden_dim} ({hidden_dim / predicted_threshold:.2f}x D*)",
            )
            ax.fill_between(
                episodes,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

        ax.set_title(
            (
                f"Learning threshold sweep | d={visible_dim}, K={basis_complexity}, "
                f"(2K)^d={num_frequency_vectors}, predicted D*={predicted_threshold:.1f}"
            ),
            loc="left",
            fontsize=11,
            pad=10,
        )
        ax.set_xlabel("Training episodes")
        ax.set_ylabel("Success rate")
        ax.set_xlim(0.0, max_episode)
        ax.set_ylim(0.0, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.legend(loc="best", frameon=False, fontsize=8.5)
        output_path = plots_root / "learning_curves" / f"success_rate_vis{visible_dim}.png"
        curve_plot_paths["success_rate"].append(_save_figure(fig, output_path))

        fig, ax = _make_axes()
        max_episode = 1.0
        y_values: list[np.ndarray] = []
        for color, (hidden_dim, cell) in zip(colors, cell_items):
            curve = cell["aggregate_objective_curve"]
            episodes = np.asarray(curve["episodes"], dtype=np.float64)
            mean = np.asarray(curve["mean"], dtype=np.float64)
            std = np.asarray(curve["std"], dtype=np.float64)
            if episodes.size == 0:
                continue
            max_episode = max(max_episode, float(np.max(episodes)))
            y_values.append(mean[np.isfinite(mean)])
            ax.plot(
                episodes,
                mean,
                color=color,
                linewidth=2.0,
                label=f"D={hidden_dim} ({hidden_dim / predicted_threshold:.2f}x D*)",
            )
            ax.fill_between(
                episodes,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

        finite_values = (
            np.concatenate([values for values in y_values if values.size > 0])
            if y_values
            else np.asarray([], dtype=np.float64)
        )
        ax.set_title(
            (
                f"Objective sweep | d={visible_dim}, K={basis_complexity}, "
                f"(2K)^d={num_frequency_vectors}, predicted D*={predicted_threshold:.1f}"
            ),
            loc="left",
            fontsize=11,
            pad=10,
        )
        ax.set_xlabel("Training episodes")
        ax.set_ylabel("Mean final objective")
        ax.set_xlim(0.0, max_episode)
        if finite_values.size > 0:
            y_min = float(np.min(finite_values))
            y_max = float(np.max(finite_values))
            margin = 0.08 * max(1e-6, y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.legend(loc="best", frameon=False, fontsize=8.5)
        output_path = plots_root / "learning_curves" / f"objective_vis{visible_dim}.png"
        curve_plot_paths["objective"].append(_save_figure(fig, output_path))

        fig, ax = _make_axes()
        max_episode = 1.0
        y_values = []
        for color, (hidden_dim, cell) in zip(colors, cell_items):
            curve = cell["aggregate_objective_curve_running_avg_100"]
            episodes = np.asarray(curve["episodes"], dtype=np.float64)
            mean = np.asarray(curve["mean"], dtype=np.float64)
            std = np.asarray(curve["std"], dtype=np.float64)
            if episodes.size == 0:
                continue
            max_episode = max(max_episode, float(np.max(episodes)))
            y_values.append(mean[np.isfinite(mean)])
            ax.plot(
                episodes,
                mean,
                color=color,
                linewidth=2.0,
                label=f"d={visible_dim}, D={hidden_dim}",
            )
            ax.fill_between(
                episodes,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

        finite_values = (
            np.concatenate([values for values in y_values if values.size > 0])
            if y_values
            else np.asarray([], dtype=np.float64)
        )
        ax.set_title(
            (
                f"E(F(z)) learning curves | d={visible_dim}, K={basis_complexity} | "
                "running average over 100 episodes"
            ),
            loc="left",
            fontsize=11,
            pad=10,
        )
        ax.set_xlabel("Training episodes")
        ax.set_ylabel("Observed objective E(F(z))")
        ax.set_xlim(0.0, max_episode)
        if finite_values.size > 0:
            y_min = float(np.min(finite_values))
            y_max = float(np.max(finite_values))
            margin = 0.08 * max(1e-6, y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.legend(loc="best", frameon=False, fontsize=8.3)
        output_path = (
            plots_root
            / "learning_curves"
            / f"objective_running_avg_100_vis{visible_dim}.png"
        )
        curve_plot_paths["objective_running_avg_100"].append(_save_figure(fig, output_path))

    return curve_plot_paths


def _plot_heatmap(
    *,
    matrix: np.ndarray,
    x_labels: list[int],
    y_labels: list[int],
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str,
    percent: bool = False,
) -> str:
    masked = np.ma.masked_invalid(matrix)
    fig_width = max(7.5, 0.58 * len(x_labels) + 2.8)
    fig_height = max(4.8, 0.70 * len(y_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_cmap = plt.get_cmap(cmap).copy()
    plot_cmap.set_bad(color="#f6f8fa")
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=plot_cmap)

    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.set_xlabel("Hidden dimension D")
    ax.set_ylabel("Visible dimension d")
    ax.set_xticks(np.arange(len(x_labels), dtype=np.float64))
    ax.set_xticklabels([str(value) for value in x_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels), dtype=np.float64))
    ax.set_yticklabels([str(value) for value in y_labels])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    finite_for_text = matrix[np.isfinite(matrix)]
    text_switch_value = float(np.mean(finite_for_text)) if finite_for_text.size > 0 else 0.0
    if len(x_labels) * len(y_labels) <= 80:
        for row_index, _visible_dim in enumerate(y_labels):
            for col_index, _hidden_dim in enumerate(x_labels):
                value = matrix[row_index, col_index]
                if not np.isfinite(value):
                    continue
                if percent:
                    text = f"{100.0 * value:.0f}%"
                elif value >= 1000.0:
                    text = f"{value / 1000.0:.1f}k"
                else:
                    text = f"{value:.0f}"
                ax.text(
                    col_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if float(value) > text_switch_value else "#111827",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(colorbar_label)
    if percent:
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    return _save_figure(fig, output_path)


def _plot_ratio_collapse(*, aggregate: dict[str, Any], output_path: Path) -> str:
    fig, ax = _make_axes()
    ax.axvline(1.0, color="#d73a49", linestyle="--", linewidth=1.5, alpha=0.9, label="Predicted D*")
    by_visible_dim = aggregate.get("by_visible_dim", {})
    colors = plt.cm.plasma(np.linspace(0.14, 0.88, max(1, len(by_visible_dim))))

    for color, key in zip(colors, sorted(by_visible_dim.keys(), key=lambda item: int(item))):
        dim_entry = by_visible_dim[key]
        points: list[tuple[float, float]] = []
        for hidden_dim, cell in sorted(
            ((int(hidden_dim), cell) for hidden_dim, cell in dim_entry.get("cells", {}).items()),
            key=lambda item: item[0],
        ):
            final_success = cell.get("final_success_rate_mean")
            threshold_ratio = cell.get("threshold_ratio")
            if final_success is None or threshold_ratio is None:
                continue
            points.append((float(threshold_ratio), float(final_success)))
        if not points:
            continue
        x = np.asarray([point[0] for point in points], dtype=np.float64)
        y = np.asarray([point[1] for point in points], dtype=np.float64)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            color=color,
            label=f"d={key}",
        )

    ax.set_title("Final success vs normalized hidden dimension", loc="left", fontsize=11, pad=10)
    ax.set_xlabel("D / D* (Section 7.3 prediction)")
    ax.set_ylabel("Mean final success rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="best", frameon=False, fontsize=9)
    return _save_figure(fig, output_path)


def _plot_observed_vs_predicted_threshold(*, aggregate: dict[str, Any], output_path: Path) -> str:
    by_visible_dim = aggregate.get("by_visible_dim", {})
    visible_dims = sorted(int(key) for key in by_visible_dim.keys())
    fig, ax = _make_axes()
    x = np.arange(len(visible_dims), dtype=np.float64)
    predicted = np.asarray(
        [float(by_visible_dim[str(d)]["predicted_threshold_exact"]) for d in visible_dims],
        dtype=np.float64,
    )
    observed = np.asarray(
        [
            float(by_visible_dim[str(d)]["observed_threshold_hidden_dim"])
            if by_visible_dim[str(d)]["observed_threshold_hidden_dim"] is not None
            else np.nan
            for d in visible_dims
        ],
        dtype=np.float64,
    )
    observed_final = np.asarray(
        [
            float(
                by_visible_dim[str(d)]["objective_thresholds"][
                    "final_objective_running_avg_100_near_best"
                ]["threshold_hidden_dim"]
            )
            if by_visible_dim[str(d)]["objective_thresholds"][
                "final_objective_running_avg_100_near_best"
            ]["threshold_hidden_dim"]
            is not None
            else np.nan
            for d in visible_dims
        ],
        dtype=np.float64,
    )
    observed_geom = np.asarray(
        [
            float(
                by_visible_dim[str(d)]["objective_thresholds"][
                    "geometric_mean_objective_running_avg_100_near_best"
                ]["threshold_hidden_dim"]
            )
            if by_visible_dim[str(d)]["objective_thresholds"][
                "geometric_mean_objective_running_avg_100_near_best"
            ]["threshold_hidden_dim"]
            is not None
            else np.nan
            for d in visible_dims
        ],
        dtype=np.float64,
    )

    ax.plot(x, predicted, marker="o", linewidth=2.0, color="#0969da", label="Predicted D*")
    valid_mask = np.isfinite(observed)
    if np.any(valid_mask):
        ax.plot(
            x[valid_mask],
            observed[valid_mask],
            marker="s",
            linewidth=1.6,
            color="#d73a49",
            linestyle=":",
            label="Legacy success-based threshold",
        )
    valid_final_mask = np.isfinite(observed_final)
    if np.any(valid_final_mask):
        ax.plot(
            x[valid_final_mask],
            observed_final[valid_final_mask],
            marker="s",
            linewidth=2.0,
            color="#1a7f37",
            label="Objective threshold: final near-best",
        )
    valid_geom_mask = np.isfinite(observed_geom)
    if np.any(valid_geom_mask):
        ax.plot(
            x[valid_geom_mask],
            observed_geom[valid_geom_mask],
            marker="^",
            linewidth=2.0,
            color="#bf8700",
            label="Objective threshold: geometric-mean near-best",
        )
    for index, visible_dim in enumerate(visible_dims):
        if np.isfinite(observed_final[index]) or np.isfinite(observed_geom[index]):
            continue
        max_hidden_dim = max(
            int(hidden_dim) for hidden_dim in by_visible_dim[str(visible_dim)].get("cells", {}).keys()
        )
        ax.annotate(
            f">{max_hidden_dim}",
            (x[index], predicted[index]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#57606a",
        )

    ax.set_title(
        "Objective-based empirical thresholds vs predicted D*",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("Visible dimension d")
    ax.set_ylabel("Hidden dimension D")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in visible_dims])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.legend(loc="best", frameon=False, fontsize=9)
    return _save_figure(fig, output_path)


def plot_from_manifest(*, manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    suite_root = Path(str(manifest["suite_root"])).expanduser().resolve()
    plots_root = Path(str(manifest["plots_root"])).expanduser().resolve()
    plot_data_root = Path(str(manifest["plot_data_root"])).expanduser().resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    plot_data_root.mkdir(parents=True, exist_ok=True)

    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["last_plotted_at_utc"] = datetime.now(timezone.utc).isoformat()

    aggregate = _aggregate_manifest(manifest)
    aggregate_path = plot_data_root / "visible_dim_threshold_aggregate.json"
    _write_json(aggregate_path, aggregate)

    curve_plot_paths = _plot_curves_by_visible_dim(aggregate=aggregate, plots_root=plots_root)

    all_visible_dims = sorted(int(key) for key in aggregate.get("by_visible_dim", {}).keys())
    all_d = sorted(
        {
            int(hidden_dim)
            for dim_entry in aggregate.get("by_visible_dim", {}).values()
            for hidden_dim in dim_entry.get("cells", {}).keys()
        }
    )
    success_matrix = np.full((len(all_visible_dims), len(all_d)), np.nan, dtype=np.float64)
    episode_matrix = np.full((len(all_visible_dims), len(all_d)), np.nan, dtype=np.float64)
    for row_index, visible_dim in enumerate(all_visible_dims):
        dim_entry = aggregate["by_visible_dim"][str(visible_dim)]
        for col_index, hidden_dim in enumerate(all_d):
            cell = dim_entry.get("cells", {}).get(str(hidden_dim))
            if not isinstance(cell, dict):
                continue
            final_success = cell.get("final_success_rate_mean")
            crossing_episode = cell.get("episodes_to_success_rate_threshold")
            if final_success is not None:
                success_matrix[row_index, col_index] = float(final_success)
            if crossing_episode is not None:
                episode_matrix[row_index, col_index] = float(crossing_episode)

    heatmap_paths = {
        "final_success_heatmap": _plot_heatmap(
            matrix=success_matrix,
            x_labels=all_d,
            y_labels=all_visible_dims,
            title="Final success rate across visible dimension and D",
            colorbar_label="Mean final success rate",
            output_path=plots_root / "final_success_heatmap.png",
            cmap="viridis",
            percent=True,
        ),
        "episodes_to_threshold_heatmap": _plot_heatmap(
            matrix=episode_matrix,
            x_labels=all_d,
            y_labels=all_visible_dims,
            title=(
                "Episodes until aggregate success reaches "
                f"{100.0 * float(aggregate['success_rate_threshold']):.0f}%"
            ),
            colorbar_label="Episode of first threshold crossing",
            output_path=plots_root / "episodes_to_threshold_heatmap.png",
            cmap="magma_r",
            percent=False,
        ),
        "final_success_vs_ratio": _plot_ratio_collapse(
            aggregate=aggregate,
            output_path=plots_root / "final_success_vs_ratio.png",
        ),
        "observed_vs_predicted_threshold": _plot_observed_vs_predicted_threshold(
            aggregate=aggregate,
            output_path=plots_root / "observed_vs_predicted_threshold.png",
        ),
    }

    manifest["aggregate_json"] = str(aggregate_path.resolve())
    manifest["curve_plot_paths"] = curve_plot_paths
    manifest["summary_plot_paths"] = heatmap_paths
    manifest["success_thresholds_by_visible_dim"] = {
        key: {
            "success_rate_threshold": aggregate["success_rate_threshold"],
            "observed_threshold_hidden_dim": aggregate["by_visible_dim"][key][
                "observed_threshold_hidden_dim"
            ],
            "observed_threshold_ratio": aggregate["by_visible_dim"][key][
                "observed_threshold_ratio"
            ],
        }
        for key in sorted(aggregate.get("by_visible_dim", {}).keys(), key=lambda item: int(item))
    }
    manifest["objective_thresholds_by_visible_dim"] = {
        key: {
            "predicted_threshold_exact": aggregate["by_visible_dim"][key]["predicted_threshold_exact"],
            "predicted_threshold_asymptotic": aggregate["by_visible_dim"][key][
                "predicted_threshold_asymptotic"
            ],
            "final_objective_running_avg_100_near_best": aggregate["by_visible_dim"][key][
                "objective_thresholds"
            ]["final_objective_running_avg_100_near_best"],
            "geometric_mean_objective_running_avg_100_near_best": aggregate["by_visible_dim"][key][
                "objective_thresholds"
            ]["geometric_mean_objective_running_avg_100_near_best"],
        }
        for key in sorted(aggregate.get("by_visible_dim", {}).keys(), key=lambda item: int(item))
    }
    manifest["observed_thresholds_by_visible_dim"] = {
        key: {
            "predicted_threshold_exact": aggregate["by_visible_dim"][key]["predicted_threshold_exact"],
            "predicted_threshold_asymptotic": aggregate["by_visible_dim"][key][
                "predicted_threshold_asymptotic"
            ],
            "final_objective_running_avg_100_near_best": aggregate["by_visible_dim"][key][
                "objective_thresholds"
            ]["final_objective_running_avg_100_near_best"],
            "geometric_mean_objective_running_avg_100_near_best": aggregate["by_visible_dim"][key][
                "objective_thresholds"
            ]["geometric_mean_objective_running_avg_100_near_best"],
        }
        for key in sorted(aggregate.get("by_visible_dim", {}).keys(), key=lambda item: int(item))
    }

    _write_json(manifest_path, manifest)

    suite_summary = {
        "manifest_path": str(manifest_path.resolve()),
        "aggregate_json": str(aggregate_path.resolve()),
        "plots_root": str(plots_root.resolve()),
        "curve_plot_paths": curve_plot_paths,
        "summary_plot_paths": heatmap_paths,
        "success_thresholds_by_visible_dim": manifest["success_thresholds_by_visible_dim"],
        "objective_thresholds_by_visible_dim": manifest["objective_thresholds_by_visible_dim"],
        "observed_thresholds_by_visible_dim": manifest["observed_thresholds_by_visible_dim"],
        "plot_cmd": (
            "python3 -m tasks.spatial.run_visible_dim_threshold_experiment "
            f"--manifest {manifest_path}"
        ),
    }
    summary_path = suite_root / "suite_summary.json"
    _write_json(summary_path, suite_summary)
    suite_summary["summary_path"] = str(summary_path.resolve())
    return suite_summary


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Sweep visible dimension d and hidden dimension D at fixed spatial Fourier "
            "basis complexity K to test the Section 7.3 hidden-dimension threshold prediction."
        )
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="If set, skip training and regenerate plots/summary from an existing manifest.",
    )

    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument(
        "--suite_output_dir",
        type=str,
        default="plots/spatial_visible_dim_threshold_experiment",
        help="Directory for the manifest, aggregate JSON, and generated plots.",
    )
    parser.add_argument(
        "--suite_name",
        type=str,
        default="spatial_visible_dim_threshold_experiment",
    )
    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--run_name_prefix", type=str, default="spatial_visible_dim_threshold")
    parser.add_argument(
        "--basis_complexity",
        type=int,
        default=2,
        help="Fixed Fourier basis complexity K used for the sweep.",
    )
    parser.add_argument(
        "--freq_sparsity",
        type=int,
        default=defaults.spatial_freq_sparsity,
        help=(
            "Max nonzero components per Fourier frequency vector (interaction order r). "
            "0 = dense (all d components, original behavior). "
            "1 = axis-aligned only. 2 = pairwise interactions. etc."
        ),
    )
    parser.add_argument(
        "--visible_dims",
        type=str,
        default="2,3,4,5,6",
        help="Comma-separated visible dimensions d to sweep.",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="",
        help=(
            "Optional explicit comma-separated D values used for every visible dimension. "
            "If omitted, --hidden_dim_ratios are converted into d-specific D values."
        ),
    )
    parser.add_argument(
        "--hidden_dim_ratios",
        type=str,
        default="0.25,0.5,0.75,1.0,1.25,1.5,2.0",
        help=(
            "If --hidden_dims is not provided, generate each d sweep from "
            "ratio * predicted D*."
        ),
    )
    parser.add_argument(
        "--max_hidden_dim",
        type=int,
        default=320,
        help="Upper cap applied only when D values are generated from --hidden_dim_ratios.",
    )
    parser.add_argument(
        "--success_rate_threshold",
        type=float,
        default=0.8,
        help="Aggregate success-rate threshold used to define the observed learning threshold.",
    )
    parser.add_argument(
        "--objective_threshold_tolerance",
        type=float,
        default=0.1,
        help="Near-best tolerance for objective-based empirical thresholds.",
    )
    parser.add_argument(
        "--objective_log_epsilon",
        type=float,
        default=1e-6,
        help="Stability epsilon used for geometric-mean objective scoring.",
    )
    parser.add_argument("--train_steps", type=int, default=defaults.train_steps)
    parser.add_argument("--n_env", type=int, default=defaults.n_env)
    parser.add_argument("--rollout_len", type=int, default=defaults.rollout_len)
    parser.add_argument("--running_avg_window", type=int, default=defaults.running_avg_window)
    parser.add_argument(
        "--save_metrics_interval_episodes",
        type=int,
        default=defaults.save_metrics_interval_episodes,
    )
    parser.add_argument("--eval_interval_episodes", type=int, default=defaults.eval_interval_episodes)
    parser.add_argument("--max_horizon", type=int, default=defaults.max_horizon)
    parser.add_argument("--policy_hidden_dim", type=int, default=64)
    parser.add_argument(
        "--oracle_proj_dim",
        type=int,
        default=defaults.oracle_proj_dim,
        help="Oracle-token projection width before the policy trunk (0 disables projection).",
    )
    parser.add_argument("--token_embed_dim", type=int, default=defaults.token_embed_dim)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--ppo_epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)
    parser.add_argument("--spatial_coord_limit", type=int, default=3)
    parser.add_argument(
        "--base_lr",
        dest="spatial_step_size",
        type=float,
        default=defaults.spatial_step_size,
        help="Base step size used by the visible-space dynamics.",
    )
    parser.add_argument(
        "--spatial_success_threshold",
        type=float,
        default=0.01,
        help="Fixed task success threshold during training/eval.",
    )
    parser.add_argument("--sensing", type=str, choices=["S0", "S1"], default=defaults.sensing)
    parser.add_argument(
        "--disable_success_curriculum",
        dest="disable_success_curriculum",
        action="store_true",
        default=True,
        help="Disable the spatial success-threshold curriculum. This is now the default.",
    )
    parser.add_argument(
        "--enable_success_curriculum",
        dest="disable_success_curriculum",
        action="store_false",
        help="Enable the spatial success-threshold curriculum.",
    )
    parser.add_argument(
        "--spatial_refresh_map_each_episode",
        action="store_true",
        default=defaults.spatial_refresh_map_each_episode,
    )
    parser.add_argument("--ppo_step_scale", type=float, default=defaults.ppo_step_scale)
    parser.add_argument(
        "--disable_spatial_baseline_lr_tuning",
        action="store_true",
        help="Disable automatic lr tuning for GD/Adam baselines.",
    )
    parser.add_argument(
        "--spatial_baseline_lr_candidates",
        type=str,
        default=defaults.spatial_baseline_lr_candidates,
    )
    parser.add_argument(
        "--spatial_baseline_lr_tune_tasks",
        type=int,
        default=defaults.spatial_baseline_lr_tune_tasks,
    )
    parser.add_argument(
        "--enable_spatial_baselines",
        action="store_true",
        help="Enable GD/Adam/no-oracle/visible-gradient baselines for the threshold sweep.",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Ignore any existing run directories and retrain every d/D/seed condition.",
    )
    parser.add_argument(
        "--skip_plotting",
        action="store_true",
        help="Train runs and update the manifest, but skip aggregate plotting.",
    )
    return parser.parse_args()


def main() -> None:
    defaults = TrainConfig()
    args = parse_args()
    if args.manifest.strip():
        summary = plot_from_manifest(manifest_path=Path(args.manifest))
        print(json.dumps(summary, indent=2))
        return

    seeds = _parse_int_list(args.seeds, "seeds")
    visible_dims = _parse_int_list(args.visible_dims, "visible_dims", min_value=2)
    explicit_hidden_dims = (
        _parse_int_list(args.hidden_dims, "hidden_dims", min_value=1)
        if str(args.hidden_dims).strip()
        else None
    )
    hidden_dim_ratios = _parse_float_list(
        args.hidden_dim_ratios,
        "hidden_dim_ratios",
        min_value=0.0,
    )
    if int(args.basis_complexity) < 1:
        raise ValueError("basis_complexity must be >= 1")
    if not 0.0 < float(args.success_rate_threshold) <= 1.0:
        raise ValueError("success_rate_threshold must be in (0, 1]")
    if float(args.objective_threshold_tolerance) < 0.0:
        raise ValueError("objective_threshold_tolerance must be >= 0")
    if float(args.objective_log_epsilon) <= 0.0:
        raise ValueError("objective_log_epsilon must be > 0")
    if int(args.oracle_proj_dim) < 0:
        raise ValueError("oracle_proj_dim must be >= 0")
    if int(args.max_hidden_dim) < 1 and explicit_hidden_dims is None:
        raise ValueError("max_hidden_dim must be >= 1 when hidden dims are ratio-generated")

    hidden_dim_grid, predictions = _build_hidden_dim_grid(
        basis_complexity=int(args.basis_complexity),
        visible_dims=visible_dims,
        explicit_hidden_dims=explicit_hidden_dims,
        hidden_dim_ratios=hidden_dim_ratios,
        max_hidden_dim=(None if explicit_hidden_dims is not None else int(args.max_hidden_dim)),
    )

    suite_root = Path(args.suite_output_dir).expanduser().resolve()
    plots_root = suite_root / "plots"
    plot_data_root = suite_root / "plot_data"
    suite_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    plot_data_root.mkdir(parents=True, exist_ok=True)

    manifest_path = suite_root / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}

    base_config = TrainConfig(
        task="spatial",
        train_steps=int(args.train_steps),
        n_env=int(args.n_env),
        rollout_len=int(args.rollout_len),
        running_avg_window=int(args.running_avg_window),
        save_metrics_interval_episodes=int(args.save_metrics_interval_episodes),
        eval_interval_episodes=int(args.eval_interval_episodes),
        max_horizon=int(args.max_horizon),
        hidden_dim=int(args.policy_hidden_dim),
        oracle_proj_dim=int(args.oracle_proj_dim),
        token_embed_dim=int(args.token_embed_dim),
        logdir=str(args.logdir),
        device=str(args.device),
        sensing=str(args.sensing),
        lr=float(args.lr),
        ppo_epochs=int(args.ppo_epochs),
        minibatches=int(args.minibatches),
        spatial_coord_limit=int(args.spatial_coord_limit),
        spatial_step_size=float(args.spatial_step_size),
        ppo_step_scale=float(args.ppo_step_scale),
        spatial_success_threshold=float(args.spatial_success_threshold),
        spatial_enable_success_curriculum=not bool(args.disable_success_curriculum),
        spatial_basis_complexity=int(args.basis_complexity),
        spatial_freq_sparsity=int(args.freq_sparsity),
        spatial_f_type="FOURIER",
        spatial_policy_arch=defaults.spatial_policy_arch,
        spatial_refresh_map_each_episode=bool(args.spatial_refresh_map_each_episode),
        spatial_fixed_start_target=False,
        spatial_plot_interval_episodes=0,
        spatial_enable_baselines=bool(args.enable_spatial_baselines),
        spatial_tune_baseline_lrs=not bool(args.disable_spatial_baseline_lr_tuning),
        spatial_early_stop_on_all_methods_success=False,
        spatial_baseline_lr_candidates=str(args.spatial_baseline_lr_candidates),
        spatial_baseline_lr_tune_tasks=int(args.spatial_baseline_lr_tune_tasks),
        spatial_enable_optimization_curve_eval=False,
        enable_training_plots=False,
        oracle_mode="convex_gradient",
        spatial_visible_dim=int(visible_dims[0]),
        spatial_hidden_dim=int(hidden_dim_grid[int(visible_dims[0])][0]),
        spatial_token_dim=int(hidden_dim_grid[int(visible_dims[0])][0]),
    )

    manifest.update(
        {
            "version": 1,
            "created_at_utc": manifest.get("created_at_utc", datetime.now(timezone.utc).isoformat()),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "suite_name": str(args.suite_name),
            "suite_root": str(suite_root.resolve()),
            "plots_root": str(plots_root.resolve()),
            "plot_data_root": str(plot_data_root.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "seeds": [int(seed) for seed in seeds],
            "basis_complexity": int(args.basis_complexity),
            "visible_dims": [int(value) for value in visible_dims],
            "hidden_dim_grid": {
                str(key): [int(value) for value in values] for key, values in hidden_dim_grid.items()
            },
            "success_rate_threshold": float(args.success_rate_threshold),
            "objective_threshold_tolerance": float(args.objective_threshold_tolerance),
            "objective_log_epsilon": float(args.objective_log_epsilon),
            "base_config": asdict(base_config),
            "predictions": {str(key): value for key, value in predictions.items()},
        }
    )
    existing_runs = manifest.get("runs", {})
    if not isinstance(existing_runs, dict):
        existing_runs = {}
    expected_run_keys = {
        f"vis{int(visible_dim)}_D{int(hidden_dim)}_seed{int(seed)}"
        for visible_dim in visible_dims
        for hidden_dim in hidden_dim_grid[int(visible_dim)]
        for seed in seeds
    }
    manifest["runs"] = {
        key: value for key, value in existing_runs.items() if key in expected_run_keys
    }
    _write_json(manifest_path, manifest)

    for visible_dim in visible_dims:
        prediction = predictions[int(visible_dim)]
        num_frequency_vectors = int(prediction["num_frequency_vectors"])
        predicted_threshold = float(prediction["predicted_threshold_exact"])
        for hidden_dim in hidden_dim_grid[int(visible_dim)]:
            for seed in seeds:
                run_name = (
                    f"{args.run_name_prefix}_vis{int(visible_dim)}_K{int(args.basis_complexity)}"
                    f"_D{int(hidden_dim)}"
                    f"_T{int(base_config.train_steps)}"
                    f"_H{int(base_config.max_horizon)}"
                    f"_W{int(base_config.hidden_dim)}"
                    f"_P{int(base_config.oracle_proj_dim)}"
                    f"_C{int(bool(base_config.spatial_enable_success_curriculum))}"
                    f"_B{int(bool(base_config.spatial_enable_baselines))}"
                    f"_BT{int(bool(base_config.spatial_tune_baseline_lrs))}"
                    f"_PS{_value_tag(base_config.ppo_step_scale)}"
                    f"_seed{int(seed)}"
                )
                config = replace(
                    base_config,
                    seed=int(seed),
                    run_name=run_name,
                    spatial_visible_dim=int(visible_dim),
                    spatial_hidden_dim=int(hidden_dim),
                    spatial_token_dim=int(hidden_dim),
                )
                run_dir = config.resolve_run_dir().expanduser().resolve()
                if args.force_retrain or not _run_artifacts_exist(run_dir):
                    output = run_training(config, return_artifacts=False)
                    run_dir = Path(output["summary"]["run_dir"]).expanduser().resolve()

                expected_unique = _expected_unique_frequency_vectors(
                    int(hidden_dim),
                    num_frequency_vectors,
                )
                run_key = f"vis{int(visible_dim)}_D{int(hidden_dim)}_seed{int(seed)}"
                manifest["runs"][run_key] = {
                    "key": run_key,
                    "visible_dim": int(visible_dim),
                    "basis_complexity": int(args.basis_complexity),
                    "hidden_dim": int(hidden_dim),
                    "seed": int(seed),
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "config_json": str((run_dir / "config.json").resolve()),
                    "metrics_csv": str((run_dir / "metrics.csv").resolve()),
                    "metrics_jsonl": str((run_dir / "metrics.jsonl").resolve()),
                    "summary_json": str((run_dir / "summary.json").resolve()),
                    "num_frequency_vectors": int(num_frequency_vectors),
                    "predicted_threshold_exact": float(predicted_threshold),
                    "predicted_threshold_asymptotic": float(
                        prediction["predicted_threshold_asymptotic"]
                    ),
                    "effective_train_steps": int(config.train_steps),
                    "effective_max_horizon": int(config.max_horizon),
                    "effective_policy_hidden_dim": int(config.hidden_dim),
                    "effective_success_curriculum": bool(config.spatial_enable_success_curriculum),
                    "threshold_ratio": float(hidden_dim / predicted_threshold),
                    "expected_unique_frequency_vectors": float(expected_unique),
                    "expected_coverage_fraction": float(
                        expected_unique / max(1.0, float(num_frequency_vectors))
                    ),
                }
                manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(manifest_path, manifest)

    if args.skip_plotting:
        print(
            json.dumps(
                {
                    "manifest_path": str(manifest_path.resolve()),
                    "suite_root": str(suite_root.resolve()),
                    "plot_cmd": (
                        "python3 -m tasks.spatial.run_visible_dim_threshold_experiment "
                        f"--manifest {manifest_path}"
                    ),
                },
                indent=2,
            )
        )
        return

    summary = plot_from_manifest(manifest_path=manifest_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
