import argparse
import json
import re
from pathlib import Path

import numpy as np

from v2.plotting import (
    plot_objective_curve,
    plot_objective_curve_multi_seed,
    plot_success_curve,
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _median_episode_spacing(metric_rows: list[dict]) -> int:
    episodes: list[int] = []
    for row in metric_rows:
        value = row.get("episodes")
        if value is None:
            continue
        episodes.append(int(value))
    if len(episodes) < 2:
        return 1
    episodes = sorted(set(episodes))
    if len(episodes) < 2:
        return 1
    diffs = np.diff(np.asarray(episodes, dtype=np.int64))
    if diffs.size == 0:
        return 1
    return int(np.median(diffs))


def _build_graph_curve_metrics(metric_rows: list[dict]) -> list[dict]:
    curve_metrics: list[dict] = []
    for row in metric_rows:
        episodes = row.get("episodes")
        success_rate = row.get("success_rate")
        if episodes is None or success_rate is None:
            continue
        item = {
            "episodes": int(episodes),
            "success_rate": float(success_rate),
        }
        success = row.get("success")
        if success is not None:
            item["success"] = float(success)
        curve_metrics.append(item)
    return curve_metrics


def _discover_spatial_runs(
    runs_dir: Path, run_prefix: str
) -> list[tuple[Path, int]]:
    """Find spatial runs matching prefix_seed{N}. Returns [(run_dir, seed), ...]."""
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        return []
    pattern = re.compile(re.escape(run_prefix) + r"_seed(\d+)$")
    found: list[tuple[Path, int]] = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            seed = int(match.group(1))
            config_path = child / "config.json"
            metrics_path = child / "metrics.jsonl"
            if config_path.exists() and metrics_path.exists():
                config = _read_json(config_path)
                if str(config.get("task", "")) == "spatial":
                    found.append((child, seed))
    return sorted(found, key=lambda x: x[1])


def _build_spatial_curve_metrics(metric_rows: list[dict]) -> list[dict]:
    curve_metrics: list[dict] = []
    for row in metric_rows:
        objective_value = float(row.get("avg_final_objective", float("nan")))
        if not np.isfinite(objective_value):
            continue
        curve_metrics.append(
            {
                "episodes": int(row["episodes"]),
                "success_rate": float(row.get("success_rate", float("nan"))),
                "baseline_success_rate": float(
                    row.get("avg_baseline_success_rate", float("nan"))
                ),
                "sgd_baseline_success_rate": float(
                    row.get("avg_sgd_baseline_success_rate", float("nan"))
                ),
                "no_oracle_success_rate": float(
                    row.get("avg_no_oracle_success_rate", float("nan"))
                ),
                "objective_value": objective_value,
                "baseline_objective_value": float(
                    row.get("avg_baseline_final_objective", float("nan"))
                ),
                "sgd_baseline_objective_value": float(
                    row.get("avg_sgd_baseline_final_objective", float("nan"))
                ),
                "no_oracle_objective_value": float(
                    row.get("avg_no_oracle_final_objective", float("nan"))
                ),
                "distance_value": float(row.get("avg_final_ref_distance", float("nan"))),
                "baseline_distance_value": float(
                    row.get("avg_baseline_final_ref_distance", float("nan"))
                ),
                "sgd_baseline_distance_value": float(
                    row.get("avg_sgd_baseline_final_ref_distance", float("nan"))
                ),
                "no_oracle_distance_value": float(
                    row.get("avg_no_oracle_final_ref_distance", float("nan"))
                ),
            }
        )
    return curve_metrics


def _png_name(base: str, suffix: str | None) -> str:
    return f"{base}_{suffix}.png" if suffix else f"{base}.png"


def plot_learning_curves_for_run(
    run_dir: Path,
    output_dir: Path | None = None,
    filename_suffix: str | None = None,
) -> list[Path]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_dir = (
        output_dir.expanduser().resolve() if output_dir is not None else resolved_run_dir
    )
    config_path = resolved_run_dir / "config.json"
    metrics_path = resolved_run_dir / "metrics.jsonl"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    config = _read_json(config_path)
    metric_rows = _read_jsonl(metrics_path)
    if not metric_rows:
        raise ValueError(f"No metrics found in {metrics_path}")

    task = str(config.get("task", "graph"))
    running_avg_window = int(config.get("running_avg_window", 100))
    sensing = str(config.get("sensing", "S0"))
    reward_noise_std = float(config.get("reward_noise_std", 0.0))
    sgd_noise_std = float(config.get("spatial_sgd_gradient_noise_std", 0.1))

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    if task == "graph":
        spacing = _median_episode_spacing(metric_rows)
        if spacing > 1:
            print(
                f"warning: metrics rows are spaced every ~{spacing} episodes; "
                "raw per-episode plotting is limited by available data."
            )
        curve_metrics = _build_graph_curve_metrics(metric_rows)
        title_context = (
            f"n={int(config.get('n', 0))}, d={int(config.get('d', 0))}, "
            f"mode={config.get('oracle_mode', 'unknown')}, "
            f"sigma={int(config.get('sigma_size', 0))}"
        )
        title = (
            f"success_rate (running avg {running_avg_window}) vs episodes | "
            f"task={task}, {title_context}, sensing={sensing}, "
            f"reward_noise_std={reward_noise_std:g}"
        )
        output_path = resolved_output_dir / "success_rate_vs_episodes.png"
        plot_success_curve(
            curve_metrics,
            output_path,
            title,
            running_avg_window=running_avg_window,
        )
        output_paths.append(output_path)
        return output_paths

    if task != "spatial":
        raise ValueError(f"Unsupported task: {task}")

    curve_metrics = _build_spatial_curve_metrics(metric_rows)
    has_success_rate = any(
        np.isfinite(float(m.get("success_rate", float("nan")))) for m in curve_metrics
    )
    if has_success_rate:
        success_rate_title = (
            f"Success rate (running avg {running_avg_window}) vs episodes | "
            f"task={task}, sensing={sensing}, reward_noise_std={reward_noise_std:g}"
        )
        success_rate_output = resolved_output_dir / _png_name(
            "success_rate_vs_episodes", filename_suffix
        )
        plot_objective_curve(
            curve_metrics,
            success_rate_output,
            title=success_rate_title,
            value_key="success_rate",
            y_label="Success rate",
            main_label="PPO with oracle",
            baseline_value_key="baseline_success_rate",
            baseline_label="Visible GD",
            secondary_baseline_value_key="sgd_baseline_success_rate",
            secondary_baseline_label="Visible SGD",
            comparison_value_key="no_oracle_success_rate",
            comparison_label="PPO (no oracle)",
            y_axis_formatter="percent",
        )
        output_paths.append(success_rate_output)

    title_context = (
        f"D={int(config.get('spatial_hidden_dim', 0))}, "
        f"visible={int(config.get('spatial_visible_dim', 0))}, "
        f"mode={config.get('oracle_mode', 'unknown')}, "
        f"k={int(config.get('spatial_token_dim', 0))}"
    )
    objective_title = (
        f"E(F(z)) (running avg {running_avg_window}) vs episodes | "
        f"task={task}, {title_context}, sensing={sensing}, "
        f"reward_noise_std={reward_noise_std:g}"
    )
    objective_output = resolved_output_dir / _png_name(
        "objective_vs_episodes", filename_suffix
    )
    plot_objective_curve(
        curve_metrics,
        objective_output,
        title=objective_title,
        value_key="objective_value",
        y_label="E(F(z))",
        main_label="PPO with oracle",
        baseline_value_key="baseline_objective_value",
        baseline_label="Visible GD",
        secondary_baseline_value_key="sgd_baseline_objective_value",
        secondary_baseline_label="Visible SGD",
        comparison_value_key="no_oracle_objective_value",
        comparison_label="PPO (no oracle)",
    )
    output_paths.append(objective_output)

    distance_title = (
        f"distance to reference min (running avg {running_avg_window}) vs episodes | "
        f"task={task}, {title_context}, sensing={sensing}, "
        f"reward_noise_std={reward_noise_std:g}"
    )
    distance_output = resolved_output_dir / _png_name(
        "distance_vs_episodes", filename_suffix
    )
    plot_objective_curve(
        curve_metrics,
        distance_output,
        title=distance_title,
        value_key="distance_value",
        y_label="Euclidean distance to reference min",
        main_label="PPO with oracle",
        baseline_value_key="baseline_distance_value",
        baseline_label="Visible GD",
        secondary_baseline_value_key="sgd_baseline_distance_value",
        secondary_baseline_label="Visible SGD",
        comparison_value_key="no_oracle_distance_value",
        comparison_label="PPO (no oracle)",
    )
    output_paths.append(distance_output)
    return output_paths


def plot_learning_curves_multi_seed(
    runs_dir: Path,
    run_prefix: str,
    output_dir: Path | None = None,
) -> list[Path]:
    """Discover spatial runs, plot individual curves (with seed in filename), and average curves."""
    runs_dir = runs_dir.expanduser().resolve()
    resolved_output_dir = (
        output_dir.expanduser().resolve() if output_dir is not None else runs_dir
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_spatial_runs(runs_dir, run_prefix)
    if not runs:
        raise ValueError(
            f"No spatial runs found matching {run_prefix}_seed* in {runs_dir}"
        )

    config = _read_json(runs[0][0] / "config.json")
    running_avg_window = int(config.get("running_avg_window", 100))
    sensing = str(config.get("sensing", "S0"))
    reward_noise_std = float(config.get("reward_noise_std", 0.0))
    sgd_noise_std = float(config.get("spatial_sgd_gradient_noise_std", 0.1))
    title_context = (
        f"D={int(config.get('spatial_hidden_dim', 0))}, "
        f"visible={int(config.get('spatial_visible_dim', 0))}, "
        f"mode={config.get('oracle_mode', 'unknown')}, "
        f"k={int(config.get('spatial_token_dim', 0))}"
    )

    output_paths: list[Path] = []
    seed_metrics: list[tuple[int, list[dict]]] = []

    for run_dir, seed in runs:
        metric_rows = _read_jsonl(run_dir / "metrics.jsonl")
        if not metric_rows:
            continue
        curve_metrics = _build_spatial_curve_metrics(metric_rows)
        if not curve_metrics:
            continue
        seed_metrics.append((seed, curve_metrics))

        suffix = f"seed{seed}"
        paths = plot_learning_curves_for_run(
            run_dir,
            output_dir=resolved_output_dir,
            filename_suffix=suffix,
        )
        output_paths.extend(paths)

    if not seed_metrics:
        raise ValueError("No valid metrics found in any run")

    task = "spatial"
    has_success_rate = any(
        np.isfinite(float(m.get("success_rate", float("nan"))))
        for _, cm in seed_metrics
        for m in cm
    )

    if has_success_rate:
        success_rate_title = (
            f"Success rate (mean ± std over {len(seed_metrics)} seeds) | "
            f"task={task}, sensing={sensing}, reward_noise_std={reward_noise_std:g}"
        )
        plot_objective_curve_multi_seed(
            seed_metrics,
            resolved_output_dir / "success_rate_vs_episodes_mean.png",
            title=success_rate_title,
            value_key="success_rate",
            y_label="Success rate",
            baseline_value_key="baseline_success_rate",
            baseline_label="Visible GD",
            secondary_baseline_value_key="sgd_baseline_success_rate",
            secondary_baseline_label="Visible SGD",
            comparison_value_key="no_oracle_success_rate",
            comparison_label="PPO (no oracle)",
            y_axis_formatter="percent",
        )
        output_paths.append(resolved_output_dir / "success_rate_vs_episodes_mean.png")

    objective_title = (
        f"E(F(z)) (mean ± std over {len(seed_metrics)} seeds) | "
        f"task={task}, {title_context}, sensing={sensing}, "
        f"reward_noise_std={reward_noise_std:g}"
    )
    plot_objective_curve_multi_seed(
        seed_metrics,
        resolved_output_dir / "objective_vs_episodes_mean.png",
        title=objective_title,
        value_key="objective_value",
        y_label="E(F(z))",
        baseline_value_key="baseline_objective_value",
        baseline_label="Visible GD",
        secondary_baseline_value_key="sgd_baseline_objective_value",
        secondary_baseline_label="Visible SGD",
        comparison_value_key="no_oracle_objective_value",
        comparison_label="PPO (no oracle)",
    )
    output_paths.append(resolved_output_dir / "objective_vs_episodes_mean.png")

    distance_title = (
        f"distance to reference min (mean ± std over {len(seed_metrics)} seeds) | "
        f"task={task}, {title_context}, sensing={sensing}, "
        f"reward_noise_std={reward_noise_std:g}"
    )
    plot_objective_curve_multi_seed(
        seed_metrics,
        resolved_output_dir / "distance_vs_episodes_mean.png",
        title=distance_title,
        value_key="distance_value",
        y_label="Euclidean distance to reference min",
        baseline_value_key="baseline_distance_value",
        baseline_label="Visible GD",
        secondary_baseline_value_key="sgd_baseline_distance_value",
        secondary_baseline_label="Visible SGD",
        comparison_value_key="no_oracle_distance_value",
        comparison_label="PPO (no oracle)",
    )
    output_paths.append(resolved_output_dir / "distance_vs_episodes_mean.png")

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning curves for a saved training run or multi-seed spatial runs"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Single run directory containing config.json and metrics.jsonl",
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=None,
        help="Directory containing multiple run subdirs (use with --run_prefix for multi-seed)",
    )
    parser.add_argument(
        "--run_prefix",
        type=str,
        default=None,
        help="Prefix for run dirs, e.g. 'spatial_with_gd_baseline_curve' matches spatial_with_gd_baseline_curve_seed*",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory for generated PNGs (defaults to run_dir or runs_dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs_dir is not None and args.run_prefix is not None:
        output_paths = plot_learning_curves_multi_seed(
            args.runs_dir, args.run_prefix, args.output_dir
        )
        print(
            json.dumps(
                {
                    "runs_dir": str(args.runs_dir),
                    "run_prefix": args.run_prefix,
                    "output_paths": [str(path) for path in output_paths],
                },
                indent=2,
            )
        )
    elif args.run_dir is not None:
        output_paths = plot_learning_curves_for_run(args.run_dir, args.output_dir)
        print(
            json.dumps(
                {
                    "run_dir": str(args.run_dir),
                    "output_paths": [str(path) for path in output_paths],
                },
                indent=2,
            )
        )
    else:
        raise SystemExit(
            "Either --run_dir or (--runs_dir and --run_prefix) must be provided"
        )


if __name__ == "__main__":
    main()
