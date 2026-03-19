import argparse
import json
from pathlib import Path

import numpy as np

from .plotting import plot_success_curve


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

    running_avg_window = int(config.get("running_avg_window", 100))
    sensing = str(config.get("sensing", "S0"))
    reward_noise_std = float(config.get("reward_noise_std", 0.0))

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

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
        f"task=graph, {title_context}, sensing={sensing}, "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning curves for a saved graph training run"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Single run directory containing config.json and metrics.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory for generated PNGs (defaults to run_dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir is not None:
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
        raise SystemExit("--run_dir must be provided")


if __name__ == "__main__":
    main()
