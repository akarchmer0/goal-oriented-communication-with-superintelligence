import argparse
import errno
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from v2.config import PlotConfig, RequestedExperimentsConfig

PLOT_DEFAULTS = PlotConfig()
REQUESTED_DEFAULTS = RequestedExperimentsConfig()
MAX_X_EPISODES = 50_000
MAX_X_EPISODES_GRAPH_SCALING = 10_000


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path, retries: int = 2, initial_delay_s: float = 0.2) -> list[dict]:
    delay_s = initial_delay_s
    for attempt in range(retries + 1):
        rows: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows
        except OSError as exc:
            is_timeout = isinstance(exc, TimeoutError) or exc.errno == errno.ETIMEDOUT
            if not is_timeout or attempt >= retries:
                raise
            time.sleep(delay_s)
            delay_s *= 2.0
    return []


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _extract_curve(metrics: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not metrics:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    episodes = np.asarray([float(item["episodes"]) for item in metrics], dtype=np.float64)
    success_rate = np.asarray([float(item["success_rate"]) for item in metrics], dtype=np.float64)
    mask = np.isfinite(episodes) & np.isfinite(success_rate)
    episodes = episodes[mask]
    success_rate = np.clip(success_rate[mask], 0.0, 1.0)
    return episodes, success_rate


def _extract_metric_curve(
    metrics: list[dict],
    metric_key: str,
    clip_range: tuple[float | None, float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not metrics:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    rows = [item for item in metrics if "episodes" in item and metric_key in item]
    if not rows:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    episodes = np.asarray([float(item["episodes"]) for item in rows], dtype=np.float64)
    values = np.asarray([float(item[metric_key]) for item in rows], dtype=np.float64)
    mask = np.isfinite(episodes) & np.isfinite(values)
    episodes = episodes[mask]
    values = values[mask]
    if clip_range is not None:
        lower, upper = clip_range
        if lower is None:
            lower = -np.inf
        if upper is None:
            upper = np.inf
        values = np.clip(values, lower, upper)
    return episodes, values


def _aggregate_seed_curves(
    seed_metrics: list[list[dict]],
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for metrics in seed_metrics:
        episodes, success_rate = _extract_curve(metrics)
        if episodes.size == 0:
            continue
        curves.append((episodes, success_rate))

    if not curves:
        return None

    max_common_episode = int(min(float(episodes[-1]) for episodes, _ in curves))
    if max_common_episode < 1:
        return None

    num_points = int(min(max_points, max_common_episode))
    num_points = max(1, num_points)
    x_grid = np.linspace(1.0, float(max_common_episode), num=num_points, dtype=np.float64)

    interpolated = np.vstack(
        [np.interp(x_grid, episodes, success_rate) for episodes, success_rate in curves]
    )
    mean_curve = interpolated.mean(axis=0)

    if interpolated.shape[0] > 1:
        std = interpolated.std(axis=0, ddof=1)
        sem = std / np.sqrt(float(interpolated.shape[0]))
        ci = 1.96 * sem
    else:
        ci = np.zeros_like(mean_curve)

    lower = np.clip(mean_curve - ci, 0.0, 1.0)
    upper = np.clip(mean_curve + ci, 0.0, 1.0)
    return x_grid, mean_curve, lower, upper, int(interpolated.shape[0])


def _aggregate_seed_metric_curves(
    seed_metrics: list[list[dict]],
    metric_key: str,
    max_points: int,
    clip_range: tuple[float | None, float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for metrics in seed_metrics:
        episodes, values = _extract_metric_curve(
            metrics=metrics, metric_key=metric_key, clip_range=clip_range
        )
        if episodes.size == 0:
            continue
        curves.append((episodes, values))

    if not curves:
        return None

    max_common_episode = int(min(float(episodes[-1]) for episodes, _ in curves))
    if max_common_episode < 1:
        return None

    num_points = int(min(max_points, max_common_episode))
    num_points = max(1, num_points)
    x_grid = np.linspace(1.0, float(max_common_episode), num=num_points, dtype=np.float64)

    interpolated = np.vstack([np.interp(x_grid, episodes, values) for episodes, values in curves])
    mean_curve = interpolated.mean(axis=0)

    if interpolated.shape[0] > 1:
        std = interpolated.std(axis=0, ddof=1)
        sem = std / np.sqrt(float(interpolated.shape[0]))
        ci = 1.96 * sem
    else:
        ci = np.zeros_like(mean_curve)

    lower = mean_curve - ci
    upper = mean_curve + ci
    return x_grid, mean_curve, lower, upper, int(interpolated.shape[0])


def _mean_shortest_path_length(seed_metrics: list[list[dict]]) -> float | None:
    per_seed_means: list[float] = []
    for metrics in seed_metrics:
        _, shortest_dist = _extract_metric_curve(
            metrics=metrics, metric_key="avg_shortest_dist", clip_range=(0.0, None)
        )
        if shortest_dist.size == 0:
            continue
        per_seed_means.append(float(np.mean(shortest_dist)))

    if not per_seed_means:
        return None
    return float(np.mean(np.asarray(per_seed_means, dtype=np.float64)))


def _plot_learning_curves_with_ci(
    series: list[tuple[str, list[list[dict]]]],
    output_path: Path,
    title: str,
    max_points: int,
    colors: list | None = None,
    max_x_episodes: int = MAX_X_EPISODES,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    max_episode = 0.0
    palette = colors if colors is not None else list(
        plt.cm.plasma(np.linspace(0.1, 0.9, max(1, len(series))))
    )

    plotted = 0
    for label, seed_metrics in series:
        aggregated = _aggregate_seed_curves(seed_metrics, max_points=max_points)
        if aggregated is None:
            continue

        x_grid, mean_curve, lower, upper, num_seeds = aggregated
        color = palette[plotted % len(palette)]
        ax.plot(
            x_grid,
            mean_curve,
            linewidth=2.2,
            color=color,
            label=f"{label}",
        )
        ax.fill_between(x_grid, lower, upper, color=color, alpha=0.18, linewidth=0.0)
        max_episode = max(max_episode, float(x_grid[-1]))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Success rate (running average)")
    ax.set_xlim(0.0, min(float(max_x_episodes), max(1.0, max_episode)))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_path_length_curves_with_ci(
    series: list[tuple[str, list[list[dict]]]],
    output_path: Path,
    title: str,
    max_points: int,
    colors: list | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    max_episode = 0.0
    palette = colors if colors is not None else list(
        plt.cm.plasma(np.linspace(0.1, 0.9, max(1, len(series))))
    )

    plotted = 0
    y_values: list[np.ndarray] = []
    y_lower_bounds: list[float] = []
    for label, seed_metrics in series:
        aggregated = _aggregate_seed_metric_curves(
            seed_metrics=seed_metrics,
            metric_key="avg_path_len",
            max_points=max_points,
            clip_range=(0.0, None),
        )
        if aggregated is None:
            continue

        x_grid, mean_curve, lower, upper, num_seeds = aggregated
        color = palette[plotted % len(palette)]
        ax.plot(
            x_grid,
            mean_curve,
            linewidth=2.2,
            color=color,
            label=f"{label}",
        )
        ax.fill_between(x_grid, lower, upper, color=color, alpha=0.18, linewidth=0.0)

        shortest_path_mean = _mean_shortest_path_length(seed_metrics)
        if shortest_path_mean is not None and np.isfinite(shortest_path_mean):
            ax.axhline(
                y=shortest_path_mean,
                color=color,
                linestyle=":",
                linewidth=1.8,
                alpha=0.95,
                label=f"{label} shortest-path mean",
            )
            y_lower_bounds.append(float(shortest_path_mean))

        y_values.extend([mean_curve, lower, upper])
        max_episode = max(max_episode, float(x_grid[-1]))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    finite_y = np.concatenate([arr[np.isfinite(arr)] for arr in y_values if arr.size > 0])
    finite_y = finite_y[finite_y >= 0.0]
    if finite_y.size > 0:
        y_min = float(np.min(finite_y))
        y_max = float(np.max(finite_y))
    else:
        y_min, y_max = 0.0, 1.0

    if y_lower_bounds:
        y_min = min(y_min, min(y_lower_bounds))
        y_max = max(y_max, max(y_lower_bounds))

    y_padding = max(0.5, 0.08 * (y_max - y_min))

    ax.set_title(title)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Bob path length")
    ax.set_xlim(0.0, min(float(MAX_X_EPISODES), max(1.0, max_episode)))
    ax.set_ylim(max(0.0, y_min - y_padding), y_max + y_padding)
    ax.grid(True, alpha=0.25)
    legend_columns = 2 if plotted > 3 else 1
    ax.legend(loc="upper right", frameon=False, ncol=legend_columns)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _aggregate_numeric(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _summarize_seed_records(seed_records: list[dict]) -> dict:
    per_seed: dict[str, dict] = {}
    final_success_values: list[float] = []
    episodes_to_80_values: list[float] = []

    for item in seed_records:
        seed = int(item["seed"])
        run_dir = _resolve_path(item["run_dir"])
        summary_path = run_dir / "summary.json"
        summary = _read_json(summary_path)
        per_seed[str(seed)] = summary

        final_success = summary.get("final_success_rate")
        if final_success is not None:
            final_success_values.append(float(final_success))

        episodes_to_80 = summary.get("episodes_to_80")
        if episodes_to_80 is not None:
            episodes_to_80_values.append(float(episodes_to_80))

    num_seeds = len(seed_records)
    reached_80 = len(episodes_to_80_values)
    return {
        "num_seeds": int(num_seeds),
        "seeds": [int(item["seed"]) for item in seed_records],
        "aggregate": {
            "final_success_rate": _aggregate_numeric(final_success_values),
            "episodes_to_80_reached_only": _aggregate_numeric(episodes_to_80_values),
            "fraction_reached_80": (float(reached_80) / float(num_seeds)) if num_seeds > 0 else None,
        },
        "per_seed": per_seed,
    }


def _load_seed_metrics(seed_records: list[dict]) -> list[list[dict]]:
    metrics_by_seed: list[list[dict]] = []
    for item in seed_records:
        run_dir = _resolve_path(item["run_dir"])
        metrics_path = run_dir / "metrics.jsonl"
        seed = item.get("seed")
        try:
            rows = _read_jsonl(metrics_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"Warning: skipping seed {seed} due to unreadable metrics file "
                f"'{metrics_path}': {exc}",
                file=sys.stderr,
            )
            continue
        metrics: list[dict] = []
        for row in rows:
            if "episodes" not in row or "success_rate" not in row:
                continue
            metric_row: dict[str, float | int] = {
                "episodes": int(row["episodes"]),
                "success_rate": float(row["success_rate"]),
            }
            if "avg_path_len" in row:
                metric_row["avg_path_len"] = float(row["avg_path_len"])
            if "avg_shortest_dist" in row:
                metric_row["avg_shortest_dist"] = float(row["avg_shortest_dist"])
            metrics.append(metric_row)
        metrics_by_seed.append(metrics)
    return metrics_by_seed


def _resolve_manifest_key(section: dict, *candidates: str) -> str | None:
    for key in candidates:
        if key in section:
            return key
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate requested experiment plots from an existing run manifest."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--plotdir", type=str, default=PLOT_DEFAULTS.plotdir)
    parser.add_argument("--max_points", type=int, default=PLOT_DEFAULTS.max_points)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _read_json(_resolve_path(args.manifest))
    cfg = manifest.get("config", {})

    sensing = str(cfg.get("sensing", REQUESTED_DEFAULTS.sensing))
    n = int(cfg.get("n", REQUESTED_DEFAULTS.n))
    running_avg_window = int(cfg.get("running_avg_window", REQUESTED_DEFAULTS.running_avg_window))
    seeds_used = list(cfg.get("seeds_used", []))
    num_seeds = int(cfg.get("num_seeds", len(seeds_used)))
    n_values = [int(v) for v in cfg.get("n_values", list(REQUESTED_DEFAULTS.n_values))]
    lie_probs = [float(v) for v in cfg.get("lie_probs", list(REQUESTED_DEFAULTS.lie_probs))]
    noise_sigmas = [float(v) for v in cfg.get("noise_sigmas", list(REQUESTED_DEFAULTS.noise_sigmas))]
    fst_k_values = [int(v) for v in cfg.get("fst_k_values", list(REQUESTED_DEFAULTS.fst_k_values))]
    d_values = [int(v) for v in cfg.get("d_values", list(REQUESTED_DEFAULTS.d_values))]
    binary_signal_sensing = str(
        cfg.get("binary_signal_sensing", REQUESTED_DEFAULTS.binary_signal_sensing)
    )

    plotdir = _resolve_path(args.plotdir)
    plotdir.mkdir(parents=True, exist_ok=True)

    exp1_manifest: dict = manifest["experiment_1"]
    exp1_conditions = [
        ("Permutation Cipher", ("fixed_cipher", "Permutation Cipher")),
        ("Fresh Cipher", ("fresh_cipher", "Fresh Cipher")),
        # ("Finite State Transducer", ("fst_cipher", "Finite State Transducer")),
        ("Random Message", ("random_message", "Random Message")),
        ("No Message", ("no_message", "No Message")),
    ]
    exp1_series: list[tuple[str, list[list[dict]]]] = []
    for display_label, key_candidates in exp1_conditions:
        manifest_key = _resolve_manifest_key(exp1_manifest, *key_candidates)
        if manifest_key is None:
            continue
        exp1_series.append((display_label, _load_seed_metrics(exp1_manifest[manifest_key])))
    _plot_learning_curves_with_ci(
        series=exp1_series,
        output_path=plotdir
        / f"exp1_encoding_comparison_{sensing}_n{n}_w{running_avg_window}_seeds{num_seeds}_ci95.png",
        title=(
            "Experiment 1: Superintelligence Language with Baselines"
        ),
        max_points=args.max_points,
    )
    _plot_path_length_curves_with_ci(
        series=exp1_series,
        output_path=plotdir
        / (
            f"exp1_encoding_path_length_{sensing}_n{n}_w{running_avg_window}_"
            f"seeds{num_seeds}_ci95.png"
        ),
        title=(
            "Experiment 1: Path Length Comparison"
        ),
        max_points=args.max_points,
    )
    exp1_summary: dict[str, dict] = {}
    for display_label, key_candidates in exp1_conditions:
        manifest_key = _resolve_manifest_key(exp1_manifest, *key_candidates)
        if manifest_key is None:
            continue
        exp1_summary[display_label] = _summarize_seed_records(exp1_manifest[manifest_key])
    with (
        plotdir / f"exp1_encoding_comparison_{sensing}_n{n}_summary_seeds{num_seeds}.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(exp1_summary, handle, indent=2)

    exp2_manifest: dict = manifest["experiment_2"]
    ordered_n = sorted(int(k) for k in exp2_manifest.keys()) if not n_values else sorted(n_values)
    exp2_series = [
        (f"n={value}", _load_seed_metrics(exp2_manifest[str(value)]))
        for value in ordered_n
        if str(value) in exp2_manifest
    ]
    viridis_colors = list(plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(exp2_series)))))
    _plot_learning_curves_with_ci(
        series=exp2_series,
        output_path=plotdir
        / (
            f"exp2_graph_scaling_{sensing}_w{running_avg_window}_"
            f"seeds{num_seeds}_ci95.png"
        ),
        title=(
            "Experiment 2: Graph-size Scaling"
        ),
        max_points=args.max_points,
        colors=viridis_colors,
        max_x_episodes=MAX_X_EPISODES_GRAPH_SCALING,
    )
    _plot_path_length_curves_with_ci(
        series=exp2_series,
        output_path=plotdir
        / (
            f"exp2_graph_scaling_path_length_{sensing}_w{running_avg_window}_"
            f"seeds{num_seeds}_ci95.png"
        ),
        title=(
            "Experiment 2: Graph-size Path Length "
        ),
        max_points=args.max_points,
        colors=viridis_colors,
    )
    exp2_summary = {
        str(value): _summarize_seed_records(exp2_manifest[str(value)])
        for value in ordered_n
        if str(value) in exp2_manifest
    }
    with (plotdir / f"exp2_graph_scaling_{sensing}_summary_seeds{num_seeds}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(exp2_summary, handle, indent=2)

    exp3_manifest: dict = manifest["experiment_3"]
    exp3_series: list[tuple[str, list[list[dict]]]] = []
    for p in lie_probs:
        display_label = f"P(lie)={p:g}"
        manifest_key = _resolve_manifest_key(exp3_manifest, f"p_lie={p:g}", display_label)
        if manifest_key is None:
            continue
        exp3_series.append((display_label, _load_seed_metrics(exp3_manifest[manifest_key])))
    _plot_learning_curves_with_ci(
        series=exp3_series,
        output_path=plotdir
        / (
            f"exp3_adversarial_lie_rate_{sensing}_n{n}_w{running_avg_window}_"
            f"seeds{num_seeds}_ci95.png"
        ),
        title=(
            "Experiment 3: Noisy Superintelligence"
        ),
        max_points=args.max_points,
    )
    _plot_path_length_curves_with_ci(
        series=exp3_series,
        output_path=plotdir
        / (
            f"exp3_adversarial_lie_rate_path_length_{sensing}_n{n}_w{running_avg_window}_"
            f"seeds{num_seeds}_ci95.png"
        ),
        title=(
            "Experiment 3: Noisy Superintelligence"
        ),
        max_points=args.max_points,
    )
    exp3_summary: dict[str, dict] = {}
    for p in lie_probs:
        display_label = f"P(lie)={p:g}"
        manifest_key = _resolve_manifest_key(exp3_manifest, f"p_lie={p:g}", display_label)
        if manifest_key is None:
            continue
        exp3_summary[display_label] = _summarize_seed_records(exp3_manifest[manifest_key])
    with (
        plotdir / f"exp3_adversarial_lie_rate_{sensing}_n{n}_summary_seeds{num_seeds}.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(exp3_summary, handle, indent=2)

    exp4_manifest: dict = manifest.get("experiment_4", {})
    exp4_series: list[tuple[str, list[list[dict]]]] = []
    for sigma in sorted(set(noise_sigmas)):
        display_label = f"Noisy Sensing (σ={sigma:g})"
        manifest_key = _resolve_manifest_key(
            exp4_manifest,
            f"sigma={sigma:g}",
            f"σ={sigma:g}",
            display_label,
        )
        if manifest_key is None:
            continue
        exp4_series.append((display_label, _load_seed_metrics(exp4_manifest[manifest_key])))
    binary_label = "Binary Signal (no noise)"
    binary_key = _resolve_manifest_key(exp4_manifest, "binary_no_noise", binary_label)
    if binary_key is not None:
        exp4_series.append(
            (
                binary_label,
                _load_seed_metrics(exp4_manifest[binary_key]),
            )
        )
    _plot_learning_curves_with_ci(
        series=exp4_series,
        output_path=plotdir
        / (
            f"exp4_noisy_sensing_{sensing}_n{n}_w{running_avg_window}_"
            f"seeds{num_seeds}_ci95.png"
        ),
        title=(
            "Experiment 4: Noisy Sensing and Binary Signal "
        ),
        max_points=args.max_points,
    )
    exp4_summary: dict[str, dict] = {}
    for sigma in sorted(set(noise_sigmas)):
        display_label = f"Noisy {sensing} (sigma={sigma:g})"
        manifest_key = _resolve_manifest_key(
            exp4_manifest,
            f"sigma={sigma:g}",
            f"σ={sigma:g}",
            display_label,
        )
        if manifest_key is None:
            continue
        exp4_summary[display_label] = _summarize_seed_records(exp4_manifest[manifest_key])
    if binary_key is not None:
        exp4_summary[binary_label] = _summarize_seed_records(exp4_manifest[binary_key])
    with (
        plotdir / f"exp4_noisy_sensing_{sensing}_n{n}_summary_seeds{num_seeds}.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(exp4_summary, handle, indent=2)

    exp5_manifest: dict = manifest.get("experiment_5", {})
    ordered_k = sorted(
        int(k[2:]) for k in exp5_manifest.keys() if k.startswith("k=") and k[2:].isdigit()
    )
    if not ordered_k:
        ordered_k = sorted(
            int(k.split(" ", 1)[0])
            for k in exp5_manifest.keys()
            if k.endswith(" Hidden States") and k.split(" ", 1)[0].isdigit()
        )
    if fst_k_values:
        ordered_k = sorted(set(int(v) for v in fst_k_values))
    exp5_series: list[tuple[str, list[list[dict]]]] = []
    for k in ordered_k:
        display_label = f"{k} Hidden States"
        manifest_key = _resolve_manifest_key(exp5_manifest, f"k={k}", display_label)
        if manifest_key is None:
            continue
        exp5_series.append((display_label, _load_seed_metrics(exp5_manifest[manifest_key])))
    plasma_colors = list(plt.cm.plasma(np.linspace(0.1, 0.9, max(1, len(exp5_series)))))
    _plot_learning_curves_with_ci(
        series=exp5_series,
        output_path=plotdir
        / (
            f"exp5_fst_k_sweep_{sensing}_n{n}_w{running_avg_window}_"
            f"seeds{num_seeds}_ci95.png"
        ),
        title=(
            "Experiment 5: Hidden State Size Sweep"
        ),
        max_points=args.max_points,
        colors=plasma_colors,
    )
    exp5_summary: dict[str, dict] = {}
    for k in ordered_k:
        display_label = f"{k} Hidden States"
        manifest_key = _resolve_manifest_key(exp5_manifest, f"k={k}", display_label)
        if manifest_key is None:
            continue
        exp5_summary[display_label] = _summarize_seed_records(exp5_manifest[manifest_key])
    with (plotdir / f"exp5_fst_k_sweep_{sensing}_n{n}_summary_seeds{num_seeds}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(exp5_summary, handle, indent=2)

    exp6_manifest: dict = manifest.get("experiment_6", {})
    ordered_d = sorted(
        int(k[2:]) for k in exp6_manifest.keys() if k.startswith("d=") and k[2:].isdigit()
    )
    if d_values:
        ordered_d = sorted(set(int(v) for v in d_values))
    exp6_series: list[tuple[str, list[list[dict]]]] = []
    for d in ordered_d:
        display_label = f"d={d}"
        manifest_key = _resolve_manifest_key(exp6_manifest, display_label)
        if manifest_key is None:
            continue
        exp6_series.append((display_label, _load_seed_metrics(exp6_manifest[manifest_key])))
    viridis_degree_colors = list(plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(exp6_series)))))
    _plot_learning_curves_with_ci(
        series=exp6_series,
        output_path=plotdir
        / (
            f"exp6_degree_sweep_{sensing}_n{n}_w{running_avg_window}_"
            f"seeds{num_seeds}_ci95.png"
        ),
        title=(
            "Experiment 6: Graph Degree Sweep"
        ),
        max_points=args.max_points,
        colors=viridis_degree_colors,
    )
    _plot_path_length_curves_with_ci(
        series=exp6_series,
        output_path=plotdir
        / (
            f"exp6_degree_sweep_path_length_{sensing}_n{n}_w{running_avg_window}_"
            f"seeds{num_seeds}_ci95.png"
        ),
        title=(
            "Experiment 6: Graph Degree Path Length"
        ),
        max_points=args.max_points,
        colors=viridis_degree_colors,
    )
    exp6_summary: dict[str, dict] = {}
    for d in ordered_d:
        display_label = f"d={d}"
        manifest_key = _resolve_manifest_key(exp6_manifest, display_label)
        if manifest_key is None:
            continue
        exp6_summary[display_label] = _summarize_seed_records(exp6_manifest[manifest_key])
    with (plotdir / f"exp6_degree_sweep_{sensing}_n{n}_summary_seeds{num_seeds}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(exp6_summary, handle, indent=2)

    summary = {
        "seeds_used": seeds_used,
        "experiment_1": exp1_summary,
        "experiment_2": exp2_summary,
        "experiment_3": exp3_summary,
        "experiment_4": exp4_summary,
        "experiment_5": exp5_summary,
        "experiment_6": exp6_summary,
        "plotdir": str(plotdir.resolve()),
        "manifest": str(_resolve_path(args.manifest)),
    }
    with (plotdir / f"requested_experiments_summary_seeds{num_seeds}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
