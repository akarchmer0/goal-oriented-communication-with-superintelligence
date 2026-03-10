from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

MAX_X_EPISODES = 75_000



def _extract_metric_series(
    metrics: list[dict],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not metrics:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    episodes = np.asarray([float(m["episodes"]) for m in metrics], dtype=np.float64)
    values = np.asarray([float(m[key]) for m in metrics], dtype=np.float64)

    finite_mask = np.isfinite(episodes) & np.isfinite(values)
    episodes = episodes[finite_mask]
    values = values[finite_mask]
    if episodes.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    order = np.argsort(episodes, kind="mergesort")
    return episodes[order], values[order]


def _extract_success_series(metrics: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not metrics:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty, empty

    episodes = np.asarray([float(m["episodes"]) for m in metrics], dtype=np.float64)
    if "success" not in metrics[0] and "success_rate" not in metrics[0]:
        raise ValueError("Metrics must include either 'success_rate' or 'success'")

    success = np.asarray(
        [float(m["success"]) if "success" in m else np.nan for m in metrics],
        dtype=np.float64,
    )
    success_rate = np.asarray(
        [float(m["success_rate"]) if "success_rate" in m else np.nan for m in metrics],
        dtype=np.float64,
    )

    finite_mask = np.isfinite(episodes) & (np.isfinite(success) | np.isfinite(success_rate))
    episodes = episodes[finite_mask]
    success = success[finite_mask]
    success_rate = success_rate[finite_mask]
    if episodes.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    order = np.argsort(episodes, kind="mergesort")
    episodes = episodes[order]
    success = np.clip(success[order], 0.0, 1.0)
    success_rate = np.clip(success_rate[order], 0.0, 1.0)
    return episodes, success, success_rate


def _trailing_running_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()

    window = max(1, min(int(window), int(values.size)))
    cumulative = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    endpoints = np.arange(1, values.size + 1, dtype=np.int64)
    starts = np.maximum(0, endpoints - window)
    counts = endpoints - starts
    return (cumulative[endpoints] - cumulative[starts]) / counts


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()

    window = max(1, min(int(window), int(values.size)))
    if window % 2 == 0:
        window = max(1, window - 1)
    if window == 1:
        return values.copy()

    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded = np.pad(values, (left_pad, right_pad), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if window <= 1:
        return np.zeros_like(values)

    mean = _moving_average(values, window)
    mean_sq = _moving_average(values * values, window)
    variance = np.clip(mean_sq - (mean * mean), 0.0, None)
    return np.sqrt(variance)


def _choose_smoothing_window(num_points: int) -> int:
    if num_points < 10:
        return 1
    window = int(np.clip(num_points // 35, 5, 101))
    if window % 2 == 0:
        window += 1
    if window > num_points:
        window = num_points if num_points % 2 == 1 else max(1, num_points - 1)
    return max(1, window)


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    indices = np.linspace(0, x.size - 1, num=max_points, dtype=np.int64)
    indices = np.unique(indices)
    return x[indices], y[indices]


def _make_axes(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d7de")
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(colors="#57606a")
    return fig, ax


def _plasma_series_colors(num_series: int) -> list:
    return list(plt.cm.plasma(np.linspace(0.12, 0.88, max(1, num_series))))


def _set_learning_axes(ax: plt.Axes, max_episode: float, success_arrays: list[np.ndarray]) -> None:
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Success rate")
    ax.set_xlim(0.0, min(float(MAX_X_EPISODES), max(1.0, max_episode)))

    finite_chunks = [arr[np.isfinite(arr)] for arr in success_arrays if arr.size > 0]
    finite_chunks = [chunk for chunk in finite_chunks if chunk.size > 0]
    finite_values = np.concatenate(finite_chunks) if finite_chunks else np.asarray([], dtype=np.float64)
    max_success = float(np.max(finite_values)) if finite_values.size > 0 else 0.0
    if max_success <= 0.15:
        y_upper = max(0.04, max_success * 1.45 + 0.01)
        ax.set_ylim(0.0, y_upper)
    else:
        ax.set_ylim(0.0, 1.02)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, which="major", alpha=0.26, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)
    ax.minorticks_on()


def plot_success_curve(
    metrics: list[dict],
    output_path: Path,
    title: str,
    running_avg_window: int = 100,
) -> None:
    if not metrics:
        return

    episodes, success, success_rate = _extract_success_series(metrics)
    if episodes.size == 0:
        return
    raw_mask = np.isfinite(success)
    running_mask = np.isfinite(success_rate)
    if np.any(raw_mask):
        raw_x = episodes[raw_mask]
        raw_y = success[raw_mask]
        smooth = _trailing_running_average(raw_y, running_avg_window)
        smooth_x = raw_x
    elif np.any(running_mask):
        raw_x = episodes[running_mask]
        raw_y = success_rate[running_mask]
        smooth = raw_y.copy()
        smooth_x = raw_x
    else:
        return

    fig, ax = _make_axes(figsize=(8.2, 4.8))
    color = "#1f6feb"

    ax.plot(raw_x, raw_y, linewidth=1.0, color=color, alpha=0.18, label="Per-episode")
    ax.plot(
        smooth_x,
        smooth,
        linewidth=1.5,
        color=color,
        label=f"Running avg ({int(max(1, running_avg_window))}-ep)",
    )

    ax.scatter([smooth_x[-1]], [smooth[-1]], color=color, s=24, zorder=5)
    _set_learning_axes(ax, float(max(raw_x[-1], smooth_x[-1])), [raw_y, smooth])
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_objective_curve(
    metrics: list[dict],
    output_path: Path,
    title: str,
    value_key: str = "objective_value",
    y_label: str = "E(F(z))",
    main_label: str = "PPO with oracle",
    baseline_value_key: str | None = None,
    baseline_label: str = "Visible GD",
    secondary_baseline_value_key: str | None = None,
    secondary_baseline_label: str = "Visible SGD",
    comparison_value_key: str | None = None,
    comparison_label: str = "PPO (no oracle)",
    y_axis_formatter: str | None = None,
) -> None:
    if not metrics:
        return

    episodes, values = _extract_metric_series(metrics, key=value_key)
    if episodes.size == 0:
        return

    smooth_window = _choose_smoothing_window(int(episodes.size))
    smooth = _moving_average(values, smooth_window)
    spread = _rolling_std(values, smooth_window)
    raw_x, raw_y = _downsample(episodes, values, max_points=1800)

    baseline_episodes = None
    baseline_values = None
    baseline_smooth = None
    baseline_spread = None
    baseline_raw_x = None
    baseline_raw_y = None
    if baseline_value_key is not None:
        be, bv = _extract_metric_series(metrics, key=baseline_value_key)
        if be.size > 0:
            baseline_episodes = be
            baseline_values = bv
            baseline_smooth = _moving_average(bv, smooth_window)
            baseline_spread = _rolling_std(bv, smooth_window)
            baseline_raw_x, baseline_raw_y = _downsample(be, bv, max_points=1800)

    secondary_baseline_episodes = None
    secondary_baseline_values = None
    secondary_baseline_smooth = None
    secondary_baseline_spread = None
    secondary_baseline_raw_x = None
    secondary_baseline_raw_y = None
    if secondary_baseline_value_key is not None:
        sbe, sbv = _extract_metric_series(metrics, key=secondary_baseline_value_key)
        if sbe.size > 0:
            secondary_baseline_episodes = sbe
            secondary_baseline_values = sbv
            secondary_baseline_smooth = _moving_average(sbv, smooth_window)
            secondary_baseline_spread = _rolling_std(sbv, smooth_window)
            secondary_baseline_raw_x, secondary_baseline_raw_y = _downsample(
                sbe, sbv, max_points=1800
            )

    comparison_episodes = None
    comparison_values = None
    comparison_smooth = None
    comparison_spread = None
    comparison_raw_x = None
    comparison_raw_y = None
    if comparison_value_key is not None:
        ce, cv = _extract_metric_series(metrics, key=comparison_value_key)
        if ce.size > 0:
            comparison_episodes = ce
            comparison_values = cv
            comparison_smooth = _moving_average(cv, smooth_window)
            comparison_spread = _rolling_std(cv, smooth_window)
            comparison_raw_x, comparison_raw_y = _downsample(ce, cv, max_points=1800)

    fig, ax = _make_axes(figsize=(8.2, 4.8))
    color, baseline_color, secondary_baseline_color, comparison_color = _plasma_series_colors(
        num_series=4
    )

    ax.plot(raw_x, raw_y, linewidth=1.1, color=color, alpha=0.22, label="_nolegend_")
    ax.plot(episodes, smooth, linewidth=1.5, color=color, label=main_label)
    if smooth_window > 1 and episodes.size >= 12:
        lower = smooth - spread
        upper = smooth + spread
        ax.fill_between(episodes, lower, upper, color=color, alpha=0.14, linewidth=0)

    ax.scatter([episodes[-1]], [smooth[-1]], color=color, s=28, zorder=5)
    if baseline_episodes is not None and baseline_values is not None and baseline_smooth is not None:
        assert baseline_raw_x is not None and baseline_raw_y is not None
        ax.plot(
            baseline_raw_x,
            baseline_raw_y,
            linewidth=1.0,
            color=baseline_color,
            alpha=0.20,
            label="_nolegend_",
        )
        ax.plot(
            baseline_episodes,
            baseline_smooth,
            linewidth=1.4,
            color=baseline_color,
            label=baseline_label,
        )
        if baseline_spread is not None and baseline_episodes.size >= 12:
            lower_b = baseline_smooth - baseline_spread
            upper_b = baseline_smooth + baseline_spread
            ax.fill_between(
                baseline_episodes,
                lower_b,
                upper_b,
                color=baseline_color,
                alpha=0.10,
                linewidth=0,
            )
    if (
        secondary_baseline_episodes is not None
        and secondary_baseline_values is not None
        and secondary_baseline_smooth is not None
    ):
        assert secondary_baseline_raw_x is not None and secondary_baseline_raw_y is not None
        ax.plot(
            secondary_baseline_raw_x,
            secondary_baseline_raw_y,
            linewidth=1.0,
            color=secondary_baseline_color,
            alpha=0.20,
            label="_nolegend_",
        )
        ax.plot(
            secondary_baseline_episodes,
            secondary_baseline_smooth,
            linewidth=1.3,
            color=secondary_baseline_color,
            label=secondary_baseline_label,
        )
        if secondary_baseline_spread is not None and secondary_baseline_episodes.size >= 12:
            lower_sb = secondary_baseline_smooth - secondary_baseline_spread
            upper_sb = secondary_baseline_smooth + secondary_baseline_spread
            ax.fill_between(
                secondary_baseline_episodes,
                lower_sb,
                upper_sb,
                color=secondary_baseline_color,
                alpha=0.09,
                linewidth=0,
            )
    if (
        comparison_episodes is not None
        and comparison_values is not None
        and comparison_smooth is not None
    ):
        assert comparison_raw_x is not None and comparison_raw_y is not None
        ax.plot(
            comparison_raw_x,
            comparison_raw_y,
            linewidth=1.0,
            color=comparison_color,
            alpha=0.18,
            label="_nolegend_",
        )
        ax.plot(
            comparison_episodes,
            comparison_smooth,
            linewidth=1.3,
            color=comparison_color,
            label=comparison_label,
        )
        if comparison_spread is not None and comparison_episodes.size >= 12:
            lower_c = comparison_smooth - comparison_spread
            upper_c = comparison_smooth + comparison_spread
            ax.fill_between(
                comparison_episodes,
                lower_c,
                upper_c,
                color=comparison_color,
                alpha=0.08,
                linewidth=0,
            )
    ax.set_xlabel("Training episodes")
    ax.set_ylabel(y_label)
    ax.set_xlim(0.0, min(float(MAX_X_EPISODES), max(1.0, float(episodes[-1]))))

    finite_values = values[np.isfinite(values)]
    if baseline_values is not None:
        finite_values = np.concatenate([finite_values, baseline_values[np.isfinite(baseline_values)]])
    if secondary_baseline_values is not None:
        finite_values = np.concatenate(
            [finite_values, secondary_baseline_values[np.isfinite(secondary_baseline_values)]]
        )
    if comparison_values is not None:
        finite_values = np.concatenate(
            [finite_values, comparison_values[np.isfinite(comparison_values)]]
        )
    if finite_values.size > 0:
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        if y_axis_formatter == "percent":
            ax.set_ylim(0.0, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        else:
            margin = 0.08 * max(1e-6, y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, which="major", alpha=0.26, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)
    ax.minorticks_on()
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _align_series_to_grid(
    seed_metrics: list[tuple[int, list[dict]]],
    key: str,
    max_episodes: float | None = None,
    n_grid: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]:
    """Align series from multiple seeds onto a common episode grid. Returns (grid, mean, std, [(seed, values), ...])."""
    if not seed_metrics:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            [],
        )
    all_episodes: list[np.ndarray] = []
    for _, metrics in seed_metrics:
        ep, _ = _extract_metric_series(metrics, key=key)
        if ep.size > 0:
            all_episodes.append(ep)
    if not all_episodes:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            [],
        )
    max_ep = max(float(np.max(ep)) for ep in all_episodes)
    if max_episodes is not None:
        max_ep = min(max_ep, float(max_episodes))
    grid = np.linspace(0.0, max_ep, num=n_grid, dtype=np.float64)
    aligned: list[tuple[int, np.ndarray]] = []
    for seed, metrics in seed_metrics:
        ep, vals = _extract_metric_series(metrics, key=key)
        if ep.size > 0:
            interp = np.interp(grid, ep, vals, left=np.nan, right=np.nan)
            aligned.append((seed, interp))
    if not aligned:
        return grid, np.full(0, np.nan), np.full(0, np.nan), []
    stacked = np.stack([v for _, v in aligned], axis=0)
    with np.errstate(invalid="ignore"):
        mean_vals = np.nanmean(stacked, axis=0)
        std_vals = np.nanstd(stacked, axis=0)
    return grid, mean_vals, std_vals, aligned


def plot_objective_curve_multi_seed(
    seed_metrics: list[tuple[int, list[dict]]],
    output_path: Path,
    title: str,
    value_key: str = "objective_value",
    y_label: str = "E(F(z))",
    baseline_value_key: str | None = None,
    baseline_label: str = "Visible GD",
    secondary_baseline_value_key: str | None = None,
    secondary_baseline_label: str = "Visible SGD",
    comparison_value_key: str | None = None,
    comparison_label: str = "PPO (no oracle)",
    y_axis_formatter: str | None = None,
) -> None:
    """Plot objective-style curves for multiple seeds: individual (faded) + mean (bold) with std band."""
    if not seed_metrics:
        return

    fig, ax = _make_axes(figsize=(8.2, 4.8))
    color, baseline_color, secondary_baseline_color, comparison_color = _plasma_series_colors(
        num_series=4
    )

    grid, mean_vals, std_vals, aligned = _align_series_to_grid(
        seed_metrics, key=value_key, max_episodes=MAX_X_EPISODES
    )
    if grid.size == 0 or not np.any(np.isfinite(mean_vals)):
        plt.close(fig)
        return
    for _, vals in aligned:
        ax.plot(grid, vals, color=color, linewidth=0.8, alpha=0.25)
    ax.plot(grid, mean_vals, color=color, linewidth=1.8, label="PPO with oracle")
    ax.fill_between(
        grid,
        mean_vals - std_vals,
        mean_vals + std_vals,
        color=color,
        alpha=0.25,
        linewidth=0,
    )

    if baseline_value_key is not None:
        b_grid, b_mean, b_std, b_aligned = _align_series_to_grid(
            seed_metrics, key=baseline_value_key, max_episodes=MAX_X_EPISODES
        )
        if b_grid.size > 0 and np.any(np.isfinite(b_mean)):
            for _, vals in b_aligned:
                ax.plot(b_grid, vals, color=baseline_color, linewidth=0.6, alpha=0.2)
            ax.plot(b_grid, b_mean, color=baseline_color, linewidth=1.4, label=baseline_label)
            ax.fill_between(b_grid, b_mean - b_std, b_mean + b_std, color=baseline_color, alpha=0.15, linewidth=0)

    if secondary_baseline_value_key is not None:
        sb_grid, sb_mean, sb_std, sb_aligned = _align_series_to_grid(
            seed_metrics, key=secondary_baseline_value_key, max_episodes=MAX_X_EPISODES
        )
        if sb_grid.size > 0 and np.any(np.isfinite(sb_mean)):
            for _, vals in sb_aligned:
                ax.plot(sb_grid, vals, color=secondary_baseline_color, linewidth=0.6, alpha=0.2)
            ax.plot(sb_grid, sb_mean, color=secondary_baseline_color, linewidth=1.4, label=secondary_baseline_label)
            ax.fill_between(sb_grid, sb_mean - sb_std, sb_mean + sb_std, color=secondary_baseline_color, alpha=0.12, linewidth=0)

    if comparison_value_key is not None:
        c_grid, c_mean, c_std, c_aligned = _align_series_to_grid(
            seed_metrics, key=comparison_value_key, max_episodes=MAX_X_EPISODES
        )
        if c_grid.size > 0 and np.any(np.isfinite(c_mean)):
            for _, vals in c_aligned:
                ax.plot(c_grid, vals, color=comparison_color, linewidth=0.6, alpha=0.2)
            ax.plot(c_grid, c_mean, color=comparison_color, linewidth=1.4, label=comparison_label)
            ax.fill_between(c_grid, c_mean - c_std, c_mean + c_std, color=comparison_color, alpha=0.10, linewidth=0)

    ax.set_xlabel("Training episodes")
    ax.set_ylabel(y_label)
    ax.set_xlim(0.0, min(float(MAX_X_EPISODES), float(grid[-1])))
    finite = mean_vals[np.isfinite(mean_vals)]
    if finite.size > 0:
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        if y_axis_formatter == "percent":
            ax.set_ylim(0.0, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        else:
            margin = 0.08 * max(1e-6, y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, which="major", alpha=0.26, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)
    ax.minorticks_on()
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _integer_hist_bins(values: np.ndarray, max_bins: int = 60) -> np.ndarray:
    if values.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)

    low = int(np.floor(np.min(values)))
    high = int(np.ceil(np.max(values)))
    span = max(1, high - low + 1)
    if span <= max_bins:
        return np.arange(low - 0.5, high + 1.5, 1.0, dtype=np.float64)
    return np.linspace(low - 0.5, high + 0.5, num=max_bins + 1, dtype=np.float64)


def plot_path_length_histograms(
    path_lengths: list[float],
    success_path_lengths: list[float],
    failure_path_lengths: list[float],
    output_dir: Path,
    title_prefix: str,
) -> None:
    if not path_lengths:
        return

    all_values = np.asarray(path_lengths, dtype=np.float64)
    all_values = all_values[np.isfinite(all_values) & (all_values >= 0.0)]
    if all_values.size == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    bins = _integer_hist_bins(all_values)

    fig, ax = _make_axes(figsize=(8.2, 4.8))
    ax.hist(all_values, bins=bins, color="#1f6feb", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axvline(float(np.mean(all_values)), color="#cf222e", linestyle="--", linewidth=1.8, label="Mean")
    ax.set_xlabel("Episode path length")
    ax.set_ylabel("Episode count")
    ax.grid(True, axis="y", alpha=0.24)
    ax.set_title(f"{title_prefix} Path Length Histogram", loc="left", fontsize=11, pad=10)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "path_length_histogram.png", dpi=180)
    plt.close(fig)

    success_values = np.asarray(success_path_lengths, dtype=np.float64)
    success_values = success_values[np.isfinite(success_values) & (success_values >= 0.0)]
    failure_values = np.asarray(failure_path_lengths, dtype=np.float64)
    failure_values = failure_values[np.isfinite(failure_values) & (failure_values >= 0.0)]
    if success_values.size == 0 and failure_values.size == 0:
        return

    fig, ax = _make_axes(figsize=(8.2, 4.8))
    plotted_labels: list[str] = []
    if failure_values.size > 0:
        ax.hist(
            failure_values,
            bins=bins,
            color="#cf222e",
            alpha=0.55,
            edgecolor="white",
            linewidth=0.5,
            label="Failure",
        )
        plotted_labels.append("Failure")
    if success_values.size > 0:
        ax.hist(
            success_values,
            bins=bins,
            color="#2da44e",
            alpha=0.60,
            edgecolor="white",
            linewidth=0.5,
            label="Success",
        )
        plotted_labels.append("Success")

    ax.set_xlabel("Episode path length")
    ax.set_ylabel("Episode count")
    ax.grid(True, axis="y", alpha=0.24)
    ax.set_title(f"{title_prefix} Path Length Histogram by Outcome", loc="left", fontsize=11, pad=10)
    if plotted_labels:
        ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "path_length_histogram_by_outcome.png", dpi=180)
    plt.close(fig)


def plot_spatial_trajectory_with_gradients(
    trajectory_xy: np.ndarray,
    gradient_xy: np.ndarray,
    move_vectors_xy: np.ndarray,
    target_xy: np.ndarray,
    baseline_trajectory_xy: np.ndarray | None,
    sgd_baseline_trajectory_xy: np.ndarray | None,
    no_oracle_trajectory_xy: np.ndarray | None,
    output_path: Path,
    title: str,
    landscape_x: np.ndarray | None = None,
    landscape_y: np.ndarray | None = None,
    landscape_energy: np.ndarray | None = None,
) -> None:
    if trajectory_xy.ndim != 2 or trajectory_xy.shape[1] != 2 or trajectory_xy.shape[0] < 2:
        return

    # Intentionally hidden in final figure: keep signature stable for call sites.
    _ = gradient_xy
    _ = move_vectors_xy

    fig, ax = _make_axes(figsize=(7.2, 7.2))
    ax.set_facecolor("none")

    if (
        landscape_x is not None
        and landscape_y is not None
        and landscape_energy is not None
        and landscape_x.shape == landscape_y.shape == landscape_energy.shape
    ):
        levels = 28
        contour = ax.contourf(
            landscape_x,
            landscape_y,
            landscape_energy,
            levels=levels,
            cmap="YlOrRd",
            alpha=0.72,
            zorder=1,
        )
        fig.colorbar(contour, ax=ax, pad=0.01, fraction=0.045, label="E(F(z))")

    x = trajectory_xy[:, 0]
    y = trajectory_xy[:, 1]
    ax.plot(
        x,
        y,
        color="#111111",
        linewidth=1.5,
        alpha=0.92,
        zorder=4,
        marker="o",
        markersize=2.3,
        markerfacecolor="#111111",
        markeredgewidth=0.0,
        label="Trajectory",
    )
    ax.scatter([x[0]], [y[0]], color="#fb8500", edgecolor="black", linewidth=0.5, s=54, zorder=6, label="Start")
    ax.scatter(
        [target_xy[0]],
        [target_xy[1]],
        color="#1f883d",
        edgecolor="black",
        linewidth=0.7,
        s=78,
        zorder=7,
        label="Reference min",
    )
    if (
        baseline_trajectory_xy is not None
        and baseline_trajectory_xy.ndim == 2
        and baseline_trajectory_xy.shape[1] == 2
        and baseline_trajectory_xy.shape[0] >= 2
    ):
        bx = baseline_trajectory_xy[:, 0]
        by = baseline_trajectory_xy[:, 1]
        ax.plot(
            bx,
            by,
            color="#cf222e",
            linestyle="-",
            linewidth=1.5,
            alpha=0.92,
            zorder=6,
            marker="o",
            markersize=2.3,
            markerfacecolor="#cf222e",
            markeredgewidth=0.0,
            label="2D grad-descent baseline",
        )
    if (
        sgd_baseline_trajectory_xy is not None
        and sgd_baseline_trajectory_xy.ndim == 2
        and sgd_baseline_trajectory_xy.shape[1] == 2
        and sgd_baseline_trajectory_xy.shape[0] >= 2
    ):
        sx = sgd_baseline_trajectory_xy[:, 0]
        sy = sgd_baseline_trajectory_xy[:, 1]
        ax.plot(
            sx,
            sy,
            color="#8250df",
            linestyle="-",
            linewidth=1.5,
            alpha=0.90,
            zorder=6,
            marker="o",
            markersize=2.3,
            markerfacecolor="#8250df",
            markeredgewidth=0.0,
            label="2D SGD baseline",
        )
    if (
        no_oracle_trajectory_xy is not None
        and no_oracle_trajectory_xy.ndim == 2
        and no_oracle_trajectory_xy.shape[1] == 2
        and no_oracle_trajectory_xy.shape[0] >= 2
    ):
        nx = no_oracle_trajectory_xy[:, 0]
        ny = no_oracle_trajectory_xy[:, 1]
        ax.plot(
            nx,
            ny,
            color="#0969da",
            linestyle="-",
            linewidth=1.5,
            alpha=0.92,
            zorder=6,
            marker="o",
            markersize=2.3,
            markerfacecolor="#0969da",
            markeredgewidth=0.0,
            label="PPO no-oracle baseline",
        )

    x_min = min(float(np.min(x)), float(target_xy[0]))
    x_max = max(float(np.max(x)), float(target_xy[0]))
    y_min = min(float(np.min(y)), float(target_xy[1]))
    y_max = max(float(np.max(y)), float(target_xy[1]))
    if (
        baseline_trajectory_xy is not None
        and baseline_trajectory_xy.ndim == 2
        and baseline_trajectory_xy.shape[1] == 2
        and baseline_trajectory_xy.shape[0] >= 1
    ):
        x_min = min(x_min, float(np.min(baseline_trajectory_xy[:, 0])))
        x_max = max(x_max, float(np.max(baseline_trajectory_xy[:, 0])))
        y_min = min(y_min, float(np.min(baseline_trajectory_xy[:, 1])))
        y_max = max(y_max, float(np.max(baseline_trajectory_xy[:, 1])))
    if (
        sgd_baseline_trajectory_xy is not None
        and sgd_baseline_trajectory_xy.ndim == 2
        and sgd_baseline_trajectory_xy.shape[1] == 2
        and sgd_baseline_trajectory_xy.shape[0] >= 1
    ):
        x_min = min(x_min, float(np.min(sgd_baseline_trajectory_xy[:, 0])))
        x_max = max(x_max, float(np.max(sgd_baseline_trajectory_xy[:, 0])))
        y_min = min(y_min, float(np.min(sgd_baseline_trajectory_xy[:, 1])))
        y_max = max(y_max, float(np.max(sgd_baseline_trajectory_xy[:, 1])))
    if (
        no_oracle_trajectory_xy is not None
        and no_oracle_trajectory_xy.ndim == 2
        and no_oracle_trajectory_xy.shape[1] == 2
        and no_oracle_trajectory_xy.shape[0] >= 1
    ):
        x_min = min(x_min, float(np.min(no_oracle_trajectory_xy[:, 0])))
        x_max = max(x_max, float(np.max(no_oracle_trajectory_xy[:, 0])))
        y_min = min(y_min, float(np.min(no_oracle_trajectory_xy[:, 1])))
        y_max = max(y_max, float(np.max(no_oracle_trajectory_xy[:, 1])))
    if landscape_x is not None and landscape_y is not None:
        x_min = min(x_min, float(np.min(landscape_x)))
        x_max = max(x_max, float(np.max(landscape_x)))
        y_min = min(y_min, float(np.min(landscape_y)))
        y_max = max(y_max, float(np.max(landscape_y)))
    margin = 0.0 if (landscape_x is not None and landscape_y is not None) else 0.35
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
