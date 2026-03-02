from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

MAX_X_EPISODES = 75_000



def _extract_series(metrics: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not metrics:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    episodes = np.asarray([float(m["episodes"]) for m in metrics], dtype=np.float64)
    if "success_rate" in metrics[0]:
        success = np.asarray([float(m["success_rate"]) for m in metrics], dtype=np.float64)
    elif "success" in metrics[0]:
        success = np.asarray([float(m["success"]) for m in metrics], dtype=np.float64)
    else:
        raise ValueError("Metrics must include either 'success_rate' or 'success'")

    finite_mask = np.isfinite(episodes) & np.isfinite(success)
    episodes = episodes[finite_mask]
    success = success[finite_mask]
    if episodes.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    order = np.argsort(episodes, kind="mergesort")
    return episodes[order], np.clip(success[order], 0.0, 1.0)


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
    ax.set_facecolor("#f6f8fb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d7de")
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(colors="#57606a")
    return fig, ax


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


def plot_success_curve(metrics: list[dict], output_path: Path, title: str) -> None:
    if not metrics:
        return

    episodes, success = _extract_series(metrics)
    if episodes.size == 0:
        return
    smooth_window = _choose_smoothing_window(int(episodes.size))
    smooth = _moving_average(success, smooth_window)
    spread = _rolling_std(success, smooth_window)
    raw_x, raw_y = _downsample(episodes, success, max_points=1800)

    fig, ax = _make_axes(figsize=(8.2, 4.8))
    color = "#1f6feb"

    ax.plot(raw_x, raw_y, linewidth=1.2, color=color, alpha=0.22, label="Raw")
    ax.plot(episodes, smooth, linewidth=2.6, color=color, label=f"Smoothed ({smooth_window}-pt)")
    if smooth_window > 1 and episodes.size >= 12:
        lower = np.clip(smooth - spread, 0.0, 1.0)
        upper = np.clip(smooth + spread, 0.0, 1.0)
        ax.fill_between(episodes, lower, upper, color=color, alpha=0.14, linewidth=0)

    ax.scatter([episodes[-1]], [smooth[-1]], color=color, s=24, zorder=5)
    _set_learning_axes(ax, float(episodes[-1]), [success, smooth])
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.legend(loc="lower right", frameon=False)
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
