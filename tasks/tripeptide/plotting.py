"""Plotting utilities for tripeptide experiments.

Adapted from alanine_dipeptide plotting — no 2D heatmaps since d=4.
Focus on objective vs step curves and summary bar charts.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


OPTIMIZATION_METHOD_ORDER = (
    "gd",
    "adam",
    "rl_no_oracle",
    "rl_visible_oracle",
    "rl_hidden_gradient",
)
OPTIMIZATION_METHOD_INDEX = {
    method_key: idx for idx, method_key in enumerate(OPTIMIZATION_METHOD_ORDER)
}


def _trailing_running_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    window = max(1, min(int(window), int(values.size)))
    cumulative = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    endpoints = np.arange(1, values.size + 1, dtype=np.int64)
    starts = np.maximum(0, endpoints - window)
    counts = endpoints - starts
    return (cumulative[endpoints] - cumulative[starts]) / counts


def _make_axes(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d7de")
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(colors="#57606a")
    ax.grid(True, which="major", alpha=0.22, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.09, linewidth=0.5)
    ax.minorticks_on()
    return fig, ax


def _spatial_optimization_method_color(method_key: str, fallback_index: int, total: int) -> tuple:
    palette = {
        "gd": "#cf222e",
        "adam": "#1a7f37",
        "rl_no_oracle": "#0969da",
        "rl_visible_oracle": "#8250df",
        "rl_hidden_gradient": "#111111",
    }
    if method_key in palette:
        return palette[method_key]
    cmap = plt.cm.plasma
    return cmap(float(fallback_index) / max(1, total - 1))


def plot_spatial_optimization_curves_by_method(
    method_curves: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    method_labels: dict[str, str] | None = None,
    y_label: str = "Normalized objective",
) -> None:
    valid_items: list[tuple[str, np.ndarray]] = []
    for method_key, curves in method_curves.items():
        arr = np.asarray(curves, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 2:
            continue
        valid_items.append((method_key, arr))
    if not valid_items:
        return

    num_methods = len(valid_items)
    n_cols = 3 if num_methods >= 3 else num_methods
    n_rows = int(np.ceil(num_methods / max(1, n_cols)))
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(4.9 * n_cols, 3.5 * n_rows),
        sharex=True, sharey=True,
    )
    axes_array = np.asarray(axes).reshape(-1)

    for axis in axes_array:
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#d0d7de")
        axis.spines["bottom"].set_color("#d0d7de")
        axis.tick_params(colors="#57606a")
        axis.grid(True, which="major", alpha=0.22, linewidth=0.8)
        axis.grid(True, which="minor", alpha=0.09, linewidth=0.5)
        axis.minorticks_on()
        axis.set_ylim(0.0, 1.02)

    for idx, (method_key, curves) in enumerate(valid_items):
        ax = axes_array[idx]
        label = method_labels.get(method_key, method_key) if method_labels is not None else method_key
        color = _spatial_optimization_method_color(method_key, idx, num_methods)
        steps = np.arange(curves.shape[1], dtype=np.int64)
        for row in curves:
            ax.plot(steps, row, color=color, linewidth=0.7, alpha=0.10)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        ax.plot(steps, mean_curve, color=color, linewidth=1.9, label=f"{label} mean")
        ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                         color=color, alpha=0.20, linewidth=0.0, label="\u00b11 std")
        ax.set_title(f"{label} (tasks={curves.shape[0]})", loc="left", fontsize=10, pad=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if idx == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8)

    for idx in range(num_methods, axes_array.size):
        axes_array[idx].set_visible(False)

    for idx, axis in enumerate(axes_array):
        if not axis.get_visible():
            continue
        if idx // max(1, n_cols) == n_rows - 1:
            axis.set_xlabel("Optimization step")
        if idx % max(1, n_cols) == 0:
            axis.set_ylabel(y_label)

    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_spatial_optimization_curve_summary(
    method_mean_curves: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    method_std_curves: dict[str, np.ndarray] | None = None,
    method_labels: dict[str, str] | None = None,
    y_label: str = "Normalized objective",
    x_label: str = "Optimization step",
) -> None:
    valid: list[tuple[str, np.ndarray]] = []
    for method_key, mean_curve in method_mean_curves.items():
        mean_arr = np.asarray(mean_curve, dtype=np.float64).reshape(-1)
        if mean_arr.size < 2:
            continue
        valid.append((method_key, mean_arr))
    if not valid:
        return

    fig, ax = _make_axes(figsize=(8.4, 4.9))
    smooth_window = 16
    for idx, (method_key, mean_arr) in enumerate(valid):
        color = _spatial_optimization_method_color(method_key, idx, len(valid))
        label = method_labels.get(method_key, method_key) if method_labels is not None else method_key
        steps = np.arange(mean_arr.size, dtype=np.int64)
        smoothed_mean_arr = _trailing_running_average(mean_arr, smooth_window)
        ax.plot(steps, mean_arr, color=color, linewidth=1.1, alpha=0.30, label="_nolegend_")
        ax.plot(steps, smoothed_mean_arr, color=color, linewidth=2.1, label=label)

    horizon = max(arr.size for _, arr in valid) - 1
    ax.set_xlim(0.0, max(1.0, float(horizon)))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _integer_hist_bins(values: np.ndarray, max_bins: int = 60) -> np.ndarray:
    if values.size == 0:
        return np.arange(2, dtype=np.float64)
    lo = int(np.floor(np.min(values)))
    hi = int(np.ceil(np.max(values)))
    span = max(1, hi - lo)
    step = max(1, span // max_bins)
    return np.arange(lo, hi + step + 1, step, dtype=np.float64)


def plot_path_length_histograms(
    path_lengths: list[float],
    success_path_lengths: list[float],
    failure_path_lengths: list[float],
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_arr = np.asarray(path_lengths, dtype=np.float64)
    success_arr = np.asarray(success_path_lengths, dtype=np.float64)
    failure_arr = np.asarray(failure_path_lengths, dtype=np.float64)

    if all_arr.size == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    fig.patch.set_facecolor("white")

    for ax, data, label, color in zip(
        axes,
        [all_arr, success_arr, failure_arr],
        ["All episodes", "Success episodes", "Failure episodes"],
        ["#0969da", "#1a7f37", "#cf222e"],
    ):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if data.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#57606a")
            ax.set_title(f"{title_prefix} | {label} (n=0)", loc="left", fontsize=10)
            continue
        bins = _integer_hist_bins(data, max_bins=40)
        ax.hist(data, bins=bins, color=color, alpha=0.78, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Path length")
        ax.set_ylabel("Count")
        ax.set_title(f"{title_prefix} | {label} (n={data.size})", loc="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "path_length_histograms.png", dpi=180)
    plt.close(fig)


def plot_spatial_trajectory_with_gradients(
    trajectory_xy: np.ndarray,
    gradient_xy: np.ndarray,
    move_vectors_xy: np.ndarray,
    target_xy: np.ndarray,
    baseline_trajectory_xy: np.ndarray | None,
    adam_baseline_trajectory_xy: np.ndarray | None,
    basin_hopping_trajectory_xy: np.ndarray | None,
    no_oracle_trajectory_xy: np.ndarray | None,
    visible_gradient_trajectory_xy: np.ndarray | None,
    output_path: Path,
    title: str,
    landscape_x: np.ndarray | None = None,
    landscape_y: np.ndarray | None = None,
    landscape_energy: np.ndarray | None = None,
    landscape_label: str = "E(z)",
) -> None:
    """Plot 2D projection of trajectory (first two angles)."""
    if trajectory_xy.ndim != 2 or trajectory_xy.shape[0] < 2:
        return

    d = trajectory_xy.shape[1]
    fig, ax = _make_axes(figsize=(7.2, 7.2))
    ax.set_facecolor("none")

    if (
        landscape_x is not None
        and landscape_y is not None
        and landscape_energy is not None
    ):
        lx_deg = np.degrees(landscape_x)
        ly_deg = np.degrees(landscape_y)
        levels = 28
        contour = ax.contourf(
            lx_deg, ly_deg, landscape_energy,
            levels=levels, cmap="YlOrRd", alpha=0.72, zorder=1,
        )
        fig.colorbar(contour, ax=ax, pad=0.01, fraction=0.045, label=landscape_label)

    def _to_deg(arr: np.ndarray) -> np.ndarray:
        return np.degrees(arr)

    # Project to first two dimensions
    x = _to_deg(trajectory_xy[:, 0])
    y = _to_deg(trajectory_xy[:, 1])
    ax.plot(x, y, color="#111111", linewidth=1.5, alpha=0.92, zorder=4,
            marker="o", markersize=2.3, markerfacecolor="#111111",
            markeredgewidth=0.0, label="PPO hidden-gradient oracle")
    ax.scatter([x[0]], [y[0]], color="#fb8500", edgecolor="black",
               linewidth=0.5, s=54, zorder=6, label="Start")
    target_deg = _to_deg(target_xy[:2] if target_xy.size >= 2 else target_xy)
    ax.scatter([target_deg[0]], [target_deg[1]], color="#1f883d", edgecolor="black",
               linewidth=0.7, s=78, zorder=7, label="Global min")

    def _plot_traj(traj, color, label):
        if traj is not None and traj.ndim == 2 and traj.shape[0] >= 2:
            tx = _to_deg(traj[:, 0])
            ty = _to_deg(traj[:, 1])
            ax.plot(tx, ty, color=color, linestyle="-", linewidth=1.5, alpha=0.92,
                    zorder=6, marker="o", markersize=2.3, markerfacecolor=color,
                    markeredgewidth=0.0, label=label)

    _plot_traj(baseline_trajectory_xy, "#cf222e", "GD baseline")
    _plot_traj(adam_baseline_trajectory_xy, "green", "Adam baseline")
    _plot_traj(no_oracle_trajectory_xy, "#0969da", "PPO no oracle")
    _plot_traj(visible_gradient_trajectory_xy, "limegreen", "PPO visible-gradient oracle")

    ax.set_xlabel("phi1 (degrees)")
    ax.set_ylabel("psi1 (degrees)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_learning_curves(
    episodes: list[int],
    success_rates: list[float],
    avg_rewards: list[float],
    avg_objectives: list[float],
    output_path: Path,
    title_prefix: str = "",
    window: int = 100,
) -> None:
    """Plot 3-panel learning curves: success rate, episode reward, final objective."""
    if len(episodes) < 2:
        return

    ep = np.asarray(episodes, dtype=np.int64)
    sr = np.asarray(success_rates, dtype=np.float64)
    rew = np.asarray(avg_rewards, dtype=np.float64)
    obj = np.asarray(avg_objectives, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    fig.patch.set_facecolor("white")

    panels = [
        (sr, "Success rate", "#1a7f37", (0.0, 1.05)),
        (rew, "Episode reward", "#0969da", None),
        (obj, "Final objective (norm.)", "#cf222e", (0.0, 1.05)),
    ]

    for ax, (values, ylabel, color, ylim) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d0d7de")
        ax.spines["bottom"].set_color("#d0d7de")
        ax.tick_params(colors="#57606a")
        ax.grid(True, which="major", alpha=0.22, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.09, linewidth=0.5)
        ax.minorticks_on()

        smoothed = _trailing_running_average(values, window)
        ax.plot(ep, values, color=color, linewidth=0.5, alpha=0.25)
        ax.plot(ep, smoothed, color=color, linewidth=1.8, label=f"window={window}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.legend(loc="best", frameon=False, fontsize=8)

    fig.suptitle(f"{title_prefix} Learning curves" if title_prefix else "Learning curves",
                 x=0.01, y=0.995, ha="left", va="top", fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cumulative_success_by_step(
    success_steps: list[int],
    total_episodes: int,
    max_horizon: int,
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """Plot cumulative success rate as a function of optimization step.

    For each step t in [1, max_horizon], shows the fraction of all completed
    episodes that achieved success at or before step t.
    """
    if total_episodes == 0:
        return

    steps_arr = np.asarray(success_steps, dtype=np.int64)
    t = np.arange(1, max_horizon + 1)
    cum_success = np.array(
        [float(np.sum(steps_arr <= step)) / total_episodes for step in t]
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d7de")
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(colors="#57606a")
    ax.grid(True, which="major", alpha=0.22, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.09, linewidth=0.5)
    ax.minorticks_on()

    ax.plot(t, cum_success, color="#0969da", linewidth=1.8)
    ax.fill_between(t, 0, cum_success, color="#0969da", alpha=0.12)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Cumulative success rate")
    ax.set_xlim(1, max_horizon)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    title = f"{title_prefix} Cumulative success by step" if title_prefix else "Cumulative success by step"
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
