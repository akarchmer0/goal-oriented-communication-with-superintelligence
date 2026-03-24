from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

MAX_X_EPISODES = 75_000
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
    y_label: str = "Normalized energy E(phi,psi)",
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
    y_label: str = "Normalized energy E(phi,psi)",
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


def _rad_to_deg(rad: float) -> float:
    return float(np.degrees(rad))


def _render_landscape_on_ax(
    ax: plt.Axes,
    landscape_x: np.ndarray | None,
    landscape_y: np.ndarray | None,
    landscape_energy: np.ndarray | None,
    fig: plt.Figure | None = None,
    landscape_label: str = "E(phi,psi)",
    show_colorbar: bool = True,
) -> None:
    """Draw energy contour on an axis."""
    if (
        landscape_x is None
        or landscape_y is None
        or landscape_energy is None
        or landscape_x.shape != landscape_y.shape
        or landscape_x.shape != landscape_energy.shape
    ):
        return
    lx_deg = np.degrees(landscape_x)
    ly_deg = np.degrees(landscape_y)
    contour = ax.contourf(
        lx_deg, ly_deg, landscape_energy,
        levels=28, cmap="YlOrRd", alpha=0.72, zorder=1,
    )
    if show_colorbar and fig is not None:
        fig.colorbar(contour, ax=ax, pad=0.01, fraction=0.045, label=landscape_label)


def _break_at_wraps(tx: np.ndarray, ty: np.ndarray, threshold_deg: float = 180.0):
    """Insert NaN at points where either coordinate jumps > threshold.

    This prevents matplotlib from drawing long lines across the plot when the
    trajectory wraps around the torus boundary.  Returns new (tx, ty) arrays
    with NaN gaps inserted.
    """
    dx = np.abs(np.diff(tx))
    dy = np.abs(np.diff(ty))
    wrap_mask = (dx > threshold_deg) | (dy > threshold_deg)
    if not np.any(wrap_mask):
        return tx, ty
    # Indices *after* which we need a NaN break
    break_idxs = np.where(wrap_mask)[0] + 1  # position of the second point
    tx_out = np.insert(tx.astype(np.float64), break_idxs, np.nan)
    ty_out = np.insert(ty.astype(np.float64), break_idxs, np.nan)
    return tx_out, ty_out


def _render_single_trajectory(
    ax: plt.Axes,
    traj: np.ndarray,
    start_xy_deg: np.ndarray,
    target_xy_deg: np.ndarray,
    color: str,
    label: str,
    title: str,
) -> None:
    """Draw one trajectory + start/target markers on an axis."""
    if traj is None or traj.ndim != 2 or traj.shape[1] != 2 or traj.shape[0] < 2:
        ax.set_title(f"{title} (no data)", loc="left", fontsize=10, pad=6)
        return
    tx = np.degrees(traj[:, 0])
    ty = np.degrees(traj[:, 1])
    # Break the line at torus-wrap boundaries so matplotlib doesn't draw
    # lines across the entire plot.
    tx_line, ty_line = _break_at_wraps(tx, ty)
    ax.plot(
        tx_line, ty_line, color=color, linewidth=1.5, alpha=0.92, zorder=4,
        label=label,
    )
    # Plot dots separately (NaN gaps would hide them in the line call)
    ax.scatter(
        tx, ty, color=color, s=5.3, alpha=0.92, zorder=5,
        edgecolors="none",
    )
    ax.scatter(
        [start_xy_deg[0]], [start_xy_deg[1]], color="#fb8500", edgecolor="black",
        linewidth=0.5, s=54, zorder=6, label="Start",
    )
    ax.scatter(
        [target_xy_deg[0]], [target_xy_deg[1]], color="#1f883d", edgecolor="black",
        linewidth=0.7, s=78, zorder=7, label="Global min",
    )
    ax.set_xlabel("phi (degrees)")
    ax.set_ylabel("psi (degrees)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", fontsize=10, pad=6)
    ax.legend(loc="best", frameon=False, fontsize=7)


# Method definitions used for individual + combined trajectory plots
_TRAJECTORY_METHODS = [
    ("rl_hidden_gradient", "PPO hidden-gradient oracle", "#111111"),
    ("gd", "GD baseline", "#cf222e"),
    ("adam", "Adam baseline", "green"),
    ("rl_no_oracle", "PPO no oracle", "#0969da"),
    ("rl_visible_oracle", "PPO visible-gradient oracle", "limegreen"),
]


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
    landscape_label: str = "E(phi,psi)",
) -> None:
    """Plot each method's trajectory individually + a combined subplot figure.

    Saves to output_path.parent / trajectories/ directory:
      - One PNG per method (traj_<method_key>.png)
      - A combined subplot figure (trajectory_combined.png)
    The legacy single-file path (output_path) is also written as the combined figure.
    """
    if trajectory_xy.ndim != 2 or trajectory_xy.shape[1] != 2 or trajectory_xy.shape[0] < 2:
        return

    _ = gradient_xy
    _ = move_vectors_xy

    # Build method -> trajectory mapping
    method_trajs: dict[str, np.ndarray | None] = {
        "rl_hidden_gradient": trajectory_xy,
        "gd": baseline_trajectory_xy,
        "adam": adam_baseline_trajectory_xy,
        "rl_no_oracle": no_oracle_trajectory_xy,
        "rl_visible_oracle": visible_gradient_trajectory_xy,
    }

    start_deg = np.degrees(trajectory_xy[0])
    target_deg = np.degrees(target_xy)

    traj_dir = output_path.parent / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # --- Individual plots ---
    for method_key, method_label, color in _TRAJECTORY_METHODS:
        traj = method_trajs.get(method_key)
        fig, ax = _make_axes(figsize=(7.2, 7.2))
        ax.set_facecolor("none")
        _render_landscape_on_ax(
            ax, landscape_x, landscape_y, landscape_energy,
            fig=fig, landscape_label=landscape_label,
        )
        _render_single_trajectory(
            ax, traj, start_deg, target_deg, color, method_label,
            title=f"{title} | {method_label}",
        )
        fig.tight_layout()
        fig.savefig(traj_dir / f"traj_{method_key}.png", dpi=200)
        plt.close(fig)

    # --- Combined subplot figure ---
    present = [
        (key, label, color)
        for key, label, color in _TRAJECTORY_METHODS
        if method_trajs.get(key) is not None
    ]
    n = max(1, len(present))
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig_w = 5.8 * ncols
    fig_h = 5.4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("white")

    for idx, (method_key, method_label, color) in enumerate(present):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_facecolor("none")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d0d7de")
        ax.spines["bottom"].set_color("#d0d7de")
        ax.tick_params(colors="#57606a")
        _render_landscape_on_ax(
            ax, landscape_x, landscape_y, landscape_energy,
            fig=fig, landscape_label=landscape_label, show_colorbar=False,
        )
        _render_single_trajectory(
            ax, method_trajs[method_key], start_deg, target_deg,
            color, method_label, title=method_label,
        )

    # Hide unused axes
    for idx in range(len(present), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    combined_path = traj_dir / "trajectory_combined.png"
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)

    # Also save to the legacy output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(combined_path), str(output_path))
