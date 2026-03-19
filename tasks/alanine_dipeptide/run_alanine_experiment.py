"""Multi-seed experiment runner for alanine dipeptide optimization.

Compares hidden-gradient PPO, visible-gradient PPO, no-oracle PPO, Adam, GD,
and random search on the Ramachandran energy surface using SDP-based oracle.

Example usage:
    python -m tasks.alanine_dipeptide.run_alanine_experiment \
        --seeds 0,1,2 --train_steps 300000 --K_map 6 --K_relax 4
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from .config import TrainConfig
from .train import (
    _resolve_device,
    _rollout_adam_curve,
    _rollout_descent_curve,
    _rollout_policy_curve,
    maybe_save_trajectory_plot,
    run_training,
)

TWO_PI = 2.0 * np.pi

META_METHOD_ORDER = (
    "gd",
    "adam",
    "random_search",
    "rl_no_oracle",
    "rl_visible_oracle",
    "rl_hidden_gradient",
)

METHOD_LABELS = {
    "gd": "GD",
    "adam": "Adam",
    "random_search": "Random search",
    "rl_no_oracle": "RL no oracle",
    "rl_visible_oracle": "RL visible oracle",
    "rl_hidden_gradient": "RL hidden gradient",
}


def _parse_int_list(raw: str, arg_name: str, *, min_value: int | None = None) -> list[int]:
    values: list[int] = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{arg_name} contains non-integer token: {token!r}") from exc
        if min_value is not None and value < min_value:
            raise ValueError(f"{arg_name} must be >= {min_value}, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must include at least one integer")
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _method_colors(method_order: tuple[str, ...]) -> dict[str, tuple]:
    palette = plt.cm.plasma(np.linspace(0.12, 0.88, max(1, len(method_order))))
    return {method: palette[idx] for idx, method in enumerate(method_order)}


def _make_axes(figsize: tuple[float, float] = (8.6, 5.0)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d7de")
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(colors="#57606a")
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.minorticks_on()
    return fig, ax


def _ci95(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 0.0
    return float(1.96 * np.std(finite, ddof=1) / np.sqrt(float(finite.size)))


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "ci95": float("nan"),
            "num_values": 0,
        }
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return {
        "mean": float(np.mean(finite)),
        "std": std,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
        "ci95": _ci95(finite),
        "num_values": int(finite.size),
    }


# ---------------------------------------------------------------------------
# Rollout helpers for evaluation (operate on a single env_index=0)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_policy_search_curve(
    *,
    model: torch.nn.Module,
    env,
    device: torch.device,
    start_xy: np.ndarray,
    horizon: int,
    env_index: int = 0,
    deterministic: bool,
) -> np.ndarray:
    state = start_xy.astype(np.float32).copy()
    h = max(1, int(horizon))
    curve = np.empty(h + 1, dtype=np.float32)
    curve[0] = float(env._normalized_objective_value(state, env_index=env_index))
    hidden_state = model.initial_state(batch_size=1, device=device)

    for step in range(h):
        token_features = env._obs_token_features(state, env_index=env_index)[None, :]
        dist_feature = env._normalized_objective_value(state, env_index=env_index)
        step_fraction = float(step / max(1, h))

        token_t = torch.tensor(token_features, dtype=torch.float32, device=device)
        dist_t = torch.tensor([dist_feature], dtype=torch.float32, device=device)
        step_t = torch.tensor([step_fraction], dtype=torch.float32, device=device)

        action_t, _, _, hidden_state = model.act(
            token_t,
            dist_t,
            step_t,
            hidden_state=hidden_state,
            deterministic=deterministic,
        )
        action = action_t.squeeze(0).cpu().numpy()
        state = env._apply_action(state, action)
        curve[step + 1] = float(env._normalized_objective_value(state, env_index=env_index))

    return curve


def _rollout_random_search_curve(
    *,
    env,
    start_xy: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    env_index: int = 0,
) -> np.ndarray:
    h = max(1, int(horizon))
    best_objective = float(
        env._normalized_objective_value(start_xy.astype(np.float32), env_index=env_index)
    )
    curve = np.empty(h + 1, dtype=np.float32)
    curve[0] = best_objective
    for step in range(h):
        candidate = rng.uniform(0.0, TWO_PI, size=int(env.visible_dim)).astype(np.float32)
        candidate_objective = float(
            env._normalized_objective_value(candidate, env_index=env_index)
        )
        if candidate_objective < best_objective:
            best_objective = candidate_objective
        curve[step + 1] = best_objective
    return curve


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------

def _evaluate_seed(
    *,
    seed: int,
    device: torch.device,
    hidden_model: torch.nn.Module,
    hidden_env,
    no_oracle_model: torch.nn.Module | None,
    no_oracle_env,
    visible_gradient_model: torch.nn.Module | None,
    visible_gradient_env,
    num_tasks: int,
    horizon: int,
) -> dict[str, Any]:
    hidden_was_training = bool(hidden_model.training)
    no_oracle_was_training = bool(no_oracle_model.training) if no_oracle_model is not None else None
    visible_was_training = (
        bool(visible_gradient_model.training) if visible_gradient_model is not None else None
    )

    hidden_model.eval()
    if no_oracle_model is not None:
        no_oracle_model.eval()
    if visible_gradient_model is not None:
        visible_gradient_model.eval()

    method_curves: dict[str, list[np.ndarray]] = {method: [] for method in META_METHOD_ORDER}
    random_rng = np.random.default_rng(int(seed) + 50_003)

    for _ in range(max(1, int(num_tasks))):
        # Sample a random starting point on the torus
        start_xy = hidden_env.rng.uniform(0.0, TWO_PI, size=int(hidden_env.visible_dim)).astype(
            np.float32
        )

        # GD baseline
        gd_curve = _rollout_descent_curve(
            env=hidden_env,
            start_xy=start_xy,
            horizon=horizon,
            env_index=0,
            base_lr=hidden_env.baseline_lr_gd,
        )
        method_curves["gd"].append(gd_curve)

        # Adam baseline
        adam_curve = _rollout_adam_curve(
            env=hidden_env,
            start_xy=start_xy,
            horizon=horizon,
            env_index=0,
            base_lr=hidden_env.baseline_lr_adam,
        )
        method_curves["adam"].append(adam_curve)

        # Random search
        random_search_curve = _rollout_random_search_curve(
            env=hidden_env,
            start_xy=start_xy,
            horizon=horizon,
            rng=random_rng,
            env_index=0,
        )
        method_curves["random_search"].append(random_search_curve)

        # Hidden gradient RL
        hidden_curve = _rollout_policy_curve(
            model=hidden_model,
            env=hidden_env,
            device=device,
            start_xy=start_xy,
            horizon=horizon,
            env_index=0,
        )
        method_curves["rl_hidden_gradient"].append(hidden_curve)

        # No oracle RL
        if no_oracle_model is not None and no_oracle_env is not None:
            no_curve = _rollout_policy_curve(
                model=no_oracle_model,
                env=no_oracle_env,
                device=device,
                start_xy=start_xy,
                horizon=horizon,
                env_index=0,
            )
            method_curves["rl_no_oracle"].append(no_curve)

        # Visible gradient RL
        if visible_gradient_model is not None and visible_gradient_env is not None:
            visible_curve = _rollout_policy_curve(
                model=visible_gradient_model,
                env=visible_gradient_env,
                device=device,
                start_xy=start_xy,
                horizon=horizon,
                env_index=0,
            )
            method_curves["rl_visible_oracle"].append(visible_curve)

    # Restore training mode
    if hidden_was_training:
        hidden_model.train()
    if no_oracle_model is not None and bool(no_oracle_was_training):
        no_oracle_model.train()
    if visible_gradient_model is not None and bool(visible_was_training):
        visible_gradient_model.train()

    methods_payload: dict[str, Any] = {}
    for method in META_METHOD_ORDER:
        curves_list = method_curves.get(method, [])
        if not curves_list:
            continue
        curves = np.stack(curves_list, axis=0).astype(np.float64)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        final_values = curves[:, -1]
        best_values = np.nanmin(curves, axis=1)
        methods_payload[method] = {
            "label": METHOD_LABELS.get(method, method),
            "num_tasks": int(curves.shape[0]),
            "mean_curve": mean_curve.tolist(),
            "std_curve": std_curve.tolist(),
            "task_curves": curves.tolist(),
            "final_objective_values": [float(v) for v in final_values],
            "best_objective_values": [float(v) for v in best_values],
            "final_objective_stats": _stats([float(v) for v in final_values]),
            "best_objective_stats": _stats([float(v) for v in best_values]),
        }

    return {
        "seed": int(seed),
        "num_tasks": int(num_tasks),
        "horizon": int(horizon),
        "method_order": list(META_METHOD_ORDER),
        "methods": methods_payload,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_results(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_method_curves: dict[str, list[np.ndarray]] = {m: [] for m in META_METHOD_ORDER}
    by_method_seed_final_mean: dict[str, list[float]] = {m: [] for m in META_METHOD_ORDER}
    by_method_seed_best_mean: dict[str, list[float]] = {m: [] for m in META_METHOD_ORDER}
    by_method_all_final: dict[str, list[float]] = {m: [] for m in META_METHOD_ORDER}
    by_method_all_best: dict[str, list[float]] = {m: [] for m in META_METHOD_ORDER}

    horizon = None
    for payload in seed_payloads:
        if horizon is None:
            horizon = int(payload["horizon"])
        for method in META_METHOD_ORDER:
            method_data = payload.get("methods", {}).get(method)
            if method_data is None:
                continue
            task_curves = np.asarray(method_data.get("task_curves", []), dtype=np.float64)
            if task_curves.ndim == 2 and task_curves.shape[0] > 0 and task_curves.shape[1] >= 2:
                by_method_curves[method].append(task_curves)
                final_values = task_curves[:, -1]
                best_values = np.min(task_curves, axis=1)
                by_method_all_final[method].extend(float(v) for v in final_values)
                by_method_all_best[method].extend(float(v) for v in best_values)
                by_method_seed_final_mean[method].append(float(np.mean(final_values)))
                by_method_seed_best_mean[method].append(float(np.mean(best_values)))

    success_threshold = 0.01
    methods_payload: dict[str, Any] = {}
    for method in META_METHOD_ORDER:
        if not by_method_curves[method]:
            continue
        stacked = np.concatenate(by_method_curves[method], axis=0)
        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)
        # Success rate at each step: fraction of tasks with objective <= threshold
        instantaneous_success = np.mean(stacked <= success_threshold, axis=0)
        # Cumulative success: fraction of tasks that have EVER reached threshold by step t
        cummin = np.minimum.accumulate(stacked, axis=1)
        cumulative_success = np.mean(cummin <= success_threshold, axis=0)
        methods_payload[method] = {
            "label": METHOD_LABELS.get(method, method),
            "num_total_tasks": int(stacked.shape[0]),
            "num_seeds": int(len(by_method_seed_final_mean[method])),
            "mean_curve": [float(v) for v in mean_curve],
            "std_curve": [float(v) for v in std_curve],
            "instantaneous_success_curve": [float(v) for v in instantaneous_success],
            "cumulative_success_curve": [float(v) for v in cumulative_success],
            "seed_mean_final_objective_values": [
                float(v) for v in by_method_seed_final_mean[method]
            ],
            "seed_mean_best_objective_values": [
                float(v) for v in by_method_seed_best_mean[method]
            ],
            "all_task_final_objective_values": [float(v) for v in by_method_all_final[method]],
            "all_task_best_objective_values": [float(v) for v in by_method_all_best[method]],
            "seed_mean_final_objective_stats": _stats(by_method_seed_final_mean[method]),
            "seed_mean_best_objective_stats": _stats(by_method_seed_best_mean[method]),
            "all_task_final_objective_stats": _stats(by_method_all_final[method]),
            "all_task_best_objective_stats": _stats(by_method_all_best[method]),
        }

    return {
        "method_order": list(META_METHOD_ORDER),
        "horizon": int(horizon) if horizon is not None else None,
        "num_seed_payloads": int(len(seed_payloads)),
        "methods": methods_payload,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_objective_vs_step(*, aggregate: dict[str, Any], output_path: Path) -> None:
    methods = aggregate.get("methods", {})
    horizon = int(aggregate.get("horizon") or 0)
    if not methods or horizon < 1:
        return
    colors = _method_colors(META_METHOD_ORDER)
    fig, ax = _make_axes(figsize=(8.8, 5.2))
    finite_chunks: list[np.ndarray] = []
    for method in META_METHOD_ORDER:
        data = methods.get(method)
        if data is None:
            continue
        mean_curve = np.asarray(data.get("mean_curve", []), dtype=np.float64)
        std_curve = np.asarray(data.get("std_curve", []), dtype=np.float64)
        if mean_curve.size < 2:
            continue
        steps = np.arange(mean_curve.size, dtype=np.int64)
        color = colors[method]
        label = METHOD_LABELS.get(method, method)
        ax.plot(steps, mean_curve, color=color, linewidth=2.0, label=label)
        if std_curve.size == mean_curve.size:
            ax.fill_between(
                steps,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        finite = mean_curve[np.isfinite(mean_curve)]
        if finite.size > 0:
            finite_chunks.append(finite)

    ax.set_title(
        "Alanine dipeptide | normalized objective vs optimization step",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Normalized energy E(phi,psi)")
    ax.set_xlim(0.0, float(max(1, horizon)))
    if finite_chunks:
        finite_values = np.concatenate(finite_chunks)
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        margin = 0.08 * max(1e-6, y_max - y_min)
        ax.set_ylim(max(0.0, y_min - margin), min(1.02, y_max + margin))
    else:
        ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_final_objective_summary(*, aggregate: dict[str, Any], output_path: Path) -> None:
    methods = aggregate.get("methods", {})
    labels: list[str] = []
    means: list[float] = []
    ci95_vals: list[float] = []
    for method in META_METHOD_ORDER:
        data = methods.get(method)
        if data is None:
            continue
        stats = data.get("seed_mean_final_objective_stats", {})
        mean_val = float(stats.get("mean", float("nan")))
        ci_val = float(stats.get("ci95", float("nan")))
        if not np.isfinite(mean_val):
            continue
        labels.append(METHOD_LABELS.get(method, method))
        means.append(mean_val)
        ci95_vals.append(0.0 if not np.isfinite(ci_val) else ci_val)
    if not labels:
        return

    fig, ax = _make_axes(figsize=(9.0, 5.0))
    x = np.arange(len(labels), dtype=np.float64)
    colors = plt.cm.plasma(np.linspace(0.16, 0.84, max(1, len(labels))))
    ax.scatter(x, means, s=46, color=colors, zorder=4)
    ax.errorbar(
        x,
        means,
        yerr=ci95_vals,
        fmt="none",
        ecolor="#24292f",
        elinewidth=1.2,
        capsize=4,
        zorder=3,
    )
    for idx, (xv, yv) in enumerate(zip(x, means)):
        ax.vlines(xv, 0.0, yv, color=colors[idx], alpha=0.28, linewidth=1.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Final normalized objective")
    ax.set_xlabel("Method")
    ax.set_title(
        "Alanine dipeptide | final objective (mean across seeds with 95% CI)",
        loc="left",
        fontsize=11,
        pad=10,
    )
    y_min = float(np.min(means))
    y_max = float(np.max(means))
    margin = 0.08 * max(1e-6, y_max - y_min)
    ax.set_ylim(max(0.0, y_min - margin), min(1.02, y_max + margin))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_success_rate_vs_step(*, aggregate: dict[str, Any], output_path: Path) -> None:
    """Cumulative success rate vs optimization step — fraction of tasks that have
    ever reached the global minimum (objective <= 0.01) by step t."""
    methods = aggregate.get("methods", {})
    horizon = int(aggregate.get("horizon") or 0)
    if not methods or horizon < 1:
        return
    colors = _method_colors(META_METHOD_ORDER)
    fig, ax = _make_axes(figsize=(8.8, 5.2))
    for method in META_METHOD_ORDER:
        data = methods.get(method)
        if data is None:
            continue
        curve = np.asarray(data.get("cumulative_success_curve", []), dtype=np.float64)
        if curve.size < 2:
            continue
        steps = np.arange(curve.size, dtype=np.int64)
        color = colors[method]
        label = METHOD_LABELS.get(method, method)
        ax.plot(steps, curve * 100.0, color=color, linewidth=2.2, label=label)

    ax.set_title(
        "Alanine dipeptide | cumulative success rate vs optimization step",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Tasks that reached global min (%)")
    ax.set_xlim(0.0, float(max(1, horizon)))
    ax.set_ylim(0.0, 105.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_final_success_rate_summary(*, aggregate: dict[str, Any], output_path: Path) -> None:
    """Bar chart of final success rates (objective <= 0.01 at last step)."""
    methods = aggregate.get("methods", {})
    labels: list[str] = []
    rates: list[float] = []
    ever_rates: list[float] = []
    for method in META_METHOD_ORDER:
        data = methods.get(method)
        if data is None:
            continue
        inst_curve = data.get("instantaneous_success_curve", [])
        cum_curve = data.get("cumulative_success_curve", [])
        if not inst_curve or not cum_curve:
            continue
        labels.append(METHOD_LABELS.get(method, method))
        rates.append(float(inst_curve[-1]) * 100.0)
        ever_rates.append(float(cum_curve[-1]) * 100.0)
    if not labels:
        return

    fig, ax = _make_axes(figsize=(9.0, 5.0))
    x = np.arange(len(labels), dtype=np.float64)
    bar_width = 0.35
    colors_end = plt.cm.plasma(np.linspace(0.16, 0.84, max(1, len(labels))))
    colors_ever = plt.cm.plasma(np.linspace(0.30, 0.92, max(1, len(labels))))

    bars_ever = ax.bar(
        x - bar_width / 2, ever_rates, bar_width,
        color=colors_ever, alpha=0.55, edgecolor="white", linewidth=0.8,
        label="Ever reached",
    )
    bars_end = ax.bar(
        x + bar_width / 2, rates, bar_width,
        color=colors_end, edgecolor="white", linewidth=0.8,
        label="At final step",
    )

    for bar, val in zip(bars_ever, ever_rates):
        if val > 3:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#57606a",
            )
    for bar, val in zip(bars_end, rates):
        if val > 3:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#57606a",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Method")
    ax.set_title(
        "Alanine dipeptide | success rate (objective <= 0.01)",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.set_ylim(0.0, 115.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-seed trajectory plots
# ---------------------------------------------------------------------------

def _maybe_save_seed_trajectory_plots(
    *,
    study_root: Path,
    seed: int,
    config: TrainConfig,
    device: torch.device,
    hidden_model: torch.nn.Module,
    hidden_env,
    no_oracle_model: torch.nn.Module | None,
    visible_gradient_model: torch.nn.Module | None,
    visible_gradient_env,
    skip_plotting: bool,
) -> dict[str, Any]:
    if bool(skip_plotting):
        return {}

    seed_plots_root = study_root / "plots" / f"seed{int(seed)}"
    seed_plots_root.mkdir(parents=True, exist_ok=True)

    trajectory_plots: list[str] = []
    num_plot_tasks = 10
    for task_index in range(num_plot_tasks):
        task_dir = seed_plots_root / f"task{int(task_index + 1):02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / "trajectory_combined.png"
        maybe_save_trajectory_plot(
            model=hidden_model,
            env=hidden_env,
            device=device,
            no_oracle_model=no_oracle_model,
            visible_gradient_model=visible_gradient_model,
            visible_gradient_env=visible_gradient_env,
            output_path=output_path,
            title=(
                f"Ramachandran trajectory | seed={int(seed)}, "
                f"task={int(task_index + 1)}, K_map={int(config.K_map)}, "
                f"mode={config.oracle_mode}"
            ),
        )
        # Individual per-method plots are in task_dir/trajectories/
        combined = task_dir / "trajectories" / "trajectory_combined.png"
        if combined.exists():
            trajectory_plots.append(str(combined.resolve()))
        elif output_path.exists():
            trajectory_plots.append(str(output_path.resolve()))

    return {
        "plots_dir": str(seed_plots_root.resolve()),
        "num_plot_tasks_requested": int(num_plot_tasks),
        "trajectory_plots": trajectory_plots,
    }


# ---------------------------------------------------------------------------
# Build config helper
# ---------------------------------------------------------------------------

def _build_train_config(
    *,
    args: argparse.Namespace,
    seed: int,
    run_name: str,
) -> TrainConfig:
    return TrainConfig(
        task="alanine_dipeptide",
        seed=int(seed),
        run_name=run_name,
        logdir=str(args.logdir),
        device=str(args.device),
        K_map=int(args.K_map),
        K_relax=int(args.K_relax),
        energy_json=str(args.energy_json),
        use_synthetic_fallback=bool(args.use_synthetic_fallback),
        train_steps=int(args.train_steps),
        n_env=int(args.n_env),
        rollout_len=int(args.rollout_len),
        running_avg_window=int(args.running_avg_window),
        eval_interval_episodes=int(args.eval_interval_episodes),
        save_metrics_interval_episodes=int(args.save_metrics_interval_episodes),
        max_horizon=int(args.max_horizon),
        sensing=str(args.sensing),
        oracle_mode="convex_gradient",
        lr=float(args.lr),
        ppo_epochs=int(args.ppo_epochs),
        minibatches=int(args.minibatches),
        hidden_dim=int(args.policy_hidden_dim),
        oracle_proj_dim=int(args.oracle_proj_dim),
        step_size=float(args.step_size),
        ppo_step_scale=float(args.ppo_step_scale),
        success_threshold=float(args.success_threshold),
        policy_arch=str(args.policy_arch),
        enable_baselines=True,
        tune_baseline_lrs=not bool(args.disable_baseline_lr_tuning),
        baseline_lr_candidates=str(args.baseline_lr_candidates),
        baseline_lr_tune_tasks=int(args.baseline_lr_tune_tasks),
        enable_optimization_curve_eval=False,
        enable_training_plots=False,
    )


# ---------------------------------------------------------------------------
# Main study runner
# ---------------------------------------------------------------------------

def _run_study(
    *,
    args: argparse.Namespace,
    seeds: list[int],
    suite_root: Path,
) -> dict[str, Any]:
    study_root = suite_root
    plots_root = study_root / "plots"
    data_root = study_root / "plot_data"
    plots_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(args.device))
    seed_payloads: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []

    for seed in seeds:
        run_name = f"alanine_Kmap{int(args.K_map)}_Krelax{int(args.K_relax)}_seed{int(seed)}"
        config = _build_train_config(args=args, seed=int(seed), run_name=run_name)
        print(
            f"[alanine_experiment] seed={int(seed)} training run={run_name} "
            f"(train_steps={int(config.train_steps)})"
        )
        train_t0 = time.perf_counter()
        output = run_training(config, return_artifacts=True)
        train_elapsed_sec = float(time.perf_counter() - train_t0)

        hidden_model = output["model"]
        hidden_env = output.get("env")
        if hidden_env is None:
            raise RuntimeError("Expected alanine env artifact in training output")
        no_oracle_model = output.get("no_oracle_model")
        no_oracle_env = output.get("no_oracle_env")
        visible_gradient_model = output.get("visible_gradient_model")
        visible_gradient_env = output.get("visible_gradient_env")

        seed_payload = _evaluate_seed(
            seed=int(seed),
            device=device,
            hidden_model=hidden_model,
            hidden_env=hidden_env,
            no_oracle_model=no_oracle_model,
            no_oracle_env=no_oracle_env,
            visible_gradient_model=visible_gradient_model,
            visible_gradient_env=visible_gradient_env,
            num_tasks=int(args.eval_num_tasks),
            horizon=int(args.eval_horizon),
        )
        seed_payload_path = data_root / f"seed{int(seed)}_evaluation.json"
        with seed_payload_path.open("w", encoding="utf-8") as handle:
            json.dump(seed_payload, handle, indent=2)
        seed_payloads.append(seed_payload)

        seed_plot_paths = _maybe_save_seed_trajectory_plots(
            study_root=study_root,
            seed=int(seed),
            config=config,
            device=device,
            hidden_model=hidden_model,
            hidden_env=hidden_env,
            no_oracle_model=no_oracle_model,
            visible_gradient_model=visible_gradient_model,
            visible_gradient_env=visible_gradient_env,
            skip_plotting=bool(args.skip_plotting),
        )

        run_dir = Path(str(output["summary"]["run_dir"])).expanduser().resolve()
        run_entries.append(
            {
                "seed": int(seed),
                "run_name": run_name,
                "run_dir": str(run_dir),
                "summary_json": str((run_dir / "summary.json").resolve()),
                "training_wall_time_sec": train_elapsed_sec,
                "evaluation_json": str(seed_payload_path.resolve()),
                "seed_plot_paths": seed_plot_paths,
            }
        )

        del output

    aggregate = _aggregate_results(seed_payloads)
    aggregate_path = data_root / "aggregate.json"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    plot_paths: dict[str, str] = {}
    if not bool(args.skip_plotting):
        curves_plot = plots_root / "objective_vs_step.png"
        final_plot = plots_root / "final_objective_summary.png"
        success_curve_plot = plots_root / "success_rate_vs_step.png"
        success_bar_plot = plots_root / "final_success_rate_summary.png"
        _plot_objective_vs_step(aggregate=aggregate, output_path=curves_plot)
        _plot_final_objective_summary(aggregate=aggregate, output_path=final_plot)
        _plot_success_rate_vs_step(aggregate=aggregate, output_path=success_curve_plot)
        _plot_final_success_rate_summary(aggregate=aggregate, output_path=success_bar_plot)
        if curves_plot.exists():
            plot_paths["objective_vs_step_plot"] = str(curves_plot.resolve())
        if final_plot.exists():
            plot_paths["final_objective_summary_plot"] = str(final_plot.resolve())
        if success_curve_plot.exists():
            plot_paths["success_rate_vs_step_plot"] = str(success_curve_plot.resolve())
        if success_bar_plot.exists():
            plot_paths["final_success_rate_summary_plot"] = str(success_bar_plot.resolve())

    return {
        "study": "alanine_dipeptide",
        "study_root": str(study_root.resolve()),
        "method_order": list(META_METHOD_ORDER),
        "method_labels": dict(METHOD_LABELS),
        "seeds": [int(seed) for seed in seeds],
        "eval_num_tasks": int(args.eval_num_tasks),
        "eval_horizon": int(args.eval_horizon),
        "runs": run_entries,
        "aggregate_json": str(aggregate_path.resolve()),
        "plots": plot_paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Run alanine dipeptide multi-seed optimizer comparison experiment."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds.",
    )
    parser.add_argument(
        "--suite_output_dir",
        type=str,
        default="plots/alanine_experiment",
    )
    parser.add_argument("--skip_plotting", action="store_true")

    # Training config
    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--K_map", type=int, default=defaults.K_map)
    parser.add_argument("--K_relax", type=int, default=defaults.K_relax)
    parser.add_argument("--energy_json", type=str, default=defaults.energy_json)
    parser.add_argument("--use_synthetic_fallback", action="store_true")
    parser.add_argument("--train_steps", type=int, default=defaults.train_steps)
    parser.add_argument("--n_env", type=int, default=defaults.n_env)
    parser.add_argument("--rollout_len", type=int, default=defaults.rollout_len)
    parser.add_argument("--running_avg_window", type=int, default=defaults.running_avg_window)
    parser.add_argument(
        "--save_metrics_interval_episodes",
        type=int,
        default=defaults.save_metrics_interval_episodes,
    )
    parser.add_argument(
        "--eval_interval_episodes", type=int, default=defaults.eval_interval_episodes
    )
    parser.add_argument("--max_horizon", type=int, default=defaults.max_horizon)
    parser.add_argument("--sensing", type=str, choices=["S0", "S1"], default=defaults.sensing)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--ppo_epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)
    parser.add_argument("--policy_hidden_dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--oracle_proj_dim", type=int, default=defaults.oracle_proj_dim)
    parser.add_argument("--step_size", type=float, default=defaults.step_size)
    parser.add_argument("--ppo_step_scale", type=float, default=defaults.ppo_step_scale)
    parser.add_argument("--success_threshold", type=float, default=defaults.success_threshold)
    parser.add_argument(
        "--policy_arch", type=str, choices=["mlp", "gru"], default=defaults.policy_arch
    )
    parser.add_argument(
        "--disable_baseline_lr_tuning",
        action="store_true",
        help="Disable automatic lr tuning for GD/Adam baselines.",
    )
    parser.add_argument(
        "--baseline_lr_candidates",
        type=str,
        default=defaults.baseline_lr_candidates,
    )
    parser.add_argument(
        "--baseline_lr_tune_tasks",
        type=int,
        default=defaults.baseline_lr_tune_tasks,
    )

    # Evaluation config
    parser.add_argument(
        "--eval_num_tasks",
        type=int,
        default=500,
        help="Number of random-start evaluation tasks per seed.",
    )
    parser.add_argument(
        "--eval_horizon",
        type=int,
        default=defaults.max_horizon,
        help="Optimization steps per evaluation task.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds, "seeds", min_value=0)

    suite_root = Path(args.suite_output_dir).expanduser().resolve()
    suite_root.mkdir(parents=True, exist_ok=True)

    manifest = _run_study(args=args, seeds=seeds, suite_root=suite_root)

    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = suite_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "suite_root": str(suite_root.resolve()),
        "num_seeds": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "plots": manifest.get("plots", {}),
    }
    summary_path = suite_root / "suite_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
