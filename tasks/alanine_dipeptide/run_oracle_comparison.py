"""Multi-seed experiment comparing SDP oracle vs norm-ball oracle on alanine dipeptide.

Trains PPO agents under both oracle regimes across multiple seeds, then produces:
  - Per-seed training learning curves (success rate, objective vs episode)
  - Post-training evaluation (optimization curves on held-out tasks)
  - Aggregate comparison plots and JSON data for analysis

Example usage:
    python -m tasks.alanine_dipeptide.run_oracle_comparison \
        --seeds 0,1,2,3,4 --train_steps 300000 --K_map 6 --K_relax 4
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
    run_training,
)

TWO_PI = 2.0 * np.pi

ORACLE_TYPES = ("sdp", "norm_ball")

ORACLE_LABELS = {
    "sdp": "SDP oracle",
    "norm_ball": "Norm-ball oracle",
}

ORACLE_COLORS = {
    "sdp": "#2563eb",       # blue
    "norm_ball": "#dc2626",  # red
}

META_METHOD_ORDER = (
    "gd",
    "adam",
    "rl_no_oracle",
    "rl_visible_oracle",
    "rl_hidden_gradient",
)

METHOD_LABELS = {
    "gd": "GD",
    "adam": "Adam",
    "rl_no_oracle": "RL no oracle",
    "rl_visible_oracle": "RL visible oracle",
    "rl_hidden_gradient": "RL hidden gradient",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def _ci95(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 0.0
    return float(1.96 * np.std(finite, ddof=1) / np.sqrt(float(finite.size)))


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "median": float("nan"), "ci95": float("nan"),
                "num_values": 0}
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return {
        "mean": float(np.mean(finite)), "std": std,
        "min": float(np.min(finite)), "max": float(np.max(finite)),
        "median": float(np.median(finite)), "ci95": _ci95(finite),
        "num_values": int(finite.size),
    }


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


# ---------------------------------------------------------------------------
# Learning curve extraction from metrics.jsonl
# ---------------------------------------------------------------------------

def _extract_learning_curves(run_dir: Path, window: int = 50) -> dict[str, Any]:
    """Read metrics.jsonl and extract smoothed learning curves."""
    jsonl_path = run_dir / "metrics.jsonl"
    if not jsonl_path.exists():
        return {}

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {}

    episodes = [int(r["episodes"]) for r in rows]
    steps = [int(r["steps"]) for r in rows]

    # Raw per-episode values
    success_raw = [float(r.get("success", 0.0)) for r in rows]
    objective_raw = [float(r.get("final_objective", float("nan"))) for r in rows]

    # Running-average values (already computed by the training loop)
    success_rate = [float(r.get("success_rate", float("nan"))) for r in rows]
    avg_objective = [float(r.get("avg_final_objective", float("nan"))) for r in rows]
    avg_adam_objective = [float(r.get("avg_adam_baseline_final_objective", float("nan"))) for r in rows]
    avg_gd_objective = [float(r.get("avg_baseline_final_objective", float("nan"))) for r in rows]
    avg_no_oracle_sr = [float(r.get("avg_no_oracle_success_rate", float("nan"))) for r in rows]
    avg_vis_grad_sr = [float(r.get("avg_visible_gradient_success_rate", float("nan"))) for r in rows]

    # Also compute a custom smoothed success rate with specified window
    def _smooth(arr, w):
        out = []
        buf = []
        for v in arr:
            buf.append(v)
            if len(buf) > w:
                buf.pop(0)
            out.append(float(np.mean(buf)))
        return out

    success_smoothed = _smooth(success_raw, window)

    return {
        "episodes": episodes,
        "steps": steps,
        "success_rate": success_rate,
        "success_smoothed": success_smoothed,
        "avg_final_objective": avg_objective,
        "avg_adam_baseline_final_objective": avg_adam_objective,
        "avg_baseline_final_objective": avg_gd_objective,
        "avg_no_oracle_success_rate": avg_no_oracle_sr,
        "avg_visible_gradient_success_rate": avg_vis_grad_sr,
        "num_episodes": len(rows),
        "smoothing_window": window,
    }


# ---------------------------------------------------------------------------
# Post-training evaluation
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
    visible_was_training = bool(visible_gradient_model.training) if visible_gradient_model is not None else None

    hidden_model.eval()
    if no_oracle_model is not None:
        no_oracle_model.eval()
    if visible_gradient_model is not None:
        visible_gradient_model.eval()

    method_curves: dict[str, list[np.ndarray]] = {m: [] for m in META_METHOD_ORDER}

    for _ in range(max(1, int(num_tasks))):
        start_xy = hidden_env.rng.uniform(0.0, TWO_PI, size=int(hidden_env.visible_dim)).astype(np.float32)

        method_curves["gd"].append(_rollout_descent_curve(
            env=hidden_env, start_xy=start_xy, horizon=horizon,
            env_index=0, base_lr=hidden_env.baseline_lr_gd,
        ))
        method_curves["adam"].append(_rollout_adam_curve(
            env=hidden_env, start_xy=start_xy, horizon=horizon,
            env_index=0, base_lr=hidden_env.baseline_lr_adam,
        ))
        method_curves["rl_hidden_gradient"].append(_rollout_policy_curve(
            model=hidden_model, env=hidden_env, device=device,
            start_xy=start_xy, horizon=horizon, env_index=0,
        ))
        if no_oracle_model is not None and no_oracle_env is not None:
            method_curves["rl_no_oracle"].append(_rollout_policy_curve(
                model=no_oracle_model, env=no_oracle_env, device=device,
                start_xy=start_xy, horizon=horizon, env_index=0,
            ))
        if visible_gradient_model is not None and visible_gradient_env is not None:
            method_curves["rl_visible_oracle"].append(_rollout_policy_curve(
                model=visible_gradient_model, env=visible_gradient_env, device=device,
                start_xy=start_xy, horizon=horizon, env_index=0,
            ))

    if hidden_was_training:
        hidden_model.train()
    if no_oracle_model is not None and bool(no_oracle_was_training):
        no_oracle_model.train()
    if visible_gradient_model is not None and bool(visible_was_training):
        visible_gradient_model.train()

    success_threshold = 0.01
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
        final_success = float(np.mean(final_values <= success_threshold))
        ever_success = float(np.mean(best_values <= success_threshold))
        methods_payload[method] = {
            "label": METHOD_LABELS.get(method, method),
            "num_tasks": int(curves.shape[0]),
            "mean_curve": mean_curve.tolist(),
            "std_curve": std_curve.tolist(),
            "final_objective_stats": _stats([float(v) for v in final_values]),
            "best_objective_stats": _stats([float(v) for v in best_values]),
            "final_success_rate": final_success,
            "ever_success_rate": ever_success,
        }

    return {
        "seed": int(seed),
        "num_tasks": int(num_tasks),
        "horizon": int(horizon),
        "method_order": list(META_METHOD_ORDER),
        "methods": methods_payload,
    }


# ---------------------------------------------------------------------------
# Aggregate across seeds (within one oracle type)
# ---------------------------------------------------------------------------

def _aggregate_eval_results(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_method_final: dict[str, list[float]] = {m: [] for m in META_METHOD_ORDER}
    by_method_success: dict[str, list[float]] = {m: [] for m in META_METHOD_ORDER}
    by_method_ever_success: dict[str, list[float]] = {m: [] for m in META_METHOD_ORDER}
    by_method_curves: dict[str, list[np.ndarray]] = {m: [] for m in META_METHOD_ORDER}

    for payload in seed_payloads:
        for method in META_METHOD_ORDER:
            mdata = payload.get("methods", {}).get(method)
            if mdata is None:
                continue
            by_method_final[method].append(float(mdata["final_objective_stats"]["mean"]))
            by_method_success[method].append(float(mdata["final_success_rate"]))
            by_method_ever_success[method].append(float(mdata["ever_success_rate"]))
            mean_curve = np.asarray(mdata.get("mean_curve", []), dtype=np.float64)
            if mean_curve.size >= 2:
                by_method_curves[method].append(mean_curve)

    methods_payload: dict[str, Any] = {}
    for method in META_METHOD_ORDER:
        if not by_method_final[method]:
            continue
        # Mean curve across seeds (average of per-seed means)
        if by_method_curves[method]:
            stacked = np.stack(by_method_curves[method], axis=0)
            grand_mean_curve = np.mean(stacked, axis=0).tolist()
            grand_std_curve = np.std(stacked, axis=0).tolist()
        else:
            grand_mean_curve = []
            grand_std_curve = []
        methods_payload[method] = {
            "label": METHOD_LABELS.get(method, method),
            "final_objective_stats": _stats(by_method_final[method]),
            "final_success_rate_stats": _stats(by_method_success[method]),
            "ever_success_rate_stats": _stats(by_method_ever_success[method]),
            "grand_mean_curve": grand_mean_curve,
            "grand_std_curve": grand_std_curve,
            "num_seeds": len(by_method_final[method]),
        }

    return {
        "method_order": list(META_METHOD_ORDER),
        "num_seeds": len(seed_payloads),
        "methods": methods_payload,
    }


# ---------------------------------------------------------------------------
# Plotting: training learning curves
# ---------------------------------------------------------------------------

def _plot_learning_curves_comparison(
    *,
    sdp_curves: list[dict[str, Any]],
    nb_curves: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    """Plot success rate and objective vs episode for both oracle types."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_created: list[str] = []

    # --- Success rate vs episode ---
    fig, ax = _make_axes(figsize=(10, 5.5))

    for curves, oracle_type in [(sdp_curves, "sdp"), (nb_curves, "norm_ball")]:
        color = ORACLE_COLORS[oracle_type]
        label = ORACLE_LABELS[oracle_type]

        # Collect all per-seed smoothed success curves, interpolate to common episode axis
        all_episodes = []
        all_success = []
        for c in curves:
            if not c or "episodes" not in c:
                continue
            all_episodes.append(np.array(c["episodes"], dtype=np.float64))
            all_success.append(np.array(c["success_smoothed"], dtype=np.float64))

        if not all_episodes:
            continue

        # Interpolate to common grid
        max_ep = max(float(ep[-1]) for ep in all_episodes)
        common_eps = np.linspace(0, max_ep, 500)
        interpolated = []
        for ep, sr in zip(all_episodes, all_success):
            interpolated.append(np.interp(common_eps, ep, sr))
        interpolated = np.array(interpolated)
        mean_sr = np.mean(interpolated, axis=0)
        std_sr = np.std(interpolated, axis=0)

        ax.plot(common_eps, mean_sr * 100, color=color, linewidth=2.0, label=label)
        ax.fill_between(common_eps, (mean_sr - std_sr) * 100, (mean_sr + std_sr) * 100,
                        color=color, alpha=0.15, linewidth=0)
        # Individual seeds as thin lines
        for i, (ep, sr) in enumerate(zip(all_episodes, all_success)):
            ax.plot(ep, np.array(sr) * 100, color=color, alpha=0.2, linewidth=0.7)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Training success rate: SDP vs norm-ball oracle", loc="left", fontsize=11, pad=10)
    ax.set_ylim(-2, 102)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    p = output_dir / "learning_curve_success_rate.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    plots_created.append(str(p))

    # --- Objective vs episode ---
    fig, ax = _make_axes(figsize=(10, 5.5))

    for curves, oracle_type in [(sdp_curves, "sdp"), (nb_curves, "norm_ball")]:
        color = ORACLE_COLORS[oracle_type]
        label = ORACLE_LABELS[oracle_type]

        all_episodes = []
        all_obj = []
        for c in curves:
            if not c or "episodes" not in c:
                continue
            eps = np.array(c["episodes"], dtype=np.float64)
            obj = np.array(c["avg_final_objective"], dtype=np.float64)
            # Filter NaN
            mask = np.isfinite(obj)
            if mask.sum() < 2:
                continue
            all_episodes.append(eps[mask])
            all_obj.append(obj[mask])

        if not all_episodes:
            continue

        max_ep = max(float(ep[-1]) for ep in all_episodes)
        common_eps = np.linspace(0, max_ep, 500)
        interpolated = []
        for ep, obj in zip(all_episodes, all_obj):
            interpolated.append(np.interp(common_eps, ep, obj))
        interpolated = np.array(interpolated)
        mean_obj = np.mean(interpolated, axis=0)
        std_obj = np.std(interpolated, axis=0)

        ax.plot(common_eps, mean_obj, color=color, linewidth=2.0, label=label)
        ax.fill_between(common_eps, mean_obj - std_obj, mean_obj + std_obj,
                        color=color, alpha=0.15, linewidth=0)
        for ep, obj in zip(all_episodes, all_obj):
            ax.plot(ep, obj, color=color, alpha=0.2, linewidth=0.7)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Avg final objective (normalized)")
    ax.set_title("Training objective: SDP vs norm-ball oracle", loc="left", fontsize=11, pad=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    p = output_dir / "learning_curve_objective.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    plots_created.append(str(p))

    # --- Success rate vs training steps (instead of episodes) ---
    fig, ax = _make_axes(figsize=(10, 5.5))

    for curves, oracle_type in [(sdp_curves, "sdp"), (nb_curves, "norm_ball")]:
        color = ORACLE_COLORS[oracle_type]
        label = ORACLE_LABELS[oracle_type]

        all_steps = []
        all_success = []
        for c in curves:
            if not c or "steps" not in c:
                continue
            all_steps.append(np.array(c["steps"], dtype=np.float64))
            all_success.append(np.array(c["success_smoothed"], dtype=np.float64))

        if not all_steps:
            continue

        max_step = max(float(s[-1]) for s in all_steps)
        common_steps = np.linspace(0, max_step, 500)
        interpolated = []
        for st, sr in zip(all_steps, all_success):
            interpolated.append(np.interp(common_steps, st, sr))
        interpolated = np.array(interpolated)
        mean_sr = np.mean(interpolated, axis=0)
        std_sr = np.std(interpolated, axis=0)

        ax.plot(common_steps, mean_sr * 100, color=color, linewidth=2.0, label=label)
        ax.fill_between(common_steps, (mean_sr - std_sr) * 100, (mean_sr + std_sr) * 100,
                        color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Training success rate vs steps: SDP vs norm-ball oracle", loc="left", fontsize=11, pad=10)
    ax.set_ylim(-2, 102)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    p = output_dir / "learning_curve_success_rate_vs_steps.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    plots_created.append(str(p))

    return plots_created


# ---------------------------------------------------------------------------
# Plotting: post-training evaluation comparison
# ---------------------------------------------------------------------------

def _plot_eval_comparison(
    *,
    sdp_aggregate: dict[str, Any],
    nb_aggregate: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Side-by-side bar charts comparing SDP vs norm-ball on eval metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_created: list[str] = []

    # --- Final objective comparison (rl_hidden_gradient only, bar chart) ---
    fig, ax = _make_axes(figsize=(9, 5.5))
    methods_to_plot = [m for m in META_METHOD_ORDER
                       if m in sdp_aggregate.get("methods", {}) or m in nb_aggregate.get("methods", {})]

    x = np.arange(len(methods_to_plot), dtype=np.float64)
    bar_width = 0.35

    sdp_means = []
    sdp_ci = []
    nb_means = []
    nb_ci = []
    labels = []
    for method in methods_to_plot:
        labels.append(METHOD_LABELS.get(method, method))
        sdp_data = sdp_aggregate.get("methods", {}).get(method, {})
        nb_data = nb_aggregate.get("methods", {}).get(method, {})
        sdp_stats = sdp_data.get("final_objective_stats", {})
        nb_stats = nb_data.get("final_objective_stats", {})
        sdp_means.append(float(sdp_stats.get("mean", float("nan"))))
        sdp_ci.append(float(sdp_stats.get("ci95", 0.0)))
        nb_means.append(float(nb_stats.get("mean", float("nan"))))
        nb_ci.append(float(nb_stats.get("ci95", 0.0)))

    ax.bar(x - bar_width / 2, sdp_means, bar_width, yerr=sdp_ci,
           color=ORACLE_COLORS["sdp"], alpha=0.8, label="SDP oracle",
           edgecolor="white", linewidth=0.8, capsize=3)
    ax.bar(x + bar_width / 2, nb_means, bar_width, yerr=nb_ci,
           color=ORACLE_COLORS["norm_ball"], alpha=0.8, label="Norm-ball oracle",
           edgecolor="white", linewidth=0.8, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Final objective (mean across seeds, 95% CI)")
    ax.set_title("Post-training evaluation: final objective by method", loc="left", fontsize=11, pad=10)
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = output_dir / "eval_final_objective_comparison.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    plots_created.append(str(p))

    # --- Success rate comparison ---
    fig, ax = _make_axes(figsize=(9, 5.5))

    sdp_sr = []
    nb_sr = []
    for method in methods_to_plot:
        sdp_data = sdp_aggregate.get("methods", {}).get(method, {})
        nb_data = nb_aggregate.get("methods", {}).get(method, {})
        sdp_sr.append(float(sdp_data.get("ever_success_rate_stats", {}).get("mean", 0.0)) * 100)
        nb_sr.append(float(nb_data.get("ever_success_rate_stats", {}).get("mean", 0.0)) * 100)

    ax.bar(x - bar_width / 2, sdp_sr, bar_width,
           color=ORACLE_COLORS["sdp"], alpha=0.8, label="SDP oracle",
           edgecolor="white", linewidth=0.8)
    ax.bar(x + bar_width / 2, nb_sr, bar_width,
           color=ORACLE_COLORS["norm_ball"], alpha=0.8, label="Norm-ball oracle",
           edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for i, (sv, nv) in enumerate(zip(sdp_sr, nb_sr)):
        if sv > 1:
            ax.text(i - bar_width / 2, sv + 1.5, f"{sv:.0f}%", ha="center", fontsize=8, color="#57606a")
        if nv > 1:
            ax.text(i + bar_width / 2, nv + 1.5, f"{nv:.0f}%", ha="center", fontsize=8, color="#57606a")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Post-training evaluation: success rate (ever reached) by method", loc="left", fontsize=11, pad=10)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    p = output_dir / "eval_success_rate_comparison.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    plots_created.append(str(p))

    # --- RL hidden gradient optimization curves overlay ---
    fig, ax = _make_axes(figsize=(10, 5.5))

    for agg, oracle_type in [(sdp_aggregate, "sdp"), (nb_aggregate, "norm_ball")]:
        rl_data = agg.get("methods", {}).get("rl_hidden_gradient", {})
        mean_curve = np.asarray(rl_data.get("grand_mean_curve", []), dtype=np.float64)
        std_curve = np.asarray(rl_data.get("grand_std_curve", []), dtype=np.float64)
        if mean_curve.size < 2:
            continue
        steps = np.arange(mean_curve.size)
        color = ORACLE_COLORS[oracle_type]
        label = ORACLE_LABELS[oracle_type]
        ax.plot(steps, mean_curve, color=color, linewidth=2.0, label=label)
        ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                        color=color, alpha=0.15, linewidth=0)

    # Also plot Adam for reference (from SDP aggregate since baselines are oracle-independent)
    adam_data = sdp_aggregate.get("methods", {}).get("adam", {})
    adam_mean = np.asarray(adam_data.get("grand_mean_curve", []), dtype=np.float64)
    if adam_mean.size >= 2:
        ax.plot(np.arange(adam_mean.size), adam_mean, color="#6b7280", linewidth=1.5,
                linestyle="--", label="Adam (baseline)", alpha=0.7)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Normalized objective")
    ax.set_title("RL hidden gradient: optimization curves by oracle type", loc="left", fontsize=11, pad=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    p = output_dir / "eval_rl_optimization_curves.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    plots_created.append(str(p))

    return plots_created


# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------

def _build_train_config(
    *,
    args: argparse.Namespace,
    seed: int,
    run_name: str,
    use_simple_s_star: bool,
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
        use_simple_s_star=use_simple_s_star,
    )


# ---------------------------------------------------------------------------
# Run one arm (SDP or norm-ball) across all seeds
# ---------------------------------------------------------------------------

def _run_arm(
    *,
    args: argparse.Namespace,
    oracle_type: str,
    seeds: list[int],
    arm_root: Path,
) -> dict[str, Any]:
    use_simple = oracle_type == "norm_ball"
    data_root = arm_root / "plot_data"
    data_root.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(args.device))
    seed_eval_payloads: list[dict[str, Any]] = []
    seed_learning_curves: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []

    for seed in seeds:
        run_name = f"oracle_cmp_{oracle_type}_Kmap{int(args.K_map)}_seed{int(seed)}"
        config = _build_train_config(
            args=args, seed=int(seed), run_name=run_name, use_simple_s_star=use_simple,
        )
        print(
            f"\n[oracle_comparison] {oracle_type} seed={int(seed)} "
            f"run={run_name} (train_steps={int(config.train_steps)})"
        )
        train_t0 = time.perf_counter()
        output = run_training(config, return_artifacts=True)
        train_elapsed_sec = float(time.perf_counter() - train_t0)

        run_dir = Path(str(output["summary"]["run_dir"])).expanduser().resolve()

        # Extract learning curves from training metrics
        lc = _extract_learning_curves(run_dir, window=int(args.running_avg_window))
        lc["seed"] = int(seed)
        lc["oracle_type"] = oracle_type
        seed_learning_curves.append(lc)

        lc_path = data_root / f"{oracle_type}_seed{int(seed)}_learning_curves.json"
        with lc_path.open("w", encoding="utf-8") as f:
            json.dump(lc, f, indent=2)

        # Post-training evaluation
        hidden_model = output["model"]
        hidden_env = output.get("env")
        no_oracle_model = output.get("no_oracle_model")
        no_oracle_env = output.get("no_oracle_env")
        visible_gradient_model = output.get("visible_gradient_model")
        visible_gradient_env = output.get("visible_gradient_env")

        eval_payload = _evaluate_seed(
            seed=int(seed), device=device,
            hidden_model=hidden_model, hidden_env=hidden_env,
            no_oracle_model=no_oracle_model, no_oracle_env=no_oracle_env,
            visible_gradient_model=visible_gradient_model, visible_gradient_env=visible_gradient_env,
            num_tasks=int(args.eval_num_tasks), horizon=int(args.eval_horizon),
        )
        eval_path = data_root / f"{oracle_type}_seed{int(seed)}_evaluation.json"
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2)
        seed_eval_payloads.append(eval_payload)

        # Read SDP/oracle info
        sdp_info_path = run_dir / "sdp_info.json"
        sdp_info = {}
        if sdp_info_path.exists():
            with sdp_info_path.open() as f:
                sdp_info = json.load(f)

        run_entries.append({
            "seed": int(seed),
            "oracle_type": oracle_type,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "training_wall_time_sec": train_elapsed_sec,
            "summary": output["summary"],
            "sdp_info": sdp_info,
            "learning_curves_json": str(lc_path),
            "evaluation_json": str(eval_path),
        })

        del output

    # Aggregate eval results
    aggregate = _aggregate_eval_results(seed_eval_payloads)
    aggregate["oracle_type"] = oracle_type
    agg_path = data_root / f"{oracle_type}_aggregate.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    return {
        "oracle_type": oracle_type,
        "use_simple_s_star": use_simple,
        "seeds": [int(s) for s in seeds],
        "runs": run_entries,
        "aggregate": aggregate,
        "aggregate_json": str(agg_path),
        "learning_curves": seed_learning_curves,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Compare SDP vs norm-ball oracle on alanine dipeptide (multi-seed)."
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seeds.")
    parser.add_argument("--suite_output_dir", type=str,
                        default="plots/oracle_comparison")
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
    parser.add_argument("--save_metrics_interval_episodes", type=int,
                        default=defaults.save_metrics_interval_episodes)
    parser.add_argument("--eval_interval_episodes", type=int,
                        default=defaults.eval_interval_episodes)
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
    parser.add_argument("--policy_arch", type=str, choices=["mlp", "gru"],
                        default=defaults.policy_arch)
    parser.add_argument("--disable_baseline_lr_tuning", action="store_true")
    parser.add_argument("--baseline_lr_candidates", type=str,
                        default=defaults.baseline_lr_candidates)
    parser.add_argument("--baseline_lr_tune_tasks", type=int,
                        default=defaults.baseline_lr_tune_tasks)

    # Evaluation config
    parser.add_argument("--eval_num_tasks", type=int, default=500,
                        help="Number of random-start evaluation tasks per seed.")
    parser.add_argument("--eval_horizon", type=int, default=defaults.max_horizon,
                        help="Optimization steps per evaluation task.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds, "seeds", min_value=0)

    suite_root = Path(args.suite_output_dir).expanduser().resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    plots_root = suite_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    print(f"Oracle comparison experiment: {len(seeds)} seeds, output -> {suite_root}")
    print(f"  Seeds: {seeds}")
    print(f"  K_map={args.K_map}, K_relax={args.K_relax}, train_steps={args.train_steps}")

    # Run both arms
    sdp_arm = _run_arm(args=args, oracle_type="sdp", seeds=seeds, arm_root=suite_root)
    nb_arm = _run_arm(args=args, oracle_type="norm_ball", seeds=seeds, arm_root=suite_root)

    # Save combined results
    combined = {
        "experiment": "oracle_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [int(s) for s in seeds],
        "K_map": int(args.K_map),
        "K_relax": int(args.K_relax),
        "train_steps": int(args.train_steps),
        "eval_num_tasks": int(args.eval_num_tasks),
        "eval_horizon": int(args.eval_horizon),
        "sdp": {
            "runs": sdp_arm["runs"],
            "aggregate": sdp_arm["aggregate"],
        },
        "norm_ball": {
            "runs": nb_arm["runs"],
            "aggregate": nb_arm["aggregate"],
        },
    }

    results_path = suite_root / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    # Generate comparison plots
    plot_paths: list[str] = []
    if not args.skip_plotting:
        plot_paths.extend(_plot_learning_curves_comparison(
            sdp_curves=sdp_arm["learning_curves"],
            nb_curves=nb_arm["learning_curves"],
            output_dir=plots_root,
        ))
        plot_paths.extend(_plot_eval_comparison(
            sdp_aggregate=sdp_arm["aggregate"],
            nb_aggregate=nb_arm["aggregate"],
            output_dir=plots_root,
        ))

    summary = {
        "suite_root": str(suite_root),
        "results_json": str(results_path),
        "num_seeds": len(seeds),
        "seeds": [int(s) for s in seeds],
        "plots": plot_paths,
        "sdp_rl_hidden_gradient_final_obj": sdp_arm["aggregate"].get("methods", {}).get(
            "rl_hidden_gradient", {}).get("final_objective_stats", {}),
        "norm_ball_rl_hidden_gradient_final_obj": nb_arm["aggregate"].get("methods", {}).get(
            "rl_hidden_gradient", {}).get("final_objective_stats", {}),
    }

    summary_path = suite_root / "suite_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("ORACLE COMPARISON COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
