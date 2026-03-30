import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Keep matplotlib fully headless and in writable cache locations.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from .config import TrainConfig
from .train import (
    _apply_spatial_task_snapshot,
    _capture_spatial_task_snapshot,
    _resolve_device,
    _rollout_spatial_adam_curve,
    _rollout_spatial_descent_curve,
    _rollout_spatial_policy_curve,
    maybe_save_spatial_trajectory_plot,
    rollout_spatial_basin_hopping_baseline,
    run_training,
)
from ..baselines import (
    rollout_cmaes_curve,
    rollout_jacobian_controller_curve,
    rollout_multistart_adam_curve,
    rollout_multistart_gd_curve,
)

META_METHOD_ORDER = (
    "gd",
    "adam",
    "basin_hop_gd",
    "random_search",
    "multistart_gd",
    "multistart_adam",
    "cmaes",
    "jacobian_controller",
    "rl_no_oracle",
    "rl_visible_oracle",
    "rl_hidden_gradient",
)

# Subplot figure `meta_optimizer_objective_vs_step_by_method.png` (excludes GD).
META_METHOD_ORDER_BY_METHOD_PANEL = tuple(m for m in META_METHOD_ORDER if m != "gd")

SEARCH_METHOD_ORDER = (
    "random_search",
    "rl_no_oracle",
    "rl_visible_oracle",
    "rl_hidden_gradient",
)

METHOD_LABELS = {
    "gd": "GD",
    "adam": "Adam",
    "basin_hop_gd": "Basin hopping + GD",
    "random_search": "Random search",
    "multistart_gd": "Multi-start GD",
    "multistart_adam": "Multi-start Adam",
    "cmaes": "CMA-ES",
    "jacobian_controller": "Jacobian controller",
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


def _parse_float_list(raw: str, arg_name: str, *, min_value: float | None = None) -> list[float]:
    values: list[float] = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"{arg_name} contains non-float token: {token!r}") from exc
        if min_value is not None and value < min_value:
            raise ValueError(f"{arg_name} must be >= {min_value}, got {value}")
        values.append(float(value))
    if not values:
        raise ValueError(f"{arg_name} must include at least one float")
    deduped = sorted(set(values))
    return deduped


def _method_colors(method_order: tuple[str, ...]) -> dict[str, tuple[float, float, float, float]]:
    palette = plt.cm.plasma(np.linspace(0.12, 0.88, max(1, len(method_order))))
    return {method: palette[idx] for idx, method in enumerate(method_order)}


def _method_colors_by_method_panel() -> dict[str, tuple[float, float, float, float]]:
    """Plasma keyed like full META_METHOD_ORDER but omitting GD's first (darkest) swatch."""
    base = plt.cm.plasma(np.linspace(0.12, 0.88, max(1, len(META_METHOD_ORDER))))
    sub = base[1 : 1 + len(META_METHOD_ORDER_BY_METHOD_PANEL)]
    return {m: sub[i] for i, m in enumerate(META_METHOD_ORDER_BY_METHOD_PANEL)}


def _style_meta_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d7de")
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(colors="#57606a")
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.minorticks_on()


def _make_axes(figsize: tuple[float, float] = (8.6, 5.0)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    _style_meta_axis(ax)
    return fig, ax


def _value_tag(value: int | float) -> str:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-12:
        return str(int(round(numeric)))
    return str(numeric).replace(".", "p").replace("-", "m")


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


@torch.no_grad()
def _rollout_spatial_policy_episode_stats(
    *,
    model: torch.nn.Module,
    env,
    device: torch.device,
    start_xy: np.ndarray,
    horizon: int,
    env_index: int = 0,
    deterministic: bool,
) -> tuple[float, float]:
    state = start_xy.astype(np.float32).copy()
    h = max(1, int(horizon))
    hidden_state = model.initial_state(batch_size=1, device=device)
    best_objective = float(env._normalized_objective_value(state, env_index=env_index))

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
        objective_now = float(env._normalized_objective_value(state, env_index=env_index))
        if objective_now < best_objective:
            best_objective = objective_now

    final_objective = float(env._normalized_objective_value(state, env_index=env_index))
    return final_objective, best_objective


@torch.no_grad()
def _rollout_spatial_policy_search_curve(
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


def _rollout_random_search_episode_stats(
    *,
    env,
    start_xy: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    env_index: int = 0,
) -> tuple[float, float]:
    h = max(1, int(horizon))
    best_objective = float(
        env._normalized_objective_value(start_xy.astype(np.float32), env_index=env_index)
    )
    for _ in range(h):
        candidate = rng.uniform(
            low=-float(env.coord_limit),
            high=float(env.coord_limit),
            size=int(env.visible_dim),
        ).astype(np.float32)
        candidate_objective = float(env._normalized_objective_value(candidate, env_index=env_index))
        if candidate_objective < best_objective:
            best_objective = candidate_objective
    # Random search returns the best state found under the budget.
    return best_objective, best_objective


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
        candidate = rng.uniform(
            low=-float(env.coord_limit),
            high=float(env.coord_limit),
            size=int(env.visible_dim),
        ).astype(np.float32)
        candidate_objective = float(env._normalized_objective_value(candidate, env_index=env_index))
        if candidate_objective < best_objective:
            best_objective = candidate_objective
        curve[step + 1] = best_objective
    return curve


def _rollout_basin_hopping_curve(
    *,
    env,
    start_xy: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    env_index: int = 0,
    base_lr: float | None = None,
    local_steps: int = 8,
    jump_scale: float = 1.0,
) -> np.ndarray:
    trajectory = rollout_spatial_basin_hopping_baseline(
        env=env,
        start_xy=start_xy,
        horizon=horizon,
        base_lr=base_lr,
        local_steps=local_steps,
        jump_scale=jump_scale,
        rng=rng,
        stop_on_success=False,
    )
    return np.asarray(
        [
            env._normalized_objective_value(point.astype(np.float32), env_index=env_index)
            for point in trajectory
        ],
        dtype=np.float32,
    )


def _sample_meta_task_snapshots(
    *,
    env,
    num_tasks: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for _ in range(max(1, int(num_tasks))):
        spec = env.sample_episode_spec(
            env_index=0,
            refresh_map=env.refresh_map_each_episode,
        )
        snapshot = _capture_spatial_task_snapshot(
            env=env,
            source_xy=spec.source,
            target_min_xy=spec.target_min_xy,
            env_index=0,
        )
        tasks.append(
            {
                "snapshot": snapshot,
                "start_xy": spec.source.copy().astype(np.float32),
                "horizon": int(spec.horizon),
            }
        )
    return tasks


def _score_meta_basin_hopping_candidate(
    *,
    env,
    tasks: list[dict[str, Any]],
    base_lr: float,
    local_steps: int,
    jump_scale: float,
    seed: int,
) -> dict[str, float]:
    final_values: list[float] = []
    best_values: list[float] = []
    successes: list[float] = []
    combo_seed = int(seed) + 70_021 + int(local_steps) * 101 + int(round(jump_scale * 1000.0)) * 17
    rng = np.random.default_rng(combo_seed)

    for task in tasks:
        _apply_spatial_task_snapshot(env, task["snapshot"], env_index=0)
        curve = _rollout_basin_hopping_curve(
            env=env,
            start_xy=np.asarray(task["start_xy"], dtype=np.float32),
            horizon=int(task["horizon"]),
            rng=rng,
            env_index=0,
            base_lr=float(base_lr),
            local_steps=int(local_steps),
            jump_scale=float(jump_scale),
        )
        final_value = float(curve[-1])
        best_value = float(np.min(curve))
        final_values.append(final_value)
        best_values.append(best_value)
        successes.append(1.0 if best_value <= float(env.success_threshold) else 0.0)

    avg_final_objective = float(np.mean(final_values)) if final_values else float("inf")
    avg_best_objective = float(np.mean(best_values)) if best_values else float("inf")
    success_rate = float(np.mean(successes)) if successes else 0.0
    return {
        "local_steps": int(local_steps),
        "jump_scale": float(jump_scale),
        "avg_final_objective": avg_final_objective,
        "avg_best_objective": avg_best_objective,
        "success_rate": success_rate,
    }


def tune_meta_basin_hopping(
    *,
    env,
    seed: int,
    num_tasks: int,
    local_steps_candidates: list[int],
    jump_scale_candidates: list[float],
) -> dict[str, Any]:
    tasks = _sample_meta_task_snapshots(env=env, num_tasks=max(1, int(num_tasks)))
    base_lr = float(env.baseline_lr_gd)
    scores: list[dict[str, float]] = []
    for local_steps in local_steps_candidates:
        for jump_scale in jump_scale_candidates:
            scores.append(
                _score_meta_basin_hopping_candidate(
                    env=env,
                    tasks=tasks,
                    base_lr=base_lr,
                    local_steps=int(local_steps),
                    jump_scale=float(jump_scale),
                    seed=int(seed),
                )
            )

    best = min(
        scores,
        key=lambda row: (
            float(row["avg_final_objective"]),
            float(row["avg_best_objective"]),
            -float(row["success_rate"]),
            float(row["jump_scale"]),
            int(row["local_steps"]),
        ),
    )
    return {
        "num_tasks": int(len(tasks)),
        "base_lr": base_lr,
        "local_steps_candidates": [int(v) for v in local_steps_candidates],
        "jump_scale_candidates": [float(v) for v in jump_scale_candidates],
        "best_local_steps": int(best["local_steps"]),
        "best_jump_scale": float(best["jump_scale"]),
        "scores": scores,
    }


def _build_search_step_budgets(*, max_steps: int, num_points: int) -> list[int]:
    total_steps = max(1, int(max_steps))
    points = max(16, int(num_points))
    dense_prefix = list(range(1, min(total_steps, 32) + 1))
    if total_steps <= len(dense_prefix):
        return dense_prefix
    geometric = np.geomspace(1.0, float(total_steps), num=points)
    budgets = {
        int(np.clip(int(round(value)), 1, total_steps))
        for value in geometric
    }
    budgets.update(dense_prefix)
    budgets.add(total_steps)
    return sorted(budgets)


def _evaluate_meta_optimizer_seed(
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
    basin_hop_local_steps: int,
    basin_hop_jump_scale: float,
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
    basin_hop_rng = np.random.default_rng(int(seed) + 60_019)
    multistart_rng = np.random.default_rng(int(seed) + 70_031)
    cmaes_rng = np.random.default_rng(int(seed) + 80_041)

    for _ in range(max(1, int(num_tasks))):
        spec = hidden_env.sample_episode_spec(
            env_index=0,
            refresh_map=hidden_env.refresh_map_each_episode,
        )
        snapshot = _capture_spatial_task_snapshot(
            env=hidden_env,
            source_xy=spec.source,
            target_min_xy=spec.target_min_xy,
            env_index=0,
        )
        start_xy = snapshot["source_xy"].astype(np.float32)

        _apply_spatial_task_snapshot(hidden_env, snapshot, env_index=0)
        gd_curve = _rollout_spatial_descent_curve(
            env=hidden_env,
            start_xy=start_xy,
            horizon=horizon,
            env_index=0,
            base_lr=hidden_env.baseline_lr_gd,
        )
        method_curves["gd"].append(gd_curve)

        adam_curve = _rollout_spatial_adam_curve(
            env=hidden_env,
            start_xy=start_xy,
            horizon=horizon,
            env_index=0,
            base_lr=hidden_env.baseline_lr_adam,
        )
        method_curves["adam"].append(adam_curve)

        basin_hop_curve = _rollout_basin_hopping_curve(
            env=hidden_env,
            start_xy=start_xy,
            horizon=horizon,
            rng=basin_hop_rng,
            env_index=0,
            base_lr=hidden_env.baseline_lr_gd,
            local_steps=basin_hop_local_steps,
            jump_scale=basin_hop_jump_scale,
        )
        method_curves["basin_hop_gd"].append(basin_hop_curve)

        random_search_curve = _rollout_random_search_curve(
            env=hidden_env,
            start_xy=start_xy,
            horizon=horizon,
            rng=random_rng,
            env_index=0,
        )
        method_curves["random_search"].append(random_search_curve)

        # Multi-start GD (5 restarts)
        multistart_gd_curve = rollout_multistart_gd_curve(
            grad_fn=lambda x: hidden_env._gradient_xy(x, env_index=0),
            obj_fn=lambda x: hidden_env._normalized_objective_value(x, env_index=0),
            clip_fn=lambda x: np.clip(x, -hidden_env.coord_limit, hidden_env.coord_limit).astype(np.float32),
            start_xy=start_xy,
            horizon=horizon,
            base_lr=hidden_env.baseline_lr_gd,
            n_restarts=5,
            rng=multistart_rng,
            domain_low=-hidden_env.coord_limit,
            domain_high=hidden_env.coord_limit,
        )
        method_curves["multistart_gd"].append(multistart_gd_curve)

        # Multi-start Adam (5 restarts)
        multistart_adam_curve = rollout_multistart_adam_curve(
            grad_fn=lambda x: hidden_env._gradient_xy(x, env_index=0),
            obj_fn=lambda x: hidden_env._normalized_objective_value(x, env_index=0),
            clip_fn=lambda x: np.clip(x, -hidden_env.coord_limit, hidden_env.coord_limit).astype(np.float32),
            start_xy=start_xy,
            horizon=horizon,
            base_lr=hidden_env.baseline_lr_adam,
            n_restarts=5,
            rng=multistart_rng,
            domain_low=-hidden_env.coord_limit,
            domain_high=hidden_env.coord_limit,
        )
        method_curves["multistart_adam"].append(multistart_adam_curve)

        # CMA-ES
        cmaes_curve = rollout_cmaes_curve(
            obj_fn=lambda x: hidden_env._normalized_objective_value(x, env_index=0),
            start_xy=start_xy,
            horizon=horizon,
            bounds_low=-hidden_env.coord_limit,
            bounds_high=hidden_env.coord_limit,
            sigma0=hidden_env.coord_limit / 3.0,
        )
        method_curves["cmaes"].append(cmaes_curve)

        # Jacobian controller (uses hidden gradient + Jacobian, no RL)
        jacobian_curve = rollout_jacobian_controller_curve(
            hidden_grad_fn=lambda x: hidden_env._gradient_hidden(x, env_index=0),
            jacobian_fn=lambda x: hidden_env._jacobian(x, env_index=0),
            obj_fn=lambda x: hidden_env._normalized_objective_value(x, env_index=0),
            clip_fn=lambda x: np.clip(x, -hidden_env.coord_limit, hidden_env.coord_limit).astype(np.float32),
            start_xy=start_xy,
            horizon=horizon,
            base_lr=hidden_env.baseline_lr_gd,
        )
        method_curves["jacobian_controller"].append(jacobian_curve)

        hidden_curve = _rollout_spatial_policy_curve(
            model=hidden_model,
            env=hidden_env,
            device=device,
            start_xy=start_xy,
            horizon=horizon,
            env_index=0,
        )
        method_curves["rl_hidden_gradient"].append(hidden_curve)

        if no_oracle_model is not None and no_oracle_env is not None:
            _apply_spatial_task_snapshot(no_oracle_env, snapshot, env_index=0)
            no_curve = _rollout_spatial_policy_curve(
                model=no_oracle_model,
                env=no_oracle_env,
                device=device,
                start_xy=start_xy,
                horizon=horizon,
                env_index=0,
            )
            method_curves["rl_no_oracle"].append(no_curve)

        if visible_gradient_model is not None and visible_gradient_env is not None:
            _apply_spatial_task_snapshot(visible_gradient_env, snapshot, env_index=0)
            visible_curve = _rollout_spatial_policy_curve(
                model=visible_gradient_model,
                env=visible_gradient_env,
                device=device,
                start_xy=start_xy,
                horizon=horizon,
                env_index=0,
            )
            method_curves["rl_visible_oracle"].append(visible_curve)

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
        best_so_far = np.minimum.accumulate(curves, axis=1)
        mean_curve = np.nanmean(best_so_far, axis=0)
        std_curve = np.nanstd(best_so_far, axis=0)
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


def _run_step_budgeted_search(
    *,
    trajectory_curve_fn: Callable[[], np.ndarray],
    step_budgets: list[int],
    max_episodes: int,
    success_threshold: float | None = None,
) -> dict[str, Any]:
    ordered_budgets = sorted(set(int(v) for v in step_budgets if int(v) > 0))
    if not ordered_budgets:
        raise ValueError("step_budgets must contain at least one positive integer")

    max_budget = int(max(ordered_budgets))
    episode_records: list[dict[str, Any]] = []
    budget_results: dict[str, Any] = {}
    budget_index = 0
    cumulative_steps = 0
    cumulative_elapsed = 0.0
    global_best: float | None = None
    first_success_steps: int | None = None
    first_success_wall_time_sec: float | None = None
    first_success_episodes_used: int | None = None

    while (cumulative_steps < max_budget or not episode_records) and len(episode_records) < max_episodes:
        t0 = time.perf_counter()
        objective_curve = np.asarray(trajectory_curve_fn(), dtype=np.float64).reshape(-1)
        elapsed = float(time.perf_counter() - t0)
        if objective_curve.size < 2:
            raise ValueError("trajectory_curve_fn must return at least two objective values")

        best_curve = np.minimum.accumulate(objective_curve)
        steps_in_episode = int(best_curve.size - 1)
        elapsed_per_step = elapsed / float(max(1, steps_in_episode))
        if global_best is None:
            global_best = float(best_curve[0])
        if (
            success_threshold is not None
            and np.isfinite(float(global_best))
            and float(global_best) <= float(success_threshold)
        ):
            first_success_steps = 0
            first_success_wall_time_sec = 0.0
            first_success_episodes_used = 0
            for remaining_budget in ordered_budgets:
                budget_results[_value_tag(remaining_budget)] = {
                    "budget_steps": int(remaining_budget),
                    "episodes_used": 0,
                    "steps_used": 0,
                    "wall_time_used_sec": 0.0,
                    "best_objective": float(global_best),
                }
            return {
                "budget_mode": "environment_steps",
                "max_episodes": int(max_episodes),
                "step_budgets": [int(v) for v in ordered_budgets],
                "episodes_collected": 0,
                "episode_records": episode_records,
                "step_budget_results": budget_results,
                "first_success_steps": int(first_success_steps),
                "first_success_wall_time_sec": float(first_success_wall_time_sec),
                "first_success_episodes_used": int(first_success_episodes_used),
            }

        for step_idx in range(1, best_curve.size):
            cumulative_steps += 1
            cumulative_elapsed += elapsed_per_step
            global_best = min(float(global_best), float(best_curve[step_idx]))
            while budget_index < len(ordered_budgets) and cumulative_steps >= ordered_budgets[budget_index]:
                budget_steps = int(ordered_budgets[budget_index])
                budget_results[_value_tag(budget_steps)] = {
                    "budget_steps": budget_steps,
                    "episodes_used": int(len(episode_records) + 1),
                    "steps_used": budget_steps,
                    "wall_time_used_sec": float(cumulative_elapsed),
                    "best_objective": float(global_best),
                }
                budget_index += 1

            if (
                success_threshold is not None
                and np.isfinite(float(global_best))
                and float(global_best) <= float(success_threshold)
            ):
                first_success_steps = int(cumulative_steps)
                first_success_wall_time_sec = float(cumulative_elapsed)
                first_success_episodes_used = int(len(episode_records) + 1)
                episode_records.append(
                    {
                        "episode_index": int(len(episode_records)),
                        "elapsed_sec": float(first_success_wall_time_sec),
                        "cumulative_elapsed_sec": float(first_success_wall_time_sec),
                        "steps": int(step_idx),
                        "cumulative_steps": int(first_success_steps),
                        "final_objective": float(best_curve[step_idx]),
                        "best_objective": float(global_best),
                        "search_best_objective_after_episode": float(global_best),
                    }
                )
                for remaining_budget in ordered_budgets[budget_index:]:
                    budget_results[_value_tag(remaining_budget)] = {
                        "budget_steps": int(remaining_budget),
                        "episodes_used": int(first_success_episodes_used),
                        "steps_used": int(first_success_steps),
                        "wall_time_used_sec": float(first_success_wall_time_sec),
                        "best_objective": float(global_best),
                    }
                return {
                    "budget_mode": "environment_steps",
                    "max_episodes": int(max_episodes),
                    "step_budgets": [int(v) for v in ordered_budgets],
                    "episodes_collected": int(len(episode_records)),
                    "episode_records": episode_records,
                    "step_budget_results": budget_results,
                    "first_success_steps": int(first_success_steps),
                    "first_success_wall_time_sec": float(first_success_wall_time_sec),
                    "first_success_episodes_used": int(first_success_episodes_used),
                }

        episode_records.append(
            {
                "episode_index": int(len(episode_records)),
                "elapsed_sec": elapsed,
                "cumulative_elapsed_sec": float(cumulative_elapsed),
                "steps": int(steps_in_episode),
                "cumulative_steps": int(cumulative_steps),
                "final_objective": float(objective_curve[-1]),
                "best_objective": float(best_curve[-1]),
                "search_best_objective_after_episode": float(global_best),
            }
        )

    if not episode_records:
        raise RuntimeError("No episodes were collected for step-budgeted search evaluation")

    if budget_index < len(ordered_budgets):
        for remaining_budget in ordered_budgets[budget_index:]:
            budget_results[_value_tag(remaining_budget)] = {
                "budget_steps": int(remaining_budget),
                "episodes_used": int(len(episode_records)),
                "steps_used": int(cumulative_steps),
                "wall_time_used_sec": float(cumulative_elapsed),
                "best_objective": float(global_best if global_best is not None else float("nan")),
            }

    return {
        "budget_mode": "environment_steps",
        "max_episodes": int(max_episodes),
        "step_budgets": [int(v) for v in ordered_budgets],
        "episodes_collected": int(len(episode_records)),
        "episode_records": episode_records,
        "step_budget_results": budget_results,
        "first_success_steps": (
            int(first_success_steps) if first_success_steps is not None else None
        ),
        "first_success_wall_time_sec": (
            float(first_success_wall_time_sec) if first_success_wall_time_sec is not None else None
        ),
        "first_success_episodes_used": (
            int(first_success_episodes_used) if first_success_episodes_used is not None else None
        ),
    }


def _build_step_budget_results_from_training_trace(
    *,
    method_trace: dict[str, Any],
    step_budgets: list[int],
) -> dict[str, Any]:
    ordered_budgets = sorted(set(int(v) for v in step_budgets if int(v) > 0))
    if not ordered_budgets:
        raise ValueError("step_budgets must contain at least one positive integer")

    raw_records = method_trace.get("records", [])
    processed_records: list[dict[str, Any]] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        steps_used = int(raw.get("steps_used", 0))
        if steps_used <= 0:
            continue
        best_objective = float(raw.get("best_objective", float("nan")))
        wall_time_used_sec = float(raw.get("wall_time_used_sec", float("nan")))
        episodes_used = int(raw.get("episodes_used", 0))
        processed_records.append(
            {
                "steps_used": steps_used,
                "best_objective": best_objective,
                "wall_time_used_sec": wall_time_used_sec,
                "episodes_used": episodes_used,
                "success_found": bool(raw.get("success_found", False)),
            }
        )

    processed_records.sort(key=lambda item: int(item["steps_used"]))
    budget_results: dict[str, Any] = {}
    if not processed_records:
        return {
            "budget_mode": "environment_steps",
            "source": "training_trace",
            "step_budgets": [int(v) for v in ordered_budgets],
            "episodes_collected": 0,
            "episode_records": [],
            "step_budget_results": budget_results,
            "first_success_steps": None,
            "first_success_wall_time_sec": None,
            "first_success_episodes_used": None,
        }

    first_success_steps_raw = method_trace.get("first_success_steps")
    first_success_wall_raw = method_trace.get("first_success_wall_time_sec")
    first_success_eps_raw = method_trace.get("first_success_episodes_used")

    first_success_record: dict[str, Any] | None = None
    if first_success_steps_raw is not None:
        first_success_steps = int(first_success_steps_raw)
        for record in processed_records:
            if int(record["steps_used"]) >= first_success_steps:
                first_success_record = {
                    "steps_used": int(first_success_steps),
                    "best_objective": float(record["best_objective"]),
                    "wall_time_used_sec": (
                        float(first_success_wall_raw)
                        if first_success_wall_raw is not None
                        else float(record["wall_time_used_sec"])
                    ),
                    "episodes_used": (
                        int(first_success_eps_raw)
                        if first_success_eps_raw is not None
                        else int(record["episodes_used"])
                    ),
                    "success_found": True,
                }
                break
    current_record: dict[str, Any] | None = None
    record_index = 0
    for budget in ordered_budgets:
        budget_int = int(budget)
        if first_success_record is not None and budget_int >= int(first_success_record["steps_used"]):
            source_record = first_success_record
        else:
            while (
                record_index < len(processed_records)
                and int(processed_records[record_index]["steps_used"]) <= budget_int
            ):
                current_record = processed_records[record_index]
                record_index += 1
            source_record = current_record
        if source_record is None:
            continue
        budget_results[_value_tag(budget_int)] = {
            "budget_steps": budget_int,
            "episodes_used": int(source_record["episodes_used"]),
            "steps_used": int(source_record["steps_used"]),
            "wall_time_used_sec": float(source_record["wall_time_used_sec"]),
            "best_objective": float(source_record["best_objective"]),
        }

    return {
        "budget_mode": "environment_steps",
        "source": "training_trace",
        "step_budgets": [int(v) for v in ordered_budgets],
        "episodes_collected": int(processed_records[-1]["episodes_used"]),
        "episode_records": processed_records,
        "step_budget_results": budget_results,
        "first_success_steps": (
            int(first_success_record["steps_used"]) if first_success_record is not None else None
        ),
        "first_success_wall_time_sec": (
            float(first_success_record["wall_time_used_sec"])
            if first_success_record is not None
            else None
        ),
        "first_success_episodes_used": (
            int(first_success_record["episodes_used"]) if first_success_record is not None else None
        ),
    }


def _build_training_time_search_seed_payload(
    *,
    seed: int,
    hidden_env,
    training_trace: dict[str, Any],
    step_budgets: list[int],
    horizon: int,
) -> dict[str, Any]:
    snapshot = _capture_spatial_task_snapshot(
        env=hidden_env,
        source_xy=hidden_env.initial_xy[0],
        target_min_xy=hidden_env.reference_min_xy_env[0],
        env_index=0,
    )
    start_xy = snapshot["source_xy"].astype(np.float32).copy()
    _apply_spatial_task_snapshot(hidden_env, snapshot, env_index=0)

    success_threshold = float(
        training_trace.get("success_threshold", getattr(hidden_env, "success_threshold", 0.01))
    )
    methods_payload: dict[str, Any] = {}
    random_rng = np.random.default_rng(int(seed) + 90_013)

    def random_search_curve() -> np.ndarray:
        return _rollout_random_search_curve(
            env=hidden_env,
            start_xy=start_xy,
            horizon=horizon,
            rng=random_rng,
            env_index=0,
        )

    methods_payload["random_search"] = _run_step_budgeted_search(
        trajectory_curve_fn=random_search_curve,
        step_budgets=step_budgets,
        max_episodes=max(1, int(np.ceil(max(step_budgets) / max(1, horizon)))),
        success_threshold=success_threshold,
    )

    trace_methods = training_trace.get("methods", {})
    if isinstance(trace_methods, dict):
        hidden_trace = trace_methods.get("rl_hidden_gradient")
        if isinstance(hidden_trace, dict):
            methods_payload["rl_hidden_gradient"] = _build_step_budget_results_from_training_trace(
                method_trace=hidden_trace,
                step_budgets=step_budgets,
            )
        no_oracle_trace = trace_methods.get("rl_no_oracle")
        if isinstance(no_oracle_trace, dict):
            methods_payload["rl_no_oracle"] = _build_step_budget_results_from_training_trace(
                method_trace=no_oracle_trace,
                step_budgets=step_budgets,
            )
        visible_trace = trace_methods.get("rl_visible_oracle")
        if isinstance(visible_trace, dict):
            methods_payload["rl_visible_oracle"] = _build_step_budget_results_from_training_trace(
                method_trace=visible_trace,
                step_budgets=step_budgets,
            )

    return {
        "seed": int(seed),
        "horizon": int(horizon),
        "method_order": list(SEARCH_METHOD_ORDER),
        "budget_mode": "environment_steps",
        "search_mode": "training_time",
        "step_budgets": [int(v) for v in sorted(set(int(x) for x in step_budgets))],
        "success_threshold": float(success_threshold),
        "fixed_task": {
            "source_xy": [float(v) for v in snapshot["source_xy"]],
            "target_min_xy": [float(v) for v in snapshot["target_min_xy"]],
        },
        "methods": methods_payload,
    }


def _evaluate_search_seed(
    *,
    seed: int,
    device: torch.device,
    hidden_model: torch.nn.Module,
    hidden_env,
    no_oracle_model: torch.nn.Module | None,
    no_oracle_env,
    visible_gradient_model: torch.nn.Module | None,
    visible_gradient_env,
    step_budgets: list[int],
    horizon: int,
    max_episodes_per_method: int,
    rl_deterministic: bool,
) -> dict[str, Any]:
    snapshot = _capture_spatial_task_snapshot(
        env=hidden_env,
        source_xy=hidden_env.initial_xy[0],
        target_min_xy=hidden_env.reference_min_xy_env[0],
        env_index=0,
    )
    start_xy = snapshot["source_xy"].astype(np.float32).copy()
    _apply_spatial_task_snapshot(hidden_env, snapshot, env_index=0)
    if no_oracle_env is not None:
        _apply_spatial_task_snapshot(no_oracle_env, snapshot, env_index=0)
    if visible_gradient_env is not None:
        _apply_spatial_task_snapshot(visible_gradient_env, snapshot, env_index=0)

    methods_payload: dict[str, Any] = {}
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

    random_rng = np.random.default_rng(int(seed) + 90_013)

    def random_search_curve() -> np.ndarray:
        return _rollout_random_search_curve(
            env=hidden_env,
            start_xy=start_xy,
            horizon=horizon,
            rng=random_rng,
            env_index=0,
        )

    methods_payload["random_search"] = _run_step_budgeted_search(
        trajectory_curve_fn=random_search_curve,
        step_budgets=step_budgets,
        max_episodes=max_episodes_per_method,
    )

    def hidden_curve() -> np.ndarray:
        return _rollout_spatial_policy_search_curve(
            model=hidden_model,
            env=hidden_env,
            device=device,
            start_xy=start_xy,
            horizon=horizon,
            env_index=0,
            deterministic=rl_deterministic,
        ).astype(np.float64)

    methods_payload["rl_hidden_gradient"] = _run_step_budgeted_search(
        trajectory_curve_fn=hidden_curve,
        step_budgets=step_budgets,
        max_episodes=max_episodes_per_method,
    )

    if no_oracle_model is not None and no_oracle_env is not None:
        def no_oracle_curve() -> np.ndarray:
            return _rollout_spatial_policy_search_curve(
                model=no_oracle_model,
                env=no_oracle_env,
                device=device,
                start_xy=start_xy,
                horizon=horizon,
                env_index=0,
                deterministic=rl_deterministic,
            ).astype(np.float64)

        methods_payload["rl_no_oracle"] = _run_step_budgeted_search(
            trajectory_curve_fn=no_oracle_curve,
            step_budgets=step_budgets,
            max_episodes=max_episodes_per_method,
        )

    if visible_gradient_model is not None and visible_gradient_env is not None:
        def visible_curve() -> np.ndarray:
            return _rollout_spatial_policy_search_curve(
                model=visible_gradient_model,
                env=visible_gradient_env,
                device=device,
                start_xy=start_xy,
                horizon=horizon,
                env_index=0,
                deterministic=rl_deterministic,
            ).astype(np.float64)

        methods_payload["rl_visible_oracle"] = _run_step_budgeted_search(
            trajectory_curve_fn=visible_curve,
            step_budgets=step_budgets,
            max_episodes=max_episodes_per_method,
        )

    if hidden_was_training:
        hidden_model.train()
    if no_oracle_model is not None and bool(no_oracle_was_training):
        no_oracle_model.train()
    if visible_gradient_model is not None and bool(visible_was_training):
        visible_gradient_model.train()

    return {
        "seed": int(seed),
        "horizon": int(horizon),
        "method_order": list(SEARCH_METHOD_ORDER),
        "budget_mode": "environment_steps",
        "step_budgets": [int(v) for v in sorted(set(int(x) for x in step_budgets))],
        "rl_deterministic": bool(rl_deterministic),
        "fixed_task": {
            "source_xy": [float(v) for v in snapshot["source_xy"]],
            "target_min_xy": [float(v) for v in snapshot["target_min_xy"]],
        },
        "methods": methods_payload,
    }


def _aggregate_meta_results(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_method_curves: dict[str, list[np.ndarray]] = {method: [] for method in META_METHOD_ORDER}
    by_method_seed_final_mean: dict[str, list[float]] = {method: [] for method in META_METHOD_ORDER}
    by_method_seed_best_mean: dict[str, list[float]] = {method: [] for method in META_METHOD_ORDER}
    by_method_all_final: dict[str, list[float]] = {method: [] for method in META_METHOD_ORDER}
    by_method_all_best: dict[str, list[float]] = {method: [] for method in META_METHOD_ORDER}

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

    methods_payload: dict[str, Any] = {}
    for method in META_METHOD_ORDER:
        if not by_method_curves[method]:
            continue
        stacked = np.concatenate(by_method_curves[method], axis=0)
        # Per-step plot: mean best-so-far along each task's trajectory. Raw
        # GD/Adam/basin/PPO rollouts are non-monotone; averaging instantaneous
        # values across tasks hides progress (out-of-phase oscillation). Random
        # search is already non-increasing, so accumulate is a no-op there.
        best_so_far = np.minimum.accumulate(stacked, axis=1)
        mean_curve = np.mean(best_so_far, axis=0)
        std_curve = np.std(best_so_far, axis=0)
        methods_payload[method] = {
            "label": METHOD_LABELS.get(method, method),
            "num_total_tasks": int(stacked.shape[0]),
            "num_seeds": int(len(by_method_seed_final_mean[method])),
            "mean_curve": [float(v) for v in mean_curve],
            "std_curve": [float(v) for v in std_curve],
            "seed_mean_final_objective_values": [float(v) for v in by_method_seed_final_mean[method]],
            "seed_mean_best_objective_values": [float(v) for v in by_method_seed_best_mean[method]],
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


def _stack_meta_per_method_best_so_far_curves(
    seed_payloads: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Concatenate per-seed task curves and apply per-task best-so-far (plotting metric)."""
    by_method: dict[str, list[np.ndarray]] = {m: [] for m in META_METHOD_ORDER}
    for payload in seed_payloads:
        for method in META_METHOD_ORDER:
            method_data = payload.get("methods", {}).get(method)
            if method_data is None:
                continue
            task_curves = np.asarray(method_data.get("task_curves", []), dtype=np.float64)
            if task_curves.ndim == 2 and task_curves.shape[0] > 0 and task_curves.shape[1] >= 2:
                by_method[method].append(task_curves)
    out: dict[str, np.ndarray] = {}
    for method in META_METHOD_ORDER:
        if not by_method[method]:
            continue
        stacked = np.concatenate(by_method[method], axis=0)
        out[method] = np.minimum.accumulate(stacked, axis=1)
    return out


def _aggregate_search_results(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    budgets: list[int] = []
    search_mode = None
    for payload in seed_payloads:
        budgets.extend(int(v) for v in payload.get("step_budgets", []))
        if search_mode is None and payload.get("search_mode") is not None:
            search_mode = str(payload.get("search_mode"))
    unique_budgets = sorted(set(budgets))
    budget_tags = [_value_tag(value) for value in unique_budgets]

    methods_payload: dict[str, Any] = {}
    for method in SEARCH_METHOD_ORDER:
        budget_rows: dict[str, dict[str, list[float]]] = {
            tag: {
                "seed_best_objective_values": [],
                "seed_wall_time_used_sec": [],
                "seed_episodes_used": [],
            }
            for tag in budget_tags
        }
        first_success_steps_values: list[float] = []
        first_success_wall_time_values: list[float] = []
        first_success_episodes_values: list[float] = []

        for payload in seed_payloads:
            method_data = payload.get("methods", {}).get(method)
            if method_data is None:
                continue
            budget_results = method_data.get("step_budget_results", {})
            first_success_steps = method_data.get("first_success_steps")
            first_success_wall_time = method_data.get("first_success_wall_time_sec")
            first_success_episodes = method_data.get("first_success_episodes_used")
            if first_success_steps is not None:
                first_success_steps_values.append(float(first_success_steps))
            if first_success_wall_time is not None:
                first_success_wall_time_values.append(float(first_success_wall_time))
            if first_success_episodes is not None:
                first_success_episodes_values.append(float(first_success_episodes))
            for budget in unique_budgets:
                tag = _value_tag(budget)
                row = budget_results.get(tag)
                if row is None:
                    continue
                budget_rows[tag]["seed_best_objective_values"].append(float(row["best_objective"]))
                budget_rows[tag]["seed_wall_time_used_sec"].append(float(row["wall_time_used_sec"]))
                budget_rows[tag]["seed_episodes_used"].append(float(row["episodes_used"]))

        if not any(
            budget_rows[tag]["seed_best_objective_values"] for tag in budget_rows.keys()
        ):
            continue

        budgets_payload: dict[str, Any] = {}
        for budget in unique_budgets:
            tag = _value_tag(budget)
            budgets_payload[tag] = {
                "budget_steps": int(budget),
                "best_objective": _stats(
                    budget_rows[tag]["seed_best_objective_values"]
                ),
                "wall_time_used_sec": _stats(
                    budget_rows[tag]["seed_wall_time_used_sec"]
                ),
                "episodes_used": _stats(
                    budget_rows[tag]["seed_episodes_used"]
                ),
            }
        methods_payload[method] = {
            "label": METHOD_LABELS.get(method, method),
            "step_budgets": budgets_payload,
            "seed_first_success_steps_values": [float(v) for v in first_success_steps_values],
            "seed_first_success_wall_time_sec_values": [
                float(v) for v in first_success_wall_time_values
            ],
            "seed_first_success_episodes_used_values": [
                float(v) for v in first_success_episodes_values
            ],
            "first_success_steps_stats": _stats(first_success_steps_values),
            "first_success_wall_time_sec_stats": _stats(first_success_wall_time_values),
            "first_success_episodes_used_stats": _stats(first_success_episodes_values),
        }

    return {
        "method_order": list(SEARCH_METHOD_ORDER),
        "budget_mode": "environment_steps",
        "search_mode": search_mode or "unknown",
        "step_budgets": [int(v) for v in unique_budgets],
        "num_seed_payloads": int(len(seed_payloads)),
        "methods": methods_payload,
    }


def _plot_meta_curves(
    *,
    aggregate: dict[str, Any],
    output_path: Path,
    x_axis_horizon: int | None = None,
) -> None:
    methods = aggregate.get("methods", {})
    aggregate_horizon = int(aggregate.get("horizon") or 0)
    plot_horizon = int(x_axis_horizon) if x_axis_horizon is not None else aggregate_horizon
    if not methods or plot_horizon < 1:
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
        if std_curve.size == mean_curve.size:
            ax.fill_between(
                steps,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.16,
                linewidth=0,
                zorder=2,
            )
        ax.plot(
            steps,
            mean_curve,
            color=color,
            linewidth=2.0,
            label=label,
            zorder=3,
        )
        finite = mean_curve[np.isfinite(mean_curve)]
        if finite.size > 0:
            finite_chunks.append(finite)

    ax.set_title(
        "Meta-optimizer study | mean best-so-far normalized objective vs steps",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Best-so-far E(F(z)), mean across tasks")
    ax.set_xlim(0.0, float(max(1, plot_horizon)))
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


def _plot_meta_curves_by_method_panels(
    *,
    aggregate: dict[str, Any],
    per_method_task_curves: dict[str, np.ndarray],
    output_path: Path,
    x_axis_horizon: int | None = None,
) -> None:
    methods = aggregate.get("methods", {})
    aggregate_horizon = int(aggregate.get("horizon") or 0)
    plot_horizon = int(x_axis_horizon) if x_axis_horizon is not None else aggregate_horizon
    if not methods or plot_horizon < 1:
        return
    colors = _method_colors_by_method_panel()
    series: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for method in META_METHOD_ORDER_BY_METHOD_PANEL:
        data = methods.get(method)
        if data is None:
            continue
        mean_curve = np.asarray(data.get("mean_curve", []), dtype=np.float64)
        std_curve = np.asarray(data.get("std_curve", []), dtype=np.float64)
        if mean_curve.size < 2:
            continue
        label = METHOD_LABELS.get(method, method)
        series.append((method, mean_curve, std_curve, label))
    if not series:
        return

    n = len(series)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig_w = 9.2
    fig_h = 3.15 * float(nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    mean_line_color = "#24292f"

    for idx, (method, mean_curve, std_curve, label) in enumerate(series):
        ax = axes[idx // ncols][idx % ncols]
        _style_meta_axis(ax)
        steps = np.arange(mean_curve.size, dtype=np.int64)
        color = colors[method]
        task_block = per_method_task_curves.get(method)
        finite_here: list[np.ndarray] = []
        if task_block is not None and task_block.ndim == 2 and task_block.shape[1] == mean_curve.size:
            ax.plot(
                steps,
                task_block.T,
                color=color,
                alpha=0.07,
                linewidth=0.55,
                solid_capstyle="round",
                zorder=1,
            )
            finite_here.append(task_block[np.isfinite(task_block)])
        if std_curve.size == mean_curve.size:
            ax.fill_between(
                steps,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=mean_line_color,
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )
        ax.plot(
            steps,
            mean_curve,
            color=mean_line_color,
            linewidth=2.0,
            zorder=3,
        )
        finite_here.append(mean_curve[np.isfinite(mean_curve)])
        vals = np.concatenate(finite_here) if finite_here else np.asarray([], dtype=np.float64)
        if vals.size > 0:
            y_min = float(np.min(vals))
            y_max = float(np.max(vals))
            margin = 0.08 * max(1e-6, y_max - y_min)
            ax.set_ylim(max(0.0, y_min - margin), min(1.02, y_max + margin))
        else:
            ax.set_ylim(0.0, 1.02)
        ax.set_xlim(0.0, float(max(1, plot_horizon)))
        ax.set_title(label, loc="left", fontsize=10, pad=6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        row = idx // ncols
        col = idx % ncols
        if row == nrows - 1:
            ax.set_xlabel("Optimization step")
        if col == 0:
            ax.set_ylabel("Best-so-far E(F(z))")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        "Meta-optimizer study | per-method mean ± std (faint = individual tasks)",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_meta_final_summary(*, aggregate: dict[str, Any], output_path: Path) -> None:
    methods = aggregate.get("methods", {})
    labels: list[str] = []
    means: list[float] = []
    ci95: list[float] = []
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
        ci95.append(0.0 if not np.isfinite(ci_val) else ci_val)
    if not labels:
        return

    fig, ax = _make_axes(figsize=(9.0, 5.0))
    x = np.arange(len(labels), dtype=np.float64)
    colors = plt.cm.plasma(np.linspace(0.16, 0.84, max(1, len(labels))))
    ax.scatter(x, means, s=46, color=colors, zorder=4)
    ax.errorbar(
        x,
        means,
        yerr=ci95,
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
        "Meta-optimizer study | final objective (mean across seeds with 95% CI)",
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


def _plot_search_budget_curve(*, aggregate: dict[str, Any], output_path: Path) -> None:
    methods = aggregate.get("methods", {})
    budgets = [int(v) for v in aggregate.get("step_budgets", [])]
    if not methods or not budgets:
        return
    colors = _method_colors(SEARCH_METHOD_ORDER)
    fig, ax = _make_axes(figsize=(9.0, 5.2))

    finite_chunks: list[np.ndarray] = []
    for method in SEARCH_METHOD_ORDER:
        method_data = methods.get(method)
        if method_data is None:
            continue
        y_vals: list[float] = []
        ci_vals: list[float] = []
        for budget in budgets:
            tag = _value_tag(budget)
            budget_row = method_data.get("step_budgets", {}).get(tag)
            if budget_row is None:
                y_vals.append(float("nan"))
                ci_vals.append(float("nan"))
                continue
            best_stats = budget_row.get("best_objective", {})
            y_vals.append(float(best_stats.get("mean", float("nan"))))
            ci_vals.append(float(best_stats.get("ci95", float("nan"))))
        y_arr = np.asarray(y_vals, dtype=np.float64)
        ci_arr = np.asarray(ci_vals, dtype=np.float64)
        finite_mask = np.isfinite(y_arr)
        if not np.any(finite_mask):
            continue
        x_arr = np.asarray(budgets, dtype=np.float64)
        color = colors[method]
        label = METHOD_LABELS.get(method, method)
        ax.plot(
            x_arr[finite_mask],
            y_arr[finite_mask],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=label,
        )
        ci_clean = np.where(np.isfinite(ci_arr), ci_arr, 0.0)
        lower = y_arr - ci_clean
        upper = y_arr + ci_clean
        ax.fill_between(
            x_arr[finite_mask],
            lower[finite_mask],
            upper[finite_mask],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        finite_chunks.append(y_arr[finite_mask])

    if not finite_chunks:
        plt.close(fig)
        return

    all_finite = np.concatenate(finite_chunks)
    y_min = float(np.min(all_finite))
    y_max = float(np.max(all_finite))
    margin = 0.08 * max(1e-6, y_max - y_min)
    ax.set_ylim(max(0.0, y_min - margin), min(1.02, y_max + margin))
    ax.set_xscale("log")
    ax.set_xlim(float(min(budgets)), float(max(budgets)))
    ax.set_xlabel("Environment-step budget")
    ax.set_ylabel("Best objective found")
    ax.set_title(
        "Search-algorithm study | best objective during search training vs environment-step budget",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_search_wall_clock_curve(*, aggregate: dict[str, Any], output_path: Path) -> None:
    methods = aggregate.get("methods", {})
    budgets = [int(v) for v in aggregate.get("step_budgets", [])]
    if not methods or not budgets:
        return
    colors = _method_colors(SEARCH_METHOD_ORDER)
    fig, ax = _make_axes(figsize=(9.0, 5.2))

    finite_chunks: list[np.ndarray] = []
    x_chunks: list[np.ndarray] = []
    for method in SEARCH_METHOD_ORDER:
        method_data = methods.get(method)
        if method_data is None:
            continue
        x_ms: list[float] = []
        y_vals: list[float] = []
        ci_vals: list[float] = []
        for budget in budgets:
            tag = _value_tag(budget)
            budget_row = method_data.get("step_budgets", {}).get(tag)
            if budget_row is None:
                continue
            best_stats = budget_row.get("best_objective", {})
            wall_stats = budget_row.get("wall_time_used_sec", {})
            best_mean = float(best_stats.get("mean", float("nan")))
            wall_mean_sec = float(wall_stats.get("mean", float("nan")))
            if not np.isfinite(best_mean) or not np.isfinite(wall_mean_sec) or wall_mean_sec <= 0.0:
                continue
            x_ms.append(1000.0 * wall_mean_sec)
            y_vals.append(best_mean)
            ci_vals.append(float(best_stats.get("ci95", float("nan"))))
        if not x_ms or not y_vals:
            continue
        x_arr = np.asarray(x_ms, dtype=np.float64)
        y_arr = np.asarray(y_vals, dtype=np.float64)
        ci_arr = np.asarray(ci_vals, dtype=np.float64)
        color = colors[method]
        label = METHOD_LABELS.get(method, method)
        ax.plot(x_arr, y_arr, color=color, linewidth=2.0, marker="o", markersize=3, label=label)
        ci_clean = np.where(np.isfinite(ci_arr), ci_arr, 0.0)
        ax.fill_between(
            x_arr,
            y_arr - ci_clean,
            y_arr + ci_clean,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        finite_chunks.append(y_arr)
        x_chunks.append(x_arr)

    if not finite_chunks or not x_chunks:
        plt.close(fig)
        return

    all_finite = np.concatenate(finite_chunks)
    all_x = np.concatenate(x_chunks)
    y_min = float(np.min(all_finite))
    y_max = float(np.max(all_finite))
    margin = 0.08 * max(1e-6, y_max - y_min)
    ax.set_ylim(max(0.0, y_min - margin), min(1.02, y_max + margin))
    ax.set_xscale("log")
    ax.set_xlim(float(np.min(all_x)), float(np.max(all_x)))
    ax.set_xlabel("Wall-clock elapsed (ms)")
    ax.set_ylabel("Best objective found")
    ax.set_title(
        "Search-algorithm study | training-time speed-quality curve",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_search_max_budget_summary(*, aggregate: dict[str, Any], output_path: Path) -> None:
    methods = aggregate.get("methods", {})
    budgets = [int(v) for v in aggregate.get("step_budgets", [])]
    if not methods or not budgets:
        return
    max_budget = int(max(budgets))
    max_tag = _value_tag(max_budget)

    labels: list[str] = []
    means: list[float] = []
    ci95: list[float] = []
    for method in SEARCH_METHOD_ORDER:
        method_data = methods.get(method)
        if method_data is None:
            continue
        budget_row = method_data.get("step_budgets", {}).get(max_tag)
        if budget_row is None:
            continue
        stats = budget_row.get("best_objective", {})
        mean_val = float(stats.get("mean", float("nan")))
        ci_val = float(stats.get("ci95", float("nan")))
        if not np.isfinite(mean_val):
            continue
        labels.append(METHOD_LABELS.get(method, method))
        means.append(mean_val)
        ci95.append(0.0 if not np.isfinite(ci_val) else ci_val)
    if not labels:
        return

    fig, ax = _make_axes(figsize=(9.0, 5.0))
    x = np.arange(len(labels), dtype=np.float64)
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, max(1, len(labels))))
    ax.scatter(x, means, s=46, color=colors, zorder=4)
    ax.errorbar(
        x,
        means,
        yerr=ci95,
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
    ax.set_ylabel("Best objective found")
    ax.set_xlabel("Method")
    ax.set_title(
        f"Search-algorithm study | best objective during search training by {max_budget} steps (95% CI)",
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


def _build_train_config(
    *,
    args: argparse.Namespace,
    seed: int,
    run_name: str,
    fixed_start_target: bool,
    enable_success_curriculum: bool,
) -> TrainConfig:
    return TrainConfig(
        task="spatial",
        seed=int(seed),
        run_name=run_name,
        logdir=str(args.logdir),
        device=str(args.device),
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
        spatial_hidden_dim=int(args.spatial_hidden_dim),
        spatial_visible_dim=int(args.spatial_visible_dim),
        spatial_coord_limit=int(args.spatial_coord_limit),
        spatial_step_size=float(args.spatial_step_size),
        ppo_step_scale=float(args.ppo_step_scale),
        spatial_success_threshold=float(args.spatial_success_threshold),
        spatial_enable_success_curriculum=bool(enable_success_curriculum),
        spatial_basis_complexity=int(args.spatial_basis_complexity),
        spatial_freq_sparsity=int(args.spatial_freq_sparsity),
        spatial_f_type=str(args.spatial_f_type),
        spatial_policy_arch=str(args.spatial_policy_arch),
        spatial_refresh_map_each_episode=False,
        spatial_fixed_start_target=bool(fixed_start_target),
        spatial_plot_interval_episodes=0,
        spatial_enable_baselines=True,
        spatial_tune_baseline_lrs=not bool(args.disable_spatial_baseline_lr_tuning),
        spatial_early_stop_on_all_methods_success=bool(fixed_start_target),
        spatial_baseline_lr_candidates=str(args.spatial_baseline_lr_candidates),
        spatial_baseline_lr_tune_tasks=int(args.spatial_baseline_lr_tune_tasks),
        spatial_enable_optimization_curve_eval=False,
        enable_training_plots=False,
        lattice_rl=bool(args.lattice_RL),
        lattice_granularity=int(args.lattice_granularity),
    )


def _maybe_save_seed_spatial_plots(
    *,
    study_root: Path,
    study_name: str,
    seed: int,
    config: TrainConfig,
    device: torch.device,
    hidden_model: torch.nn.Module,
    hidden_env,
    no_oracle_model: torch.nn.Module | None,
    visible_gradient_model: torch.nn.Module | None,
    visible_gradient_env,
    skip_plotting: bool,
    basin_hop_local_steps: int,
    basin_hop_jump_scale: float,
) -> dict[str, str]:
    if bool(skip_plotting) or int(config.spatial_visible_dim) != 2:
        return {}

    seed_plots_root = study_root / "plots" / f"seed{int(seed)}"
    seed_plots_root.mkdir(parents=True, exist_ok=True)
    original_fixed_start_target = bool(hidden_env.fixed_start_target)
    hidden_env.fixed_start_target = False
    if visible_gradient_env is not None:
        visible_gradient_env.fixed_start_target = False

    trajectory_plots: list[str] = []
    comparison_plots: list[str] = []
    num_plot_tasks = 10
    try:
        for task_index in range(num_plot_tasks):
            output_path = seed_plots_root / (
                f"spatial_trajectory_with_gradients_task{int(task_index + 1):02d}.png"
            )
            maybe_save_spatial_trajectory_plot(
                model=hidden_model,
                env=hidden_env,
                device=device,
                no_oracle_model=no_oracle_model,
                visible_gradient_model=visible_gradient_model,
                visible_gradient_env=visible_gradient_env,
                basin_hop_local_steps=basin_hop_local_steps,
                basin_hop_jump_scale=basin_hop_jump_scale,
                output_path=output_path,
                title=(
                    f"2D trajectory on energy landscape | study={study_name}, "
                    f"seed={int(seed)}, task={int(task_index + 1)}, "
                    f"D={int(config.spatial_hidden_dim)}, mode={config.oracle_mode}"
                ),
            )
            if output_path.exists():
                trajectory_plots.append(str(output_path.resolve()))

            comparison_path = output_path.with_name(
                f"{output_path.stem}_ppo_comparison{output_path.suffix}"
            )
            if comparison_path.exists():
                comparison_plots.append(str(comparison_path.resolve()))
    finally:
        hidden_env.fixed_start_target = original_fixed_start_target
        if visible_gradient_env is not None:
            visible_gradient_env.fixed_start_target = original_fixed_start_target

    return {
        "plots_dir": str(seed_plots_root.resolve()),
        "num_plot_tasks_requested": int(num_plot_tasks),
        "trajectory_plots": trajectory_plots,
        "trajectory_comparison_plots": comparison_plots,
    }


def _run_meta_optimizer_study(
    *,
    args: argparse.Namespace,
    seeds: list[int],
    suite_root: Path,
) -> dict[str, Any]:
    study_root = suite_root / "meta_optimizer"
    plots_root = study_root / "plots"
    data_root = study_root / "plot_data"
    plots_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(args.device))
    basin_hop_local_steps_candidates = _parse_int_list(
        args.meta_basin_hop_local_steps_candidates,
        "meta_basin_hop_local_steps_candidates",
        min_value=1,
    )
    basin_hop_jump_scale_candidates = _parse_float_list(
        args.meta_basin_hop_jump_scale_candidates,
        "meta_basin_hop_jump_scale_candidates",
        min_value=0.0,
    )
    seed_payloads: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []

    lattice_suffix = "_lattice" if bool(args.lattice_RL) else ""
    for seed in seeds:
        run_name = (
            f"{args.run_name_prefix}_meta_P{int(args.oracle_proj_dim)}_visible{int(args.spatial_visible_dim)}{lattice_suffix}_seed{int(seed)}"
        )
        config = _build_train_config(
            args=args,
            seed=int(seed),
            run_name=run_name,
            fixed_start_target=False,
            enable_success_curriculum=not bool(args.meta_disable_success_curriculum),
        )
        print(
            f"[meta_optimizer] seed={int(seed)} training run={run_name} "
            f"(train_steps={int(config.train_steps)})"
        )
        train_t0 = time.perf_counter()
        output = run_training(config, return_artifacts=True)
        train_elapsed_sec = float(time.perf_counter() - train_t0)

        hidden_model = output["model"]
        hidden_env = output.get("env")
        if hidden_env is None:
            raise RuntimeError("Expected spatial env artifact in training output")
        no_oracle_model = output.get("no_oracle_model")
        no_oracle_env = output.get("no_oracle_env")
        visible_gradient_model = output.get("visible_gradient_model")
        visible_gradient_env = output.get("visible_gradient_env")
        if bool(args.disable_meta_basin_hop_tuning):
            basin_hop_tuning = {
                "tuning_enabled": False,
                "num_tasks": 0,
                "base_lr": float(hidden_env.baseline_lr_gd),
                "local_steps_candidates": [int(args.meta_basin_hop_local_steps)],
                "jump_scale_candidates": [float(args.meta_basin_hop_jump_scale)],
                "best_local_steps": int(args.meta_basin_hop_local_steps),
                "best_jump_scale": float(args.meta_basin_hop_jump_scale),
                "scores": [],
            }
        else:
            basin_hop_tuning = tune_meta_basin_hopping(
                env=hidden_env,
                seed=int(seed),
                num_tasks=int(args.meta_basin_hop_tune_tasks),
                local_steps_candidates=basin_hop_local_steps_candidates,
                jump_scale_candidates=basin_hop_jump_scale_candidates,
            )
            basin_hop_tuning["tuning_enabled"] = True
        print(
            "Meta basin hopping selected: "
            f"local_steps={int(basin_hop_tuning['best_local_steps'])}, "
            f"jump_scale={float(basin_hop_tuning['best_jump_scale']):.6g}"
        )
        seed_payload = _evaluate_meta_optimizer_seed(
            seed=int(seed),
            device=device,
            hidden_model=hidden_model,
            hidden_env=hidden_env,
            no_oracle_model=no_oracle_model,
            no_oracle_env=no_oracle_env,
            visible_gradient_model=visible_gradient_model,
            visible_gradient_env=visible_gradient_env,
            num_tasks=int(args.meta_num_tasks),
            horizon=int(args.meta_eval_horizon),
            basin_hop_local_steps=int(basin_hop_tuning["best_local_steps"]),
            basin_hop_jump_scale=float(basin_hop_tuning["best_jump_scale"]),
        )
        seed_payload_path = data_root / f"meta_seed{int(seed)}_evaluation.json"
        with seed_payload_path.open("w", encoding="utf-8") as handle:
            json.dump(seed_payload, handle, indent=2)
        seed_payloads.append(seed_payload)
        seed_plot_paths = _maybe_save_seed_spatial_plots(
            study_root=study_root,
            study_name="meta_optimizer",
            seed=int(seed),
            config=config,
            device=device,
            hidden_model=hidden_model,
            hidden_env=hidden_env,
            no_oracle_model=no_oracle_model,
            visible_gradient_model=visible_gradient_model,
            visible_gradient_env=visible_gradient_env,
            skip_plotting=bool(args.skip_plotting),
            basin_hop_local_steps=int(basin_hop_tuning["best_local_steps"]),
            basin_hop_jump_scale=float(basin_hop_tuning["best_jump_scale"]),
        )

        run_dir = Path(str(output["summary"]["run_dir"])).expanduser().resolve()
        run_entries.append(
            {
                "seed": int(seed),
                "run_name": run_name,
                "run_dir": str(run_dir),
                "summary_json": str((run_dir / "summary.json").resolve()),
                "metrics_jsonl": str((run_dir / "metrics.jsonl").resolve()),
                "config_json": str((run_dir / "config.json").resolve()),
                "training_wall_time_sec": train_elapsed_sec,
                "meta_evaluation_json": str(seed_payload_path.resolve()),
                "basin_hop_tuning": basin_hop_tuning,
                "seed_plot_paths": seed_plot_paths,
            }
        )

        del output

    aggregate = _aggregate_meta_results(seed_payloads)
    aggregate_path = data_root / "meta_optimizer_aggregate.json"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    plot_paths: dict[str, str] = {}
    if not bool(args.skip_plotting):
        curves_plot = plots_root / "meta_optimizer_objective_vs_step.png"
        by_method_plot = plots_root / "meta_optimizer_objective_vs_step_by_method.png"
        final_plot = plots_root / "meta_optimizer_final_objective_summary.png"
        task_best = _stack_meta_per_method_best_so_far_curves(seed_payloads)
        _plot_meta_curves(
            aggregate=aggregate,
            output_path=curves_plot,
            x_axis_horizon=int(args.max_horizon),
        )
        _plot_meta_curves_by_method_panels(
            aggregate=aggregate,
            per_method_task_curves=task_best,
            output_path=by_method_plot,
            x_axis_horizon=int(args.max_horizon),
        )
        _plot_meta_final_summary(aggregate=aggregate, output_path=final_plot)
        if curves_plot.exists():
            plot_paths["objective_vs_step_plot"] = str(curves_plot.resolve())
        if by_method_plot.exists():
            plot_paths["objective_vs_step_by_method_plot"] = str(by_method_plot.resolve())
        if final_plot.exists():
            plot_paths["final_objective_summary_plot"] = str(final_plot.resolve())

    return {
        "study": "meta_optimizer",
        "study_root": str(study_root.resolve()),
        "plots_root": str(plots_root.resolve()),
        "plot_data_root": str(data_root.resolve()),
        "method_order": list(META_METHOD_ORDER),
        "method_labels": dict(METHOD_LABELS),
        "seeds": [int(seed) for seed in seeds],
        "max_horizon": int(args.max_horizon),
        "meta_num_tasks": int(args.meta_num_tasks),
        "meta_eval_horizon": int(args.meta_eval_horizon),
        "meta_basin_hop_local_steps": int(args.meta_basin_hop_local_steps),
        "meta_basin_hop_jump_scale": float(args.meta_basin_hop_jump_scale),
        "meta_basin_hop_tune_tasks": int(args.meta_basin_hop_tune_tasks),
        "meta_basin_hop_tuning_disabled": bool(args.disable_meta_basin_hop_tuning),
        "meta_basin_hop_local_steps_candidates": [
            int(v) for v in basin_hop_local_steps_candidates
        ],
        "meta_basin_hop_jump_scale_candidates": [
            float(v) for v in basin_hop_jump_scale_candidates
        ],
        "runs": run_entries,
        "aggregate_json": str(aggregate_path.resolve()),
        "plots": plot_paths,
    }


def _run_search_algorithm_study(
    *,
    args: argparse.Namespace,
    seeds: list[int],
    suite_root: Path,
    step_budgets: list[int],
) -> dict[str, Any]:
    study_root = suite_root / "search_algorithm"
    plots_root = study_root / "plots"
    data_root = study_root / "plot_data"
    plots_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(args.device))
    seed_payloads: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []

    lattice_suffix = "_lattice" if bool(args.lattice_RL) else ""
    for seed in seeds:
        run_name = (
            f"{args.run_name_prefix}_search_P{int(args.oracle_proj_dim)}_visible{int(args.spatial_visible_dim)}{lattice_suffix}_seed{int(seed)}"
        )
        config = _build_train_config(
            args=args,
            seed=int(seed),
            run_name=run_name,
            fixed_start_target=True,
            enable_success_curriculum=not bool(args.search_disable_success_curriculum),
        )
        print(
            f"[search_algorithm] seed={int(seed)} training run={run_name} "
            f"(train_steps={int(config.train_steps)})"
        )
        train_t0 = time.perf_counter()
        output = run_training(config, return_artifacts=True)
        train_elapsed_sec = float(time.perf_counter() - train_t0)

        hidden_env = output.get("env")
        if hidden_env is None:
            raise RuntimeError("Expected spatial env artifact in training output")
        hidden_model = output["model"]
        no_oracle_model = output.get("no_oracle_model")
        visible_gradient_model = output.get("visible_gradient_model")
        visible_gradient_env = output.get("visible_gradient_env")
        training_trace = output.get("spatial_search_training_trace")
        if not isinstance(training_trace, dict):
            raise RuntimeError(
                "Expected spatial_search_training_trace in search study output; "
                "training artifacts are missing the fixed-task search trace."
            )

        seed_payload = _build_training_time_search_seed_payload(
            seed=int(seed),
            hidden_env=hidden_env,
            training_trace=training_trace,
            step_budgets=step_budgets,
            horizon=int(args.search_eval_horizon),
        )
        seed_payload_path = data_root / f"search_seed{int(seed)}_evaluation.json"
        with seed_payload_path.open("w", encoding="utf-8") as handle:
            json.dump(seed_payload, handle, indent=2)
        seed_payloads.append(seed_payload)
        seed_plot_paths = _maybe_save_seed_spatial_plots(
            study_root=study_root,
            study_name="search_algorithm",
            seed=int(seed),
            config=config,
            device=device,
            hidden_model=hidden_model,
            hidden_env=hidden_env,
            no_oracle_model=no_oracle_model,
            visible_gradient_model=visible_gradient_model,
            visible_gradient_env=visible_gradient_env,
            skip_plotting=bool(args.skip_plotting),
            basin_hop_local_steps=int(args.meta_basin_hop_local_steps),
            basin_hop_jump_scale=float(args.meta_basin_hop_jump_scale),
        )

        run_dir = Path(str(output["summary"]["run_dir"])).expanduser().resolve()
        run_entries.append(
            {
                "seed": int(seed),
                "run_name": run_name,
                "run_dir": str(run_dir),
                "summary_json": str((run_dir / "summary.json").resolve()),
                "metrics_jsonl": str((run_dir / "metrics.jsonl").resolve()),
                "config_json": str((run_dir / "config.json").resolve()),
                "training_wall_time_sec": train_elapsed_sec,
                "training_early_stop_rule": "all_active_rl_methods_first_success",
                "training_search_trace_json": str(
                    (run_dir / "spatial_search_training_trace.json").resolve()
                ),
                "search_evaluation_json": str(seed_payload_path.resolve()),
                "seed_plot_paths": seed_plot_paths,
            }
        )

        del output

    aggregate = _aggregate_search_results(seed_payloads)
    aggregate_path = data_root / "search_algorithm_aggregate.json"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    plot_paths: dict[str, str] = {}
    if not bool(args.skip_plotting):
        budget_curve_plot = plots_root / "search_algorithm_budget_curve.png"
        wall_clock_curve_plot = plots_root / "search_algorithm_wall_clock_curve.png"
        max_budget_plot = plots_root / "search_algorithm_max_budget_summary.png"
        _plot_search_budget_curve(aggregate=aggregate, output_path=budget_curve_plot)
        _plot_search_wall_clock_curve(aggregate=aggregate, output_path=wall_clock_curve_plot)
        _plot_search_max_budget_summary(aggregate=aggregate, output_path=max_budget_plot)
        if budget_curve_plot.exists():
            plot_paths["budget_curve_plot"] = str(budget_curve_plot.resolve())
        if wall_clock_curve_plot.exists():
            plot_paths["wall_clock_curve_plot"] = str(wall_clock_curve_plot.resolve())
        if max_budget_plot.exists():
            plot_paths["max_budget_summary_plot"] = str(max_budget_plot.resolve())

    return {
        "study": "search_algorithm",
        "study_root": str(study_root.resolve()),
        "plots_root": str(plots_root.resolve()),
        "plot_data_root": str(data_root.resolve()),
        "method_order": list(SEARCH_METHOD_ORDER),
        "method_labels": dict(METHOD_LABELS),
        "seeds": [int(seed) for seed in seeds],
        "budget_mode": "environment_steps",
        "step_budgets": [int(v) for v in step_budgets],
        "search_eval_horizon": int(args.search_eval_horizon),
        "search_max_episodes_per_method": int(args.search_max_episodes_per_method),
        "search_curve_points": int(args.search_curve_points),
        "search_rl_deterministic": bool(args.search_rl_deterministic),
        "search_training_early_stop_rule": "all_active_rl_methods_first_success",
        "runs": run_entries,
        "aggregate_json": str(aggregate_path.resolve()),
        "plots": plot_paths,
    }


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run spatial optimizer studies with separate experiment tracks:\n"
            "1) meta_optimizer (equal horizon, random starts/targets)\n"
            "2) search_algorithm (fixed task, environment-step-budgeted search with random-search baseline)"
        )
    )
    parser.add_argument(
        "--study",
        type=str,
        choices=["meta_optimizer", "search_algorithm", "all"],
        default="meta_optimizer",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds. Each seed defines one Fourier map/task draw.",
    )
    parser.add_argument(
        "--suite_output_dir",
        type=str,
        default="plots/spatial_optimizer_studies",
    )
    parser.add_argument("--suite_name", type=str, default="spatial_optimizer_studies")
    parser.add_argument("--run_name_prefix", type=str, default="spatial_optimizer_study")
    parser.add_argument("--skip_plotting", action="store_true")

    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument("--device", type=str, default=defaults.device)
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
    parser.add_argument("--sensing", type=str, choices=["S0", "S1"], default=defaults.sensing)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--ppo_epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)
    parser.add_argument("--policy_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--oracle_proj_dim",
        type=int,
        default=defaults.oracle_proj_dim,
        help="Oracle-token projection width before the policy trunk (0 disables projection).",
    )

    parser.add_argument("--spatial_hidden_dim", type=int, default=150)
    parser.add_argument("--spatial_visible_dim", type=int, default=defaults.spatial_visible_dim)
    parser.add_argument("--spatial_coord_limit", type=int, default=3)
    parser.add_argument(
        "--base_lr",
        dest="spatial_step_size",
        type=float,
        default=defaults.spatial_step_size,
        help="Base learning rate for visible-space GD/Adam baselines.",
    )
    parser.add_argument(
        "--spatial_step_size",
        dest="spatial_step_size",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--ppo_step_scale", type=float, default=defaults.ppo_step_scale)
    parser.add_argument(
        "--spatial_success_threshold",
        type=float,
        default=defaults.spatial_success_threshold,
    )
    parser.add_argument(
        "--spatial_basis_complexity",
        type=int,
        default=defaults.spatial_basis_complexity,
    )
    parser.add_argument(
        "--spatial_freq_sparsity",
        type=int,
        default=defaults.spatial_freq_sparsity,
        help=(
            "Max nonzero components per Fourier frequency vector (interaction order r). "
            "0 = dense (all d components, original behavior). "
            "1 = axis-aligned only. 2 = pairwise interactions. etc."
        ),
    )
    parser.add_argument(
        "--spatial_f_type",
        type=str,
        choices=["FOURIER", "MLP"],
        default=defaults.spatial_f_type,
    )
    parser.add_argument(
        "--spatial_policy_arch",
        type=str,
        choices=["mlp", "gru"],
        default=defaults.spatial_policy_arch,
    )
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

    parser.add_argument("--meta_num_tasks", type=int, default=500)
    parser.add_argument(
        "--meta_eval_horizon",
        type=int,
        default=None,
        help=(
            "Evaluation horizon for meta-optimizer curves. "
            "If omitted, defaults to --max_horizon."
        ),
    )
    parser.add_argument(
        "--meta_basin_hop_tune_tasks",
        type=int,
        default=64,
        help="Number of sampled tasks used to tune basin-hopping local_steps and jump_scale.",
    )
    parser.add_argument(
        "--meta_basin_hop_local_steps",
        type=int,
        default=8,
        help="Fallback local-step count when basin-hop tuning is disabled.",
    )
    parser.add_argument(
        "--meta_basin_hop_jump_scale",
        type=float,
        default=1.0,
        help="Fallback jump radius scale when basin-hop tuning is disabled.",
    )
    parser.add_argument(
        "--meta_basin_hop_local_steps_candidates",
        type=str,
        default="4,8,12,20",
        help="Comma-separated local-step candidates for basin-hop tuning.",
    )
    parser.add_argument(
        "--meta_basin_hop_jump_scale_candidates",
        type=str,
        default="0.25,0.5,1.0",
        help="Comma-separated jump-scale candidates for basin-hop tuning.",
    )
    parser.add_argument(
        "--disable_meta_basin_hop_tuning",
        action="store_true",
        help="Disable tuning and use --meta_basin_hop_local_steps/--meta_basin_hop_jump_scale directly.",
    )
    parser.add_argument(
        "--meta_disable_success_curriculum",
        dest="meta_disable_success_curriculum",
        action="store_true",
        default=True,
        help="Disable success-threshold curriculum for meta-optimizer training runs. This is now the default.",
    )
    parser.add_argument(
        "--meta_disable_success",
        dest="meta_disable_success_curriculum",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--meta_enable_success_curriculum",
        dest="meta_disable_success_curriculum",
        action="store_false",
        help="Enable success-threshold curriculum for meta-optimizer training runs.",
    )

    parser.add_argument(
        "--search_step_budgets",
        type=str,
        default="",
        help=(
            "Optional comma-separated environment-step budgets for search evaluation. "
            "If omitted, a dense log-spaced curve is generated automatically."
        ),
    )
    parser.add_argument(
        "--search_eval_horizon",
        type=int,
        default=None,
        help=(
            "Per-episode horizon for search-algorithm evaluation. "
            "If omitted, defaults to --max_horizon."
        ),
    )
    parser.add_argument("--search_max_episodes_per_method", type=int, default=5000)
    parser.add_argument(
        "--search_curve_points",
        type=int,
        default=160,
        help="Number of automatically generated step-budget points when --search_step_budgets is omitted.",
    )
    parser.add_argument(
        "--search_rl_deterministic",
        action="store_true",
        help="Use deterministic policy actions for search evaluation (default: stochastic).",
    )
    parser.add_argument(
        "--search_disable_success_curriculum",
        action="store_true",
        help="Disable success-threshold curriculum for fixed-task search training runs.",
    )
    parser.add_argument(
        "--lattice_RL",
        action="store_true",
        help="Enable lattice-grid discrete PPO: the RL agent selects among moves to adjacent lattice nodes.",
    )
    parser.add_argument(
        "--lattice_granularity",
        type=int,
        default=defaults.lattice_granularity,
        help="Number of lattice nodes per dimension of feasible space (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.meta_eval_horizon is None:
        args.meta_eval_horizon = int(args.max_horizon)
    if args.search_eval_horizon is None:
        args.search_eval_horizon = int(args.max_horizon)
    if int(args.meta_basin_hop_tune_tasks) < 1:
        raise ValueError("meta_basin_hop_tune_tasks must be >= 1")
    if int(args.oracle_proj_dim) < 0:
        raise ValueError("oracle_proj_dim must be >= 0")
    if int(args.meta_basin_hop_local_steps) < 1:
        raise ValueError("meta_basin_hop_local_steps must be >= 1")
    if float(args.meta_basin_hop_jump_scale) < 0.0:
        raise ValueError("meta_basin_hop_jump_scale must be >= 0")
    if str(args.meta_basin_hop_local_steps_candidates).strip():
        _parse_int_list(
            args.meta_basin_hop_local_steps_candidates,
            "meta_basin_hop_local_steps_candidates",
            min_value=1,
        )
    if str(args.meta_basin_hop_jump_scale_candidates).strip():
        _parse_float_list(
            args.meta_basin_hop_jump_scale_candidates,
            "meta_basin_hop_jump_scale_candidates",
            min_value=0.0,
        )
    seeds = _parse_int_list(args.seeds, "seeds", min_value=0)
    max_search_steps = int(args.search_eval_horizon) * int(args.search_max_episodes_per_method)
    if str(args.search_step_budgets).strip():
        step_budgets = _parse_int_list(
            args.search_step_budgets,
            "search_step_budgets",
            min_value=1,
        )
    else:
        step_budgets = _build_search_step_budgets(
            max_steps=max_search_steps,
            num_points=int(args.search_curve_points),
        )

    base_suite = Path(args.suite_output_dir).expanduser().resolve()
    suite_suffix = f"_vis{int(args.spatial_visible_dim)}"
    if bool(args.lattice_RL):
        suite_suffix += "_lattice"
    suite_root = base_suite.parent / f"{base_suite.name}{suite_suffix}"
    suite_root.mkdir(parents=True, exist_ok=True)
    control_budget_scale = float(np.sqrt(float(max(2, int(args.spatial_visible_dim))) / 2.0))

    manifest: dict[str, Any] = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_name": str(args.suite_name),
        "suite_root": str(suite_root),
        "study_mode": str(args.study),
        "seeds": [int(seed) for seed in seeds],
        "train_config": {
            "train_steps": int(args.train_steps),
            "n_env": int(args.n_env),
            "rollout_len": int(args.rollout_len),
            "running_avg_window": int(args.running_avg_window),
            "max_horizon": int(args.max_horizon),
            "sensing": str(args.sensing),
            "lr": float(args.lr),
            "ppo_epochs": int(args.ppo_epochs),
            "minibatches": int(args.minibatches),
            "policy_hidden_dim": int(args.policy_hidden_dim),
            "oracle_proj_dim": int(args.oracle_proj_dim),
            "spatial_hidden_dim": int(args.spatial_hidden_dim),
            "spatial_visible_dim": int(args.spatial_visible_dim),
            "spatial_coord_limit": int(args.spatial_coord_limit),
            "spatial_step_size": float(args.spatial_step_size),
            "spatial_control_budget_scale": control_budget_scale,
            "spatial_effective_step_size": float(args.spatial_step_size) * control_budget_scale,
            "ppo_step_scale": float(args.ppo_step_scale),
            "spatial_success_threshold": float(args.spatial_success_threshold),
            "spatial_basis_complexity": int(args.spatial_basis_complexity),
            "spatial_f_type": str(args.spatial_f_type),
            "spatial_policy_arch": str(args.spatial_policy_arch),
            "device": str(args.device),
            "logdir": str(args.logdir),
        },
        "meta_evaluation_config": {
            "meta_num_tasks": int(args.meta_num_tasks),
            "meta_eval_horizon": int(args.meta_eval_horizon),
            "meta_basin_hop_tune_tasks": int(args.meta_basin_hop_tune_tasks),
            "meta_basin_hop_local_steps": int(args.meta_basin_hop_local_steps),
            "meta_basin_hop_jump_scale": float(args.meta_basin_hop_jump_scale),
            "meta_basin_hop_local_steps_candidates": str(args.meta_basin_hop_local_steps_candidates),
            "meta_basin_hop_jump_scale_candidates": str(args.meta_basin_hop_jump_scale_candidates),
            "meta_basin_hop_tuning_disabled": bool(args.disable_meta_basin_hop_tuning),
        },
        "meta_optimizer": None,
        "search_algorithm": None,
    }

    if args.study in {"meta_optimizer", "all"}:
        manifest["meta_optimizer"] = _run_meta_optimizer_study(
            args=args,
            seeds=seeds,
            suite_root=suite_root,
        )

    if args.study in {"search_algorithm", "all"}:
        manifest["search_algorithm"] = _run_search_algorithm_study(
            args=args,
            seeds=seeds,
            suite_root=suite_root,
            step_budgets=step_budgets,
        )

    manifest_path = suite_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "suite_root": str(suite_root.resolve()),
        "study_mode": str(args.study),
        "num_seeds": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "meta_optimizer_enabled": bool(manifest["meta_optimizer"] is not None),
        "search_algorithm_enabled": bool(manifest["search_algorithm"] is not None),
    }
    summary_path = suite_root / "suite_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
