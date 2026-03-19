import argparse
import csv
import json
import time
import warnings
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import TrainConfig
from .model import PolicyValueNet
from .oracle import SPATIAL_ORACLE_MODES, SpatialOracle
from .plotting import (
    plot_path_length_histograms,
    plot_spatial_optimization_curve_summary,
    plot_spatial_optimization_curves_by_method,
    plot_spatial_trajectory_with_gradients,
)
from .ppo import PPOHyperParams, RolloutBuffer, compute_gae, ppo_update
from .spatial_env import VectorizedSpatialEnv


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def obs_to_tensors(
    obs: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_features = torch.as_tensor(obs["token_features"], dtype=torch.float32, device=device)
    dist_feature = torch.as_tensor(obs["dist"], dtype=torch.float32, device=device)
    step_frac = torch.as_tensor(obs["step_frac"], dtype=torch.float32, device=device)
    return token_features, dist_feature, step_frac


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64).reshape(-1)
        if batch.size == 0:
            return
        batch_mean = float(np.mean(batch))
        batch_var = float(np.var(batch))
        batch_count = float(batch.size)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: float) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = float(new_mean)
        self.var = float(max(new_var, 0.0))
        self.count = float(total_count)


def normalize_rewards_with_running_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    returns_tracker: np.ndarray,
    returns_rms: RunningMeanStd,
    gamma: float,
    clip_abs: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_f64 = np.asarray(rewards, dtype=np.float64)
    dones_f64 = np.asarray(dones, dtype=np.float64)

    updated_returns = returns_tracker * float(gamma) + rewards_f64
    returns_rms.update(updated_returns)

    scale = float(np.sqrt(returns_rms.var + 1e-8))
    normalized_rewards = (rewards_f64 / scale).astype(np.float32)
    if clip_abs > 0.0:
        normalized_rewards = np.clip(normalized_rewards, -clip_abs, clip_abs).astype(np.float32)

    updated_returns = updated_returns * (1.0 - dones_f64)
    return normalized_rewards, updated_returns


SPATIAL_OPTIMIZATION_METHOD_ORDER = [
    "gd",
    "adam",
    "rl_no_oracle",
    "rl_visible_oracle",
    "rl_hidden_gradient",
]
SPATIAL_GD_TUNING_LRS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)

SPATIAL_OPTIMIZATION_METHOD_LABELS = {
    "gd": "GD",
    "adam": "Adam",
    "rl_no_oracle": "RL no oracle",
    "rl_visible_oracle": "RL visible oracle",
    "rl_hidden_gradient": "RL hidden gradient",
}


def _next_spatial_success_threshold(
    current: float,
    *,
    decay: float,
    minimum: float,
) -> float:
    keep = 1.0 - float(decay)
    if keep <= 0.0:
        return float(minimum)
    return max(float(minimum), float(current) * keep)


def _should_early_stop_spatial_search_training(
    *,
    config: TrainConfig,
    main_success_seen: bool,
    no_oracle_model: torch.nn.Module | None,
    no_oracle_success_seen: bool,
    visible_gradient_model: torch.nn.Module | None,
    visible_gradient_success_seen: bool,
) -> bool:
    if not bool(config.spatial_early_stop_on_all_methods_success):
        return False
    if not bool(config.spatial_fixed_start_target):
        return False

    if not bool(main_success_seen):
        return False

    if no_oracle_model is not None and not bool(no_oracle_success_seen):
        return False

    if visible_gradient_model is not None and not bool(visible_gradient_success_seen):
        return False

    return True


def _init_objective_plateau_tracker() -> dict[str, float | int | bool | None]:
    return {
        "best_objective": None,
        "best_episode": None,
        "last_objective": None,
        "plateau_trigger_episode": None,
    }


def _update_objective_plateau_tracker(
    *,
    tracker: dict[str, float | int | bool | None],
    avg_objective: float,
    episodes_completed: int,
    warmup_episodes: int,
    patience_episodes: int,
    min_delta: float,
) -> bool:
    if not np.isfinite(float(avg_objective)):
        return False

    tracker["last_objective"] = float(avg_objective)

    best_objective_raw = tracker.get("best_objective")
    best_episode_raw = tracker.get("best_episode")
    if best_objective_raw is None or best_episode_raw is None:
        tracker["best_objective"] = float(avg_objective)
        tracker["best_episode"] = int(episodes_completed)
        tracker["plateau_trigger_episode"] = None
        return False

    best_objective = float(best_objective_raw)
    best_episode = int(best_episode_raw)
    if float(avg_objective) <= best_objective - float(min_delta):
        tracker["best_objective"] = float(avg_objective)
        tracker["best_episode"] = int(episodes_completed)
        tracker["plateau_trigger_episode"] = None
        return False

    if int(episodes_completed) < int(warmup_episodes):
        return False

    if int(episodes_completed) - best_episode < int(patience_episodes):
        return False

    if tracker.get("plateau_trigger_episode") is None:
        tracker["plateau_trigger_episode"] = int(episodes_completed)
    return True


def _init_spatial_search_training_trace(
    *,
    config: TrainConfig,
    env: VectorizedSpatialEnv,
    no_oracle_model: torch.nn.Module | None,
    visible_gradient_model: torch.nn.Module | None,
) -> dict[str, object]:
    methods: dict[str, dict[str, object]] = {
        "rl_hidden_gradient": {
            "records": [],
            "best_objective_so_far": None,
            "first_success_steps": None,
            "first_success_wall_time_sec": None,
            "first_success_episodes_used": None,
        }
    }
    if no_oracle_model is not None:
        methods["rl_no_oracle"] = {
            "records": [],
            "best_objective_so_far": None,
            "first_success_steps": None,
            "first_success_wall_time_sec": None,
            "first_success_episodes_used": None,
        }
    if visible_gradient_model is not None:
        methods["rl_visible_oracle"] = {
            "records": [],
            "best_objective_so_far": None,
            "first_success_steps": None,
            "first_success_wall_time_sec": None,
            "first_success_episodes_used": None,
        }
    return {
        "mode": "training_time_search",
        "budget_mode": "environment_steps",
        "training_step_granularity": int(config.n_env),
        "success_threshold": float(env.success_threshold),
        "methods": methods,
    }


def _record_spatial_search_training_step(
    *,
    trace: dict[str, object] | None,
    method: str,
    obs: dict[str, np.ndarray] | None,
    infos: list[dict] | tuple[dict, ...] | None,
    dones: np.ndarray | None,
    total_steps: int,
    episodes_completed: int,
    elapsed_sec: float,
) -> None:
    if trace is None or obs is None or dones is None or infos is None:
        return

    methods = trace.get("methods")
    if not isinstance(methods, dict):
        return
    method_trace = methods.get(method)
    if not isinstance(method_trace, dict):
        return

    current_dist = np.asarray(obs.get("dist", []), dtype=np.float64).reshape(-1).copy()
    done_mask = np.asarray(dones, dtype=bool).reshape(-1)
    if current_dist.size == 0 or done_mask.size == 0:
        return

    success_now = False
    max_index = min(current_dist.size, done_mask.size, len(infos))
    for idx in range(max_index):
        if not done_mask[idx]:
            continue
        info = infos[idx]
        if not isinstance(info, dict) or not info.get("episode_done", False):
            continue
        final_objective = float(info.get("final_objective", float("nan")))
        if np.isfinite(final_objective):
            current_dist[idx] = final_objective
        success_now = success_now or bool(info.get("success", False))

    finite = current_dist[np.isfinite(current_dist)]
    if finite.size == 0:
        return

    candidate_best = float(np.min(finite))
    running_best_raw = method_trace.get("best_objective_so_far")
    running_best = float(running_best_raw) if running_best_raw is not None else candidate_best
    running_best = min(running_best, candidate_best)
    method_trace["best_objective_so_far"] = float(running_best)

    if success_now and method_trace.get("first_success_steps") is None:
        method_trace["first_success_steps"] = int(total_steps)
        method_trace["first_success_wall_time_sec"] = float(elapsed_sec)
        method_trace["first_success_episodes_used"] = int(episodes_completed)

    records = method_trace.get("records")
    if not isinstance(records, list):
        return
    records.append(
        {
            "steps_used": int(total_steps),
            "wall_time_used_sec": float(elapsed_sec),
            "episodes_used": int(episodes_completed),
            "best_objective": float(running_best),
            "success_found": bool(method_trace.get("first_success_steps") is not None),
        }
    )


def _capture_spatial_task_snapshot(
    env: VectorizedSpatialEnv,
    source_xy: np.ndarray,
    target_min_xy: np.ndarray,
    env_index: int = 0,
) -> dict[str, np.ndarray]:
    idx = int(env_index)
    snapshot = {
        "source_xy": np.asarray(source_xy, dtype=np.float32).copy(),
        "target_min_xy": np.asarray(target_min_xy, dtype=np.float32).copy(),
        "linear_w": env.linear_w[idx].copy(),
        "s_star": env.s_star[idx].copy(),
        "max_objective": np.asarray(env.max_objective_env[idx], dtype=np.float32),
    }
    if env.f_type == "FOURIER":
        snapshot.update(
            {
                "sin_w": env.sin_w[idx].copy(),
                "cos_w": env.cos_w[idx].copy(),
                "sin_phase": env.sin_phase[idx].copy(),
                "cos_phase": env.cos_phase[idx].copy(),
                "sin_amp": env.sin_amp[idx].copy(),
                "cos_amp": env.cos_amp[idx].copy(),
            }
        )
    else:
        snapshot.update(
            {
                "mlp_w1": env.mlp_w1[idx].copy(),
                "mlp_b1": env.mlp_b1[idx].copy(),
                "mlp_w2": env.mlp_w2[idx].copy(),
                "mlp_b2": env.mlp_b2[idx].copy(),
            }
        )
    return snapshot


def _apply_spatial_task_snapshot(
    env: VectorizedSpatialEnv,
    snapshot: dict[str, np.ndarray],
    env_index: int = 0,
) -> None:
    idx = int(env_index)
    env.linear_w[idx] = snapshot["linear_w"].astype(np.float32)
    if env.f_type == "FOURIER":
        env.sin_w[idx] = snapshot["sin_w"].astype(np.float32)
        env.cos_w[idx] = snapshot["cos_w"].astype(np.float32)
        env.sin_phase[idx] = snapshot["sin_phase"].astype(np.float32)
        env.cos_phase[idx] = snapshot["cos_phase"].astype(np.float32)
        env.sin_amp[idx] = snapshot["sin_amp"].astype(np.float32)
        env.cos_amp[idx] = snapshot["cos_amp"].astype(np.float32)
        env.mlp_w1[idx].fill(0.0)
        env.mlp_b1[idx].fill(0.0)
        env.mlp_w2[idx].fill(0.0)
        env.mlp_b2[idx].fill(0.0)
    else:
        env.mlp_w1[idx] = snapshot["mlp_w1"].astype(np.float32)
        env.mlp_b1[idx] = snapshot["mlp_b1"].astype(np.float32)
        env.mlp_w2[idx] = snapshot["mlp_w2"].astype(np.float32)
        env.mlp_b2[idx] = snapshot["mlp_b2"].astype(np.float32)
        env.sin_w[idx].fill(0.0)
        env.cos_w[idx].fill(0.0)
        env.sin_phase[idx].fill(0.0)
        env.cos_phase[idx].fill(0.0)
        env.sin_amp[idx].fill(0.0)
        env.cos_amp[idx].fill(0.0)

    target_min_xy = snapshot["target_min_xy"].astype(np.float32)
    env.reference_min_xy_env[idx] = target_min_xy
    if idx == 0:
        env.reference_min_xy = target_min_xy.copy()
    env.s_star[idx] = snapshot["s_star"].astype(np.float32)
    if "max_objective" in snapshot:
        env._set_max_objective(
            float(np.asarray(snapshot["max_objective"]).reshape(())),
            env_index=idx,
        )
    else:
        env._refresh_objective_scale(env_index=idx)


def _synchronize_spatial_envs(
    reference_env: VectorizedSpatialEnv,
    target_env: VectorizedSpatialEnv,
) -> None:
    if reference_env.n_env != target_env.n_env:
        raise ValueError("Cannot synchronize spatial envs with different n_env")
    if reference_env.visible_dim != target_env.visible_dim:
        raise ValueError("Cannot synchronize spatial envs with different visible_dim")
    if reference_env.hidden_dim != target_env.hidden_dim:
        raise ValueError("Cannot synchronize spatial envs with different hidden_dim")
    if reference_env.f_type != target_env.f_type:
        raise ValueError("Cannot synchronize spatial envs with different map families")

    target_env.max_objective = float(reference_env.max_objective)
    target_env.max_objective_env[:] = reference_env.max_objective_env.astype(np.float32)
    target_env.reference_min_xy = reference_env.reference_min_xy.astype(np.float32).copy()
    target_env.reference_min_xy_env[:] = reference_env.reference_min_xy_env.astype(np.float32)
    target_env.s_star[:] = reference_env.s_star.astype(np.float32)
    target_env.current_xy[:] = reference_env.current_xy.astype(np.float32)
    target_env.initial_xy[:] = reference_env.initial_xy.astype(np.float32)
    target_env.steps[:] = reference_env.steps.astype(np.int32)
    target_env.horizons[:] = reference_env.horizons.astype(np.int32)
    target_env.initial_dist[:] = reference_env.initial_dist.astype(np.int32)
    target_env.initial_objective[:] = reference_env.initial_objective.astype(np.float32)
    target_env.completed_episodes = int(reference_env.completed_episodes)
    target_env.success_threshold = float(reference_env.success_threshold)

    if reference_env._fixed_target_min_xy is None:
        target_env._fixed_target_min_xy = None
    else:
        target_env._fixed_target_min_xy = reference_env._fixed_target_min_xy.astype(np.float32).copy()
    if reference_env._fixed_source_xy is None:
        target_env._fixed_source_xy = None
    else:
        target_env._fixed_source_xy = reference_env._fixed_source_xy.astype(np.float32).copy()

    for env_index in range(reference_env.n_env):
        snapshot = _capture_spatial_task_snapshot(
            env=reference_env,
            source_xy=reference_env.initial_xy[env_index],
            target_min_xy=reference_env.reference_min_xy_env[env_index],
            env_index=env_index,
        )
        _apply_spatial_task_snapshot(target_env, snapshot, env_index=env_index)


def _rollout_spatial_descent_curve(
    env: VectorizedSpatialEnv,
    start_xy: np.ndarray,
    horizon: int,
    env_index: int = 0,
    base_lr: float | None = None,
) -> np.ndarray:
    state = start_xy.astype(np.float32).copy()
    h = max(1, int(horizon))
    curve = np.empty(h + 1, dtype=np.float32)
    curve[0] = float(env._normalized_objective_value(state, env_index=env_index))

    for step in range(h):
        grad_xy = env._gradient_xy(state, env_index=env_index)
        lr_t = env._cosine_annealed_baseline_lr(step, h, base_lr=base_lr)
        update = (-lr_t * grad_xy).astype(np.float32)
        state = env._apply_baseline_optimizer_step(state, update)
        curve[step + 1] = float(env._normalized_objective_value(state, env_index=env_index))

    return curve


def _rollout_spatial_adam_curve(
    env: VectorizedSpatialEnv,
    start_xy: np.ndarray,
    horizon: int,
    env_index: int = 0,
    base_lr: float | None = None,
) -> np.ndarray:
    state = start_xy.astype(np.float32).copy()
    h = max(1, int(horizon))
    curve = np.empty(h + 1, dtype=np.float32)
    curve[0] = float(env._normalized_objective_value(state, env_index=env_index))
    m = np.zeros(env.visible_dim, dtype=np.float64)
    v = np.zeros(env.visible_dim, dtype=np.float64)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for step in range(h):
        grad_xy = env._gradient_xy(state, env_index=env_index).astype(np.float64)
        m = beta1 * m + (1.0 - beta1) * grad_xy
        v = beta2 * v + (1.0 - beta2) * (grad_xy * grad_xy)
        t = step + 1
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        lr_t = env._cosine_annealed_baseline_lr(step, h, base_lr=base_lr)
        update = (-lr_t * (m_hat / (np.sqrt(v_hat) + eps))).astype(np.float32)
        state = env._apply_baseline_optimizer_step(state, update)
        curve[step + 1] = float(env._normalized_objective_value(state, env_index=env_index))

    return curve


@torch.no_grad()
def _rollout_spatial_policy_curve(
    model: PolicyValueNet,
    env: VectorizedSpatialEnv,
    device: torch.device,
    start_xy: np.ndarray,
    horizon: int,
    env_index: int = 0,
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
            deterministic=True,
        )
        action = action_t.squeeze(0).cpu().numpy()
        state = env._apply_action(state, action)
        curve[step + 1] = float(env._normalized_objective_value(state, env_index=env_index))

    return curve

def _compute_curve_stats(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if curves.ndim != 2 or curves.shape[0] == 0:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )
    mean_curve = np.nanmean(curves, axis=0).astype(np.float32)
    std_curve = np.nanstd(curves, axis=0).astype(np.float32)
    return mean_curve, std_curve


def _parse_baseline_lr_candidates(candidate_str: str) -> list[float]:
    pieces = [piece.strip() for piece in str(candidate_str).split(",")]
    values: list[float] = []
    for piece in pieces:
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError as exc:
            raise ValueError(f"Invalid baseline lr candidate: {piece!r}") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Baseline lr candidate must be finite and > 0, got {piece!r}")
        values.append(value)
    if not values:
        raise ValueError("spatial_baseline_lr_candidates must include at least one positive value")
    ordered_unique = sorted(set(float(v) for v in values))
    return ordered_unique


def _sample_spatial_tuning_task_snapshots(
    env: VectorizedSpatialEnv,
    num_tasks: int,
) -> list[dict]:
    tasks: list[dict] = []
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


def _rollout_baseline_with_projection_stats(
    *,
    env: VectorizedSpatialEnv,
    method: str,
    start_xy: np.ndarray,
    horizon: int,
    env_index: int = 0,
    base_lr: float,
) -> tuple[float, float, float]:
    state = start_xy.astype(np.float32).copy()
    h = max(1, int(horizon))
    projected_steps = 0
    executed_steps = 0

    m = np.zeros(env.visible_dim, dtype=np.float64)
    v = np.zeros(env.visible_dim, dtype=np.float64)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for step in range(h):
        if method == "gd":
            grad_xy = env._gradient_xy(state, env_index=env_index)
            lr_t = env._cosine_annealed_baseline_lr(step, h, base_lr=base_lr)
            update = (-lr_t * grad_xy).astype(np.float32)
        elif method == "adam":
            grad_xy = env._gradient_xy(state, env_index=env_index).astype(np.float64)
            m = beta1 * m + (1.0 - beta1) * grad_xy
            v = beta2 * v + (1.0 - beta2) * (grad_xy * grad_xy)
            t = step + 1
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            lr_t = env._cosine_annealed_baseline_lr(step, h, base_lr=base_lr)
            update = (-lr_t * (m_hat / (np.sqrt(v_hat) + eps))).astype(np.float32)
        else:
            raise ValueError(f"Unsupported baseline method for lr tuning: {method!r}")

        proposed_state = state.astype(np.float32) + update.astype(np.float32)
        clipped_state = np.clip(proposed_state, -env.coord_limit, env.coord_limit).astype(np.float32)
        if not np.allclose(proposed_state, clipped_state, rtol=0.0, atol=1e-7):
            projected_steps += 1
        state = clipped_state
        executed_steps += 1

        if env._is_success(state, env_index=env_index):
            break

    final_objective = float(env._objective_value(state, env_index=env_index))
    final_ref_distance = float(env._reference_distance(state, env_index=env_index))
    projection_rate = float(projected_steps / max(1, executed_steps))
    return final_objective, final_ref_distance, projection_rate


def _score_baseline_lr_candidate(
    env: VectorizedSpatialEnv,
    method: str,
    candidate_lr: float,
    tasks: list[dict],
) -> dict[str, float]:
    final_objectives: list[float] = []
    final_ref_distances: list[float] = []
    projection_rates: list[float] = []
    successes: list[float] = []
    for task in tasks:
        snapshot = task["snapshot"]
        start_xy = np.asarray(task["start_xy"], dtype=np.float32)
        horizon = int(task["horizon"])
        _apply_spatial_task_snapshot(env, snapshot, env_index=0)
        final_value_raw, final_ref_distance, projection_rate = _rollout_baseline_with_projection_stats(
            env=env,
            method=method,
            start_xy=start_xy,
            horizon=horizon,
            env_index=0,
            base_lr=candidate_lr,
        )
        final_value = float(env._normalized_objective_from_raw(float(final_value_raw), env_index=0))
        final_objectives.append(final_value)
        final_ref_distances.append(float(final_ref_distance))
        projection_rates.append(float(projection_rate))
        successes.append(1.0 if final_value <= float(env.success_threshold) else 0.0)

    avg_final_objective = float(np.mean(final_objectives)) if final_objectives else float("inf")
    avg_final_ref_distance = (
        float(np.mean(final_ref_distances)) if final_ref_distances else float("inf")
    )
    avg_projection_rate = float(np.mean(projection_rates)) if projection_rates else float("inf")
    success_rate = float(np.mean(successes)) if successes else 0.0
    return {
        "candidate_lr": float(candidate_lr),
        "avg_final_ref_distance": avg_final_ref_distance,
        "avg_final_objective": avg_final_objective,
        "avg_projection_rate": avg_projection_rate,
        "success_rate": success_rate,
    }


def tune_spatial_baseline_learning_rates(
    *,
    env: VectorizedSpatialEnv,
    candidate_lrs: list[float],
    num_tasks: int,
    seed: int,
) -> dict:
    tasks = _sample_spatial_tuning_task_snapshots(env=env, num_tasks=max(1, int(num_tasks)))
    methods = ("gd", "adam")
    per_method_scores: dict[str, list[dict[str, float]]] = {method: [] for method in methods}
    candidate_lrs_by_method: dict[str, list[float]] = {
        "gd": [float(v) for v in SPATIAL_GD_TUNING_LRS],
        "adam": [float(v) for v in candidate_lrs],
    }
    best_lrs: dict[str, float] = {}

    for method_idx, method in enumerate(methods):
        method_candidate_lrs = candidate_lrs_by_method[method]
        for candidate_idx, lr in enumerate(method_candidate_lrs):
            score = _score_baseline_lr_candidate(
                env=env,
                method=method,
                candidate_lr=float(lr),
                tasks=tasks,
            )
            per_method_scores[method].append(score)

        best = min(
            per_method_scores[method],
            key=lambda row: (
                float(row["avg_final_objective"]),
                float(row["avg_projection_rate"]),
                -float(row["success_rate"]),
                float(row["avg_final_ref_distance"]),
                float(row["candidate_lr"]),
            ),
        )
        best_lrs[method] = float(best["candidate_lr"])

    return {
        "num_tasks": int(len(tasks)),
        "candidate_lrs": [float(v) for v in candidate_lrs],
        "candidate_lrs_by_method": candidate_lrs_by_method,
        "best_lrs": best_lrs,
        "scores": per_method_scores,
    }


def evaluate_spatial_optimization_curves(
    *,
    config: TrainConfig,
    run_dir: Path,
    device: torch.device,
    hidden_gradient_model: PolicyValueNet,
    hidden_gradient_env: VectorizedSpatialEnv,
    no_oracle_model: PolicyValueNet | None,
    no_oracle_env: VectorizedSpatialEnv | None,
    visible_gradient_model: PolicyValueNet | None,
    visible_gradient_env: VectorizedSpatialEnv | None,
) -> dict | None:
    if not config.spatial_enable_optimization_curve_eval:
        return None

    num_tasks = max(1, int(config.spatial_optimization_curve_tasks))
    horizon = max(1, int(config.max_horizon))
    method_curves_lists: dict[str, list[np.ndarray]] = {
        method_key: [] for method_key in SPATIAL_OPTIMIZATION_METHOD_ORDER
    }

    hidden_was_training = hidden_gradient_model.training
    no_oracle_was_training = no_oracle_model.training if no_oracle_model is not None else None
    visible_was_training = visible_gradient_model.training if visible_gradient_model is not None else None

    hidden_gradient_model.eval()
    if no_oracle_model is not None:
        no_oracle_model.eval()
    if visible_gradient_model is not None:
        visible_gradient_model.eval()

    for _ in range(num_tasks):
        spec = hidden_gradient_env.sample_episode_spec(
            env_index=0,
            refresh_map=hidden_gradient_env.refresh_map_each_episode,
        )
        snapshot = _capture_spatial_task_snapshot(
            env=hidden_gradient_env,
            source_xy=spec.source,
            target_min_xy=spec.target_min_xy,
            env_index=0,
        )
        start_xy = snapshot["source_xy"]

        _apply_spatial_task_snapshot(hidden_gradient_env, snapshot, env_index=0)
        method_curves_lists["gd"].append(
            _rollout_spatial_descent_curve(
                env=hidden_gradient_env,
                start_xy=start_xy,
                horizon=horizon,
                env_index=0,
                base_lr=hidden_gradient_env.baseline_lr_gd,
            )
        )
        method_curves_lists["adam"].append(
            _rollout_spatial_adam_curve(
                env=hidden_gradient_env,
                start_xy=start_xy,
                horizon=horizon,
                env_index=0,
                base_lr=hidden_gradient_env.baseline_lr_adam,
            )
        )

        if config.oracle_mode == "convex_gradient":
            _apply_spatial_task_snapshot(hidden_gradient_env, snapshot, env_index=0)
            method_curves_lists["rl_hidden_gradient"].append(
                _rollout_spatial_policy_curve(
                    model=hidden_gradient_model,
                    env=hidden_gradient_env,
                    device=device,
                    start_xy=start_xy,
                    horizon=horizon,
                    env_index=0,
                )
            )

        if no_oracle_model is not None and no_oracle_env is not None:
            _apply_spatial_task_snapshot(no_oracle_env, snapshot, env_index=0)
            method_curves_lists["rl_no_oracle"].append(
                _rollout_spatial_policy_curve(
                    model=no_oracle_model,
                    env=no_oracle_env,
                    device=device,
                    start_xy=start_xy,
                    horizon=horizon,
                    env_index=0,
                )
            )

        if visible_gradient_model is not None and visible_gradient_env is not None:
            _apply_spatial_task_snapshot(visible_gradient_env, snapshot, env_index=0)
            method_curves_lists["rl_visible_oracle"].append(
                _rollout_spatial_policy_curve(
                    model=visible_gradient_model,
                    env=visible_gradient_env,
                    device=device,
                    start_xy=start_xy,
                    horizon=horizon,
                    env_index=0,
                )
            )

    if hidden_was_training:
        hidden_gradient_model.train()
    if no_oracle_model is not None and bool(no_oracle_was_training):
        no_oracle_model.train()
    if visible_gradient_model is not None and bool(visible_was_training):
        visible_gradient_model.train()

    method_curves: dict[str, np.ndarray] = {}
    method_means: dict[str, np.ndarray] = {}
    method_stds: dict[str, np.ndarray] = {}
    missing_methods: list[str] = []
    for method_key in SPATIAL_OPTIMIZATION_METHOD_ORDER:
        curves_list = method_curves_lists.get(method_key, [])
        if not curves_list:
            missing_methods.append(method_key)
            continue
        curves = np.stack(curves_list, axis=0).astype(np.float32)
        mean_curve, std_curve = _compute_curve_stats(curves)
        method_curves[method_key] = curves
        method_means[method_key] = mean_curve
        method_stds[method_key] = std_curve

    if not method_curves:
        return None
    if missing_methods:
        print(
            "Spatial optimization-curve evaluation skipped unavailable methods: "
            + ", ".join(missing_methods)
        )

    by_method_plot_path = run_dir / "spatial_optimization_curves_by_method.png"
    summary_plot_path = run_dir / "spatial_optimization_curves_mean_std.png"
    plot_spatial_optimization_curves_by_method(
        method_curves=method_curves,
        output_path=by_method_plot_path,
        title=(
            f"Spatial optimization curves per method (seed={config.seed}, "
            f"tasks={num_tasks}, horizon={horizon})"
        ),
        method_labels=SPATIAL_OPTIMIZATION_METHOD_LABELS,
    )
    plot_spatial_optimization_curve_summary(
        method_mean_curves=method_means,
        method_std_curves=method_stds,
        output_path=summary_plot_path,
        title=(
            f"Spatial optimization mean ± std over sampled tasks (seed={config.seed}, "
            f"tasks={num_tasks}, horizon={horizon})"
        ),
        method_labels=SPATIAL_OPTIMIZATION_METHOD_LABELS,
    )

    methods_payload: dict[str, dict[str, list[float]]] = {}
    for method_key in SPATIAL_OPTIMIZATION_METHOD_ORDER:
        if method_key not in method_curves:
            continue
        methods_payload[method_key] = {
            "label": SPATIAL_OPTIMIZATION_METHOD_LABELS.get(method_key, method_key),
            "mean_curve": [float(v) for v in method_means[method_key]],
            "std_curve": [float(v) for v in method_stds[method_key]],
            "task_curves": method_curves[method_key].astype(np.float64).tolist(),
        }

    curve_data_path = run_dir / "spatial_optimization_curves.json"
    curve_data = {
        "task": "spatial",
        "seed": int(config.seed),
        "num_tasks": int(num_tasks),
        "horizon": int(horizon),
        "baseline_lrs": {
            "gd": float(hidden_gradient_env.baseline_lr_gd),
            "adam": float(hidden_gradient_env.baseline_lr_adam),
        },
        "steps": list(range(horizon + 1)),
        "method_order": list(SPATIAL_OPTIMIZATION_METHOD_ORDER),
        "available_methods": list(method_curves.keys()),
        "missing_methods": missing_methods,
        "methods": methods_payload,
        "by_method_plot_path": str(by_method_plot_path),
        "summary_plot_path": str(summary_plot_path),
    }
    with curve_data_path.open("w", encoding="utf-8") as handle:
        json.dump(curve_data, handle, indent=2)

    return {
        "num_tasks": int(num_tasks),
        "horizon": int(horizon),
        "available_methods": list(method_curves.keys()),
        "missing_methods": missing_methods,
        "method_order": list(SPATIAL_OPTIMIZATION_METHOD_ORDER),
        "method_labels": dict(SPATIAL_OPTIMIZATION_METHOD_LABELS),
        "baseline_lrs": {
            "gd": float(hidden_gradient_env.baseline_lr_gd),
            "adam": float(hidden_gradient_env.baseline_lr_adam),
        },
        "methods": {
            key: {
                "mean_curve": [float(v) for v in method_means[key]],
                "std_curve": [float(v) for v in method_stds[key]],
            }
            for key in method_curves
        },
        "curve_data_path": str(curve_data_path),
        "by_method_plot_path": str(by_method_plot_path),
        "summary_plot_path": str(summary_plot_path),
    }


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    total_updates: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    mode = str(config.lr_scheduler).lower()
    if mode == "none":
        return None

    min_factor = float(config.lr_min_factor)
    warmup_updates = max(0, int(config.lr_warmup_updates))
    updates_total = max(1, int(total_updates))
    decay_updates = max(1, updates_total - warmup_updates)

    def _lambda(update_index: int) -> float:
        u = int(update_index)
        if warmup_updates > 0 and u < warmup_updates:
            return max(1e-8, float(u + 1) / float(warmup_updates))
        if mode == "constant":
            return 1.0
        t = float(np.clip((u - warmup_updates) / float(decay_updates), 0.0, 1.0))
        if mode == "linear":
            return float(min_factor + (1.0 - min_factor) * (1.0 - t))
        if mode == "cosine":
            return float(min_factor + 0.5 * (1.0 - min_factor) * (1.0 + np.cos(np.pi * t)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=_lambda)


@torch.no_grad()
def evaluate_policy(
    model: PolicyValueNet,
    env: VectorizedSpatialEnv,
    eval_episodes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    success_count = 0
    path_lengths: list[int] = []
    stretches: list[float] = []

    for _ in range(eval_episodes):
        spec = env.sample_episode_spec(
            env_index=0,
            refresh_map=env.refresh_map_each_episode,
        )
        state = spec.source.copy()
        steps_taken = 0
        success = False
        hidden_state = model.initial_state(batch_size=1, device=device)

        for step in range(spec.horizon):
            token_features = env._obs_token_features(state, env_index=0)[None, :]
            dist_feature = env._normalized_objective_value(state, env_index=0)
            step_fraction = float(step / max(1, spec.horizon))

            token_t = torch.tensor(token_features, dtype=torch.float32, device=device)
            dist_t = torch.tensor([dist_feature], dtype=torch.float32, device=device)
            step_t = torch.tensor([step_fraction], dtype=torch.float32, device=device)

            action_t, _, _, hidden_state = model.act(
                token_t,
                dist_t,
                step_t,
                hidden_state=hidden_state,
                deterministic=True,
            )
            action = action_t.squeeze(0).cpu().numpy()

            state = env._apply_action(state, action)
            steps_taken += 1
            if env._is_success(state, env_index=0):
                success = True
                break

        path_lengths.append(steps_taken)
        if success:
            success_count += 1
            if spec.shortest_dist > 0:
                stretches.append(steps_taken / spec.shortest_dist)

    model.train()
    return {
        "success_rate": float(success_count / max(1, eval_episodes)),
        "avg_path_len": float(np.mean(path_lengths)) if path_lengths else 0.0,
        "avg_stretch": float(np.mean(stretches)) if stretches else float("nan"),
    }


@torch.no_grad()
def collect_spatial_trajectory(
    model: PolicyValueNet,
    env: VectorizedSpatialEnv,
    device: torch.device,
    no_oracle_model: PolicyValueNet | None = None,
    visible_gradient_model: PolicyValueNet | None = None,
    visible_gradient_env: VectorizedSpatialEnv | None = None,
    basin_hop_local_steps: int = 8,
    basin_hop_jump_scale: float = 1.0,
) -> dict[str, np.ndarray] | None:
    if env.visible_dim != 2:
        return None
    model.eval()
    if no_oracle_model is not None:
        no_oracle_model.eval()
    if visible_gradient_model is not None:
        visible_gradient_model.eval()

    spec = env.sample_episode_spec(
        env_index=0,
        refresh_map=env.refresh_map_each_episode,
    )
    task_snapshot = _capture_spatial_task_snapshot(
        env=env,
        source_xy=spec.source,
        target_min_xy=spec.target_min_xy,
        env_index=0,
    )
    state = spec.source.copy()
    hidden_state = model.initial_state(batch_size=1, device=device)

    trajectory = [state.astype(np.float32)]
    gradient_xy: list[np.ndarray] = []
    move_vectors: list[np.ndarray] = []

    for step in range(spec.horizon):
        token_features = env._obs_token_features(state, env_index=0)[None, :]
        z_feature = env._normalized_objective_value(state, env_index=0)
        step_fraction = float(step / max(1, spec.horizon))

        token_t = torch.tensor(token_features, dtype=torch.float32, device=device)
        dist_t = torch.tensor([z_feature], dtype=torch.float32, device=device)
        step_t = torch.tensor([step_fraction], dtype=torch.float32, device=device)
        action_t, _, _, hidden_state = model.act(
            token_t,
            dist_t,
            step_t,
            hidden_state=hidden_state,
            deterministic=True,
        )
        action = action_t.squeeze(0).cpu().numpy()

        grad_xy = env._gradient_xy(state, env_index=0)
        next_state = env._apply_action(state, action)

        gradient_xy.append(grad_xy.astype(np.float32))
        move_vectors.append((next_state - state).astype(np.float32))
        trajectory.append(next_state.astype(np.float32))
        state = next_state

        if env._is_success(state, env_index=0):
            break

    no_oracle_trajectory_xy: np.ndarray | None = None
    if no_oracle_model is not None:
        no_state = spec.source.copy()
        no_hidden_state = no_oracle_model.initial_state(batch_size=1, device=device)
        no_trajectory = [no_state.astype(np.float32)]
        for step in range(spec.horizon):
            token_features = env._obs_token_features(no_state, env_index=0)[None, :]
            token_features[:, : env.oracle_token_dim] = 0.0
            z_feature = env._normalized_objective_value(no_state, env_index=0)
            step_fraction = float(step / max(1, spec.horizon))

            token_t = torch.tensor(token_features, dtype=torch.float32, device=device)
            dist_t = torch.tensor([z_feature], dtype=torch.float32, device=device)
            step_t = torch.tensor([step_fraction], dtype=torch.float32, device=device)
            action_t, _, _, no_hidden_state = no_oracle_model.act(
                token_t,
                dist_t,
                step_t,
                hidden_state=no_hidden_state,
                deterministic=True,
            )
            action = action_t.squeeze(0).cpu().numpy()

            no_next_state = env._apply_action(no_state, action)
            no_trajectory.append(no_next_state.astype(np.float32))
            no_state = no_next_state

            if env._is_success(no_state, env_index=0):
                break
        no_oracle_trajectory_xy = np.asarray(no_trajectory, dtype=np.float32)

    visible_gradient_trajectory_xy: np.ndarray | None = None
    if visible_gradient_model is not None and visible_gradient_env is not None:
        _apply_spatial_task_snapshot(visible_gradient_env, task_snapshot, env_index=0)
        visible_state = spec.source.copy()
        visible_hidden_state = visible_gradient_model.initial_state(batch_size=1, device=device)
        visible_trajectory = [visible_state.astype(np.float32)]
        for step in range(spec.horizon):
            token_features = visible_gradient_env._obs_token_features(visible_state, env_index=0)[None, :]
            z_feature = visible_gradient_env._normalized_objective_value(visible_state, env_index=0)
            step_fraction = float(step / max(1, spec.horizon))

            token_t = torch.tensor(token_features, dtype=torch.float32, device=device)
            dist_t = torch.tensor([z_feature], dtype=torch.float32, device=device)
            step_t = torch.tensor([step_fraction], dtype=torch.float32, device=device)
            action_t, _, _, visible_hidden_state = visible_gradient_model.act(
                token_t,
                dist_t,
                step_t,
                hidden_state=visible_hidden_state,
                deterministic=True,
            )
            action = action_t.squeeze(0).cpu().numpy()
            visible_next_state = visible_gradient_env._apply_action(visible_state, action)
            visible_trajectory.append(visible_next_state.astype(np.float32))
            visible_state = visible_next_state

            if visible_gradient_env._is_success(visible_state, env_index=0):
                break
        visible_gradient_trajectory_xy = np.asarray(visible_trajectory, dtype=np.float32)

    model.train()
    if no_oracle_model is not None:
        no_oracle_model.train()
    if visible_gradient_model is not None:
        visible_gradient_model.train()
    if len(trajectory) < 2:
        return None

    result = {
        "trajectory_xy": np.asarray(trajectory, dtype=np.float32),
        "gradient_xy": np.asarray(gradient_xy, dtype=np.float32),
        "move_vectors_xy": np.asarray(move_vectors, dtype=np.float32),
        "target_xy": spec.target_min_xy.astype(np.float32),
        "baseline_trajectory_xy": rollout_spatial_gradient_descent_baseline(
            env=env,
            start_xy=spec.source,
            horizon=spec.horizon,
            base_lr=env.baseline_lr_gd,
        ),
        "adam_baseline_trajectory_xy": rollout_spatial_adam_baseline(
            env=env,
            start_xy=spec.source,
            horizon=spec.horizon,
            base_lr=env.baseline_lr_adam,
        ),
        "basin_hopping_trajectory_xy": rollout_spatial_basin_hopping_baseline(
            env=env,
            start_xy=spec.source,
            horizon=spec.horizon,
            base_lr=env.baseline_lr_gd,
            local_steps=basin_hop_local_steps,
            jump_scale=basin_hop_jump_scale,
        ),
    }
    if no_oracle_trajectory_xy is not None and no_oracle_trajectory_xy.shape[0] >= 2:
        result["no_oracle_trajectory_xy"] = no_oracle_trajectory_xy
    if visible_gradient_trajectory_xy is not None and visible_gradient_trajectory_xy.shape[0] >= 2:
        result["visible_gradient_trajectory_xy"] = visible_gradient_trajectory_xy
    return result


def rollout_spatial_gradient_descent_baseline(
    env: VectorizedSpatialEnv,
    start_xy: np.ndarray,
    horizon: int,
    base_lr: float | None = None,
) -> np.ndarray:
    # Baseline: local descent in z-space using the true 2D gradient of E(F(z)),
    # with cosine-annealed learning rate eta_t = eta_0 * schedule_t.
    state = start_xy.astype(np.float32).copy()
    trajectory = [state.astype(np.float32)]
    if base_lr is None:
        base_lr = env.baseline_lr_gd

    for step in range(int(horizon)):
        grad_xy = env._gradient_xy(state, env_index=0)
        lr_t = env._cosine_annealed_baseline_lr(step, int(horizon), base_lr=base_lr)
        update = (-lr_t * grad_xy).astype(np.float32)
        next_state = env._apply_baseline_optimizer_step(state, update)
        trajectory.append(next_state.astype(np.float32))
        state = next_state

        if env._is_success(state, env_index=0):
            break

    return np.asarray(trajectory, dtype=np.float32)


def rollout_spatial_adam_baseline(
    env: VectorizedSpatialEnv,
    start_xy: np.ndarray,
    horizon: int,
    base_lr: float | None = None,
) -> np.ndarray:
    state = start_xy.astype(np.float32).copy()
    trajectory = [state.astype(np.float32)]
    m = np.zeros(env.visible_dim, dtype=np.float64)
    v = np.zeros(env.visible_dim, dtype=np.float64)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr0 = float(env.baseline_lr_adam if base_lr is None else base_lr)

    for step in range(int(horizon)):
        grad_xy = env._gradient_xy(state, env_index=0).astype(np.float64)
        m = beta1 * m + (1.0 - beta1) * grad_xy
        v = beta2 * v + (1.0 - beta2) * (grad_xy * grad_xy)
        t = step + 1
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        lr_t = env._cosine_annealed_baseline_lr(step, int(horizon), base_lr=lr0)
        adam_update = (-lr_t * (m_hat / (np.sqrt(v_hat) + eps))).astype(np.float32)
        next_state = env._apply_baseline_optimizer_step(state, adam_update)
        trajectory.append(next_state.astype(np.float32))
        state = next_state

        if env._is_success(state, env_index=0):
            break

    return np.asarray(trajectory, dtype=np.float32)


def rollout_spatial_basin_hopping_baseline(
    env: VectorizedSpatialEnv,
    start_xy: np.ndarray,
    horizon: int,
    base_lr: float | None = None,
    local_steps: int = 8,
    jump_scale: float = 1.0,
    rng: np.random.Generator | None = None,
    stop_on_success: bool = True,
) -> np.ndarray:
    state = start_xy.astype(np.float32).copy()
    trajectory = [state.astype(np.float32)]
    lr0 = float(env.baseline_lr_gd if base_lr is None else base_lr)
    local_steps = max(1, int(local_steps))
    jump_scale = max(0.0, float(jump_scale))
    local_rng = rng if rng is not None else np.random.default_rng(0)
    cycle_length = local_steps + 1
    jump_radius = float(jump_scale) * float(env.step_size) * float(np.sqrt(float(local_steps)))

    for step in range(int(horizon)):
        cycle_pos = int(step % cycle_length)
        if cycle_pos == local_steps and jump_radius > 0.0:
            jump_dir = local_rng.normal(size=env.visible_dim).astype(np.float32)
            jump_norm = float(np.linalg.norm(jump_dir))
            if jump_norm > 1e-8 and np.isfinite(jump_norm):
                jump = (jump_dir / jump_norm) * jump_radius
                candidate_state = np.clip(
                    state + jump.astype(np.float32),
                    -env.coord_limit,
                    env.coord_limit,
                ).astype(np.float32)
                current_objective = float(env._normalized_objective_value(state, env_index=0))
                candidate_objective = float(
                    env._normalized_objective_value(candidate_state, env_index=0)
                )
                if candidate_objective <= current_objective:
                    next_state = candidate_state
                else:
                    next_state = state.copy()
            else:
                next_state = state.copy()
        else:
            grad_xy = env._gradient_xy(state, env_index=0)
            lr_t = env._cosine_annealed_baseline_lr(step, int(horizon), base_lr=lr0)
            update = (-lr_t * grad_xy).astype(np.float32)
            next_state = env._apply_baseline_optimizer_step(state, update)

        trajectory.append(next_state.astype(np.float32))
        state = next_state

        if bool(stop_on_success) and env._is_success(state, env_index=0):
            break

    return np.asarray(trajectory, dtype=np.float32)


def maybe_save_spatial_trajectory_plot(
    model: PolicyValueNet,
    env: VectorizedSpatialEnv,
    device: torch.device,
    output_path: Path,
    title: str,
    no_oracle_model: PolicyValueNet | None = None,
    visible_gradient_model: PolicyValueNet | None = None,
    visible_gradient_env: VectorizedSpatialEnv | None = None,
    basin_hop_local_steps: int = 8,
    basin_hop_jump_scale: float = 1.0,
) -> None:
    if env.visible_dim != 2:
        return
    spatial_trace = collect_spatial_trajectory(
        model,
        env,
        device,
        no_oracle_model=no_oracle_model,
        visible_gradient_model=visible_gradient_model,
        visible_gradient_env=visible_gradient_env,
        basin_hop_local_steps=basin_hop_local_steps,
        basin_hop_jump_scale=basin_hop_jump_scale,
    )
    if spatial_trace is None:
        return

    grid_x, grid_y, grid_energy = env.energy_landscape_grid(resolution=150, env_index=0)
    grid_energy_normalized = np.clip(
        grid_energy / max(float(env.max_objective_env[0]), 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)

    # Figure 1: PPO hidden-gradient oracle + optimizer baselines (GD/Adam).
    plot_spatial_trajectory_with_gradients(
        trajectory_xy=spatial_trace["trajectory_xy"],
        gradient_xy=spatial_trace["gradient_xy"],
        move_vectors_xy=spatial_trace["move_vectors_xy"],
        target_xy=spatial_trace["target_xy"],
        baseline_trajectory_xy=spatial_trace["baseline_trajectory_xy"],
        adam_baseline_trajectory_xy=spatial_trace["adam_baseline_trajectory_xy"],
        basin_hopping_trajectory_xy=spatial_trace["basin_hopping_trajectory_xy"],
        no_oracle_trajectory_xy=None,
        visible_gradient_trajectory_xy=None,
        output_path=output_path,
        title=f"{title} | PPO hidden + GD/Adam/Basin hopping",
        landscape_x=grid_x,
        landscape_y=grid_y,
        landscape_energy=grid_energy_normalized,
        landscape_label="Normalized E(F(z))",
    )

    # Figure 2: PPO hidden-gradient oracle vs PPO visible-gradient oracle vs PPO no-oracle.
    ppo_comparison_output_path = output_path.with_name(
        f"{output_path.stem}_ppo_comparison{output_path.suffix}"
    )
    plot_spatial_trajectory_with_gradients(
        trajectory_xy=spatial_trace["trajectory_xy"],
        gradient_xy=spatial_trace["gradient_xy"],
        move_vectors_xy=spatial_trace["move_vectors_xy"],
        target_xy=spatial_trace["target_xy"],
        baseline_trajectory_xy=None,
        adam_baseline_trajectory_xy=None,
        basin_hopping_trajectory_xy=None,
        no_oracle_trajectory_xy=spatial_trace.get("no_oracle_trajectory_xy"),
        visible_gradient_trajectory_xy=spatial_trace.get("visible_gradient_trajectory_xy"),
        output_path=ppo_comparison_output_path,
        title=f"{title} | PPO hidden/visible/no-oracle",
        landscape_x=grid_x,
        landscape_y=grid_y,
        landscape_energy=grid_energy_normalized,
        landscape_label="Normalized E(F(z))",
    )


def find_episodes_to_threshold(metrics: list[dict], threshold: float = 0.8) -> int | None:
    for item in metrics:
        if item["success_rate"] >= threshold:
            return int(item["episodes"])
    return None


def _resolve_device(device_str: str) -> torch.device:
    """Resolve device from config; use CUDA if requested and available."""
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        warnings.warn("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def run_training(config: TrainConfig, return_artifacts: bool = False) -> dict:
    if config.algo != "ppo":
        raise ValueError("Only --algo ppo is implemented in this prototype")
    if config.running_avg_window < 1:
        raise ValueError("running_avg_window must be >= 1")
    if config.oracle_mode not in SPATIAL_ORACLE_MODES:
        raise ValueError(
            f"oracle_mode={config.oracle_mode!r} is invalid for task='spatial'; "
            f"choose one of {sorted(SPATIAL_ORACLE_MODES)}"
        )
    if config.oracle_mode == "visible_gradient":
        raise ValueError(
            "oracle_mode='visible_gradient' is reserved for the parallel baseline; "
            "run main PPO with oracle_mode='convex_gradient'."
        )
    if config.spatial_policy_arch not in {"mlp", "gru"}:
        raise ValueError("spatial_policy_arch must be one of {'mlp', 'gru'}")
    if config.lr_scheduler not in {"none", "constant", "linear", "cosine"}:
        raise ValueError(
            "lr_scheduler must be one of {'none', 'constant', 'linear', 'cosine'}"
        )
    if config.lr_min_factor <= 0.0 or config.lr_min_factor > 1.0:
        raise ValueError("lr_min_factor must be in (0, 1]")
    if config.lr_warmup_updates < 0:
        raise ValueError("lr_warmup_updates must be >= 0")
    if config.spatial_optimization_curve_tasks < 1:
        raise ValueError("spatial_optimization_curve_tasks must be >= 1")
    if config.spatial_baseline_lr_tune_tasks < 1:
        raise ValueError("spatial_baseline_lr_tune_tasks must be >= 1")
    if not np.isfinite(float(config.ppo_step_scale)) or float(config.ppo_step_scale) <= 0.0:
        raise ValueError("ppo_step_scale must be finite and > 0")
    if config.spatial_enable_success_curriculum:
        if config.spatial_success_curriculum_start <= 0.0:
            raise ValueError("spatial_success_curriculum_start must be > 0")
        if (
            config.spatial_success_curriculum_trigger_rate <= 0.0
            or config.spatial_success_curriculum_trigger_rate > 1.0
        ):
            raise ValueError("spatial_success_curriculum_trigger_rate must be in (0, 1]")
        if (
            config.spatial_success_curriculum_decay <= 0.0
            or config.spatial_success_curriculum_decay >= 1.0
        ):
            raise ValueError("spatial_success_curriculum_decay must be in (0, 1)")
        if config.spatial_success_curriculum_min <= 0.0:
            raise ValueError("spatial_success_curriculum_min must be > 0")
        if config.spatial_success_curriculum_start < config.spatial_success_curriculum_min:
            raise ValueError(
                "spatial_success_curriculum_start must be >= spatial_success_curriculum_min"
            )

    set_global_seed(config.seed)
    device = _resolve_device(config.device)

    spatial_curriculum_enabled = bool(config.spatial_enable_success_curriculum)
    spatial_success_threshold = float(config.spatial_success_threshold)
    if spatial_curriculum_enabled:
        spatial_success_threshold = float(config.spatial_success_curriculum_start)
        spatial_success_threshold = max(
            float(config.spatial_success_curriculum_min),
            spatial_success_threshold,
        )

    run_dir = config.resolve_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    no_oracle_env: VectorizedSpatialEnv | None = None
    no_oracle_model: PolicyValueNet | None = None
    no_oracle_optimizer: torch.optim.Optimizer | None = None
    no_oracle_buffer: RolloutBuffer | None = None
    no_oracle_obs: dict[str, np.ndarray] | None = None
    no_oracle_recurrent_state: torch.Tensor | None = None
    visible_gradient_env: VectorizedSpatialEnv | None = None
    visible_gradient_model: PolicyValueNet | None = None
    visible_gradient_optimizer: torch.optim.Optimizer | None = None
    visible_gradient_buffer: RolloutBuffer | None = None
    visible_gradient_obs: dict[str, np.ndarray] | None = None
    visible_gradient_recurrent_state: torch.Tensor | None = None
    baseline_lr_tuning_result: dict | None = None
    baseline_lr_tuning_path: Path | None = None

    spatial_token_dim = config.spatial_token_dim
    if config.oracle_mode == "convex_gradient":
        spatial_token_dim = config.spatial_hidden_dim
    oracle = SpatialOracle(
        hidden_dim=config.spatial_hidden_dim,
        token_dim=spatial_token_dim,
        mode=config.oracle_mode,
        seed=config.seed + 29,
        token_noise_std=config.spatial_token_noise_std,
    )
    env = VectorizedSpatialEnv(
        hidden_dim=config.spatial_hidden_dim,
        visible_dim=config.spatial_visible_dim,
        coord_limit=config.spatial_coord_limit,
        oracle=oracle,
        n_env=config.n_env,
        sensing=config.sensing,
        max_horizon=config.max_horizon,
        seed=config.seed + 41,
        s1_step_penalty=config.s1_step_penalty,
        reward_noise_std=config.reward_noise_std,
        step_size=config.spatial_step_size,
        ppo_step_scale=config.ppo_step_scale,
        success_threshold=spatial_success_threshold,
        basis_complexity=config.spatial_basis_complexity,
        f_type=config.spatial_f_type,
        refresh_map_each_episode=config.spatial_refresh_map_each_episode,
        compute_episode_baselines=config.spatial_enable_baselines,
        fixed_start_target=config.spatial_fixed_start_target,
    )
    action_dim = env.action_dim
    action_space_type = "continuous"

    if config.spatial_enable_baselines and config.oracle_mode != "no_oracle":
        no_oracle_oracle = SpatialOracle(
            hidden_dim=config.spatial_hidden_dim,
            token_dim=spatial_token_dim,
            mode="no_oracle",
            seed=config.seed + 131,
            token_noise_std=config.spatial_token_noise_std,
        )
        no_oracle_env = VectorizedSpatialEnv(
            hidden_dim=config.spatial_hidden_dim,
            visible_dim=config.spatial_visible_dim,
            coord_limit=config.spatial_coord_limit,
            oracle=no_oracle_oracle,
            n_env=config.n_env,
            sensing=config.sensing,
            max_horizon=config.max_horizon,
            seed=config.seed + 149,
            s1_step_penalty=config.s1_step_penalty,
            reward_noise_std=config.reward_noise_std,
            step_size=config.spatial_step_size,
            ppo_step_scale=config.ppo_step_scale,
            success_threshold=spatial_success_threshold,
            basis_complexity=config.spatial_basis_complexity,
            f_type=config.spatial_f_type,
            refresh_map_each_episode=config.spatial_refresh_map_each_episode,
            compute_episode_baselines=config.spatial_enable_baselines,
            fixed_start_target=config.spatial_fixed_start_target,
        )
        if config.oracle_mode == "convex_gradient":
            visible_gradient_oracle = SpatialOracle(
                hidden_dim=config.spatial_hidden_dim,
                token_dim=config.spatial_visible_dim,
                mode="visible_gradient",
                seed=config.seed + 167,
                token_noise_std=config.spatial_token_noise_std,
            )
            visible_gradient_env = VectorizedSpatialEnv(
                hidden_dim=config.spatial_hidden_dim,
                visible_dim=config.spatial_visible_dim,
                coord_limit=config.spatial_coord_limit,
                oracle=visible_gradient_oracle,
                n_env=config.n_env,
                sensing=config.sensing,
                max_horizon=config.max_horizon,
                seed=config.seed + 173,
                s1_step_penalty=config.s1_step_penalty,
                reward_noise_std=config.reward_noise_std,
                step_size=config.spatial_step_size,
                ppo_step_scale=config.ppo_step_scale,
                success_threshold=spatial_success_threshold,
                basis_complexity=config.spatial_basis_complexity,
                f_type=config.spatial_f_type,
                refresh_map_each_episode=config.spatial_refresh_map_each_episode,
                compute_episode_baselines=config.spatial_enable_baselines,
                fixed_start_target=config.spatial_fixed_start_target,
            )
        if no_oracle_env is not None:
            _synchronize_spatial_envs(env, no_oracle_env)
        if visible_gradient_env is not None:
            _synchronize_spatial_envs(env, visible_gradient_env)

    if config.spatial_tune_baseline_lrs:
        candidate_lrs = _parse_baseline_lr_candidates(config.spatial_baseline_lr_candidates)
        baseline_lr_tuning_result = tune_spatial_baseline_learning_rates(
            env=env,
            candidate_lrs=candidate_lrs,
            num_tasks=config.spatial_baseline_lr_tune_tasks,
            seed=config.seed + 809,
        )
        best_lrs = baseline_lr_tuning_result["best_lrs"]
        env.set_baseline_learning_rates(
            gd=float(best_lrs["gd"]),
            adam=float(best_lrs["adam"]),
        )
        if no_oracle_env is not None:
            no_oracle_env.set_baseline_learning_rates(
                gd=float(best_lrs["gd"]),
                adam=float(best_lrs["adam"]),
            )
        if visible_gradient_env is not None:
            visible_gradient_env.set_baseline_learning_rates(
                gd=float(best_lrs["gd"]),
                adam=float(best_lrs["adam"]),
            )
        for env_index in range(env.n_env):
            env._reset_env(env_index)
        if no_oracle_env is not None:
            for env_index in range(no_oracle_env.n_env):
                no_oracle_env._reset_env(env_index)
        if visible_gradient_env is not None:
            for env_index in range(visible_gradient_env.n_env):
                visible_gradient_env._reset_env(env_index)
        baseline_lr_tuning_path = run_dir / "spatial_baseline_lr_tuning.json"
        with baseline_lr_tuning_path.open("w", encoding="utf-8") as handle:
            json.dump(baseline_lr_tuning_result, handle, indent=2)
        print(
            "Spatial baseline lr tuning selected: "
            f"gd={float(best_lrs['gd']):.6g}, "
            f"adam={float(best_lrs['adam']):.6g}"
        )
    else:
        current_lrs = env.get_baseline_learning_rates()
        print(
            "Spatial baseline lr tuning disabled; using configured base lr values: "
            f"gd={float(current_lrs['gd']):.6g}, "
            f"adam={float(current_lrs['adam']):.6g}"
        )
    if no_oracle_env is not None:
        _synchronize_spatial_envs(env, no_oracle_env)
    if visible_gradient_env is not None:
        _synchronize_spatial_envs(env, visible_gradient_env)
    if spatial_curriculum_enabled:
        print(
            "Spatial success curriculum enabled: "
            f"start={float(config.spatial_success_curriculum_start):.6g}, "
            f"trigger={float(config.spatial_success_curriculum_trigger_rate):.2%}, "
            f"decay={float(config.spatial_success_curriculum_decay):.1%}, "
            f"min={float(config.spatial_success_curriculum_min):.6g}, "
            f"current={float(env.success_threshold):.6g}"
        )

    model_architecture = str(config.spatial_policy_arch)
    model = PolicyValueNet(
        token_feature_dim=env.token_feature_dim,
        oracle_token_dim=env.oracle_token_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        oracle_proj_dim=config.oracle_proj_dim,
        architecture=model_architecture,
        action_space_type=action_space_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    updates_total = int(np.ceil(config.train_steps / max(1, config.n_env * config.rollout_len)))
    lr_scheduler = _build_lr_scheduler(optimizer, config, total_updates=updates_total)

    no_oracle_lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    visible_gradient_lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    if no_oracle_env is not None:
        no_oracle_model = PolicyValueNet(
            token_feature_dim=no_oracle_env.token_feature_dim,
            oracle_token_dim=no_oracle_env.oracle_token_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            oracle_proj_dim=config.oracle_proj_dim,
            architecture=model_architecture,
            action_space_type=action_space_type,
        ).to(device)
        no_oracle_optimizer = torch.optim.Adam(no_oracle_model.parameters(), lr=config.lr)
        no_oracle_lr_scheduler = _build_lr_scheduler(
            no_oracle_optimizer,
            config,
            total_updates=updates_total,
        )
    if visible_gradient_env is not None:
        visible_gradient_model = PolicyValueNet(
            token_feature_dim=visible_gradient_env.token_feature_dim,
            oracle_token_dim=visible_gradient_env.oracle_token_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            oracle_proj_dim=config.oracle_proj_dim,
            architecture=model_architecture,
            action_space_type=action_space_type,
        ).to(device)
        visible_gradient_optimizer = torch.optim.Adam(visible_gradient_model.parameters(), lr=config.lr)
        visible_gradient_lr_scheduler = _build_lr_scheduler(
            visible_gradient_optimizer,
            config,
            total_updates=updates_total,
        )

    hparams = PPOHyperParams(
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_ratio=config.clip_ratio,
        entropy_coef=config.entropy_coef,
        value_coef=config.value_coef,
        max_grad_norm=config.max_grad_norm,
        ppo_epochs=config.ppo_epochs,
        minibatches=config.minibatches,
    )

    buffer = RolloutBuffer(
        rollout_len=config.rollout_len,
        n_env=config.n_env,
        token_feature_dim=env.token_feature_dim,
        action_dim=action_dim,
        action_dtype=action_space_type,
    )
    obs = env.get_obs()
    recurrent_state = model.initial_state(batch_size=config.n_env, device=device)
    reward_norm_clip = 10.0
    reward_returns_rms = RunningMeanStd()
    reward_returns = np.zeros(config.n_env, dtype=np.float64)
    no_oracle_reward_returns_rms: RunningMeanStd | None = None
    no_oracle_reward_returns: np.ndarray | None = None
    visible_gradient_reward_returns_rms: RunningMeanStd | None = None
    visible_gradient_reward_returns: np.ndarray | None = None

    if no_oracle_env is not None:
        no_oracle_buffer = RolloutBuffer(
            rollout_len=config.rollout_len,
            n_env=config.n_env,
            token_feature_dim=no_oracle_env.token_feature_dim,
            action_dim=action_dim,
            action_dtype=action_space_type,
        )
        no_oracle_obs = no_oracle_env.get_obs()
        assert no_oracle_model is not None
        no_oracle_recurrent_state = no_oracle_model.initial_state(batch_size=config.n_env, device=device)
        no_oracle_reward_returns_rms = RunningMeanStd()
        no_oracle_reward_returns = np.zeros(config.n_env, dtype=np.float64)
    if visible_gradient_env is not None:
        visible_gradient_buffer = RolloutBuffer(
            rollout_len=config.rollout_len,
            n_env=config.n_env,
            token_feature_dim=visible_gradient_env.token_feature_dim,
            action_dim=action_dim,
            action_dtype=action_space_type,
        )
        visible_gradient_obs = visible_gradient_env.get_obs()
        assert visible_gradient_model is not None
        visible_gradient_recurrent_state = visible_gradient_model.initial_state(
            batch_size=config.n_env,
            device=device,
        )
        visible_gradient_reward_returns_rms = RunningMeanStd()
        visible_gradient_reward_returns = np.zeros(config.n_env, dtype=np.float64)

    curve_metrics: list[dict] = []
    spatial_optimization_eval: dict | None = None
    save_metrics_interval = int(config.save_metrics_interval_episodes)
    save_every_episode = save_metrics_interval <= 0
    last_row: dict[str, float | int] | None = None
    metric_csv_path = run_dir / "metrics.csv"
    metric_jsonl_path = run_dir / "metrics.jsonl"
    metric_fields = [
        "episodes",
        "steps",
        "success",
        "success_rate",
        "success_threshold",
        "avg_baseline_success_rate",
        "avg_adam_baseline_success_rate",
        "avg_no_oracle_success_rate",
        "avg_visible_gradient_success_rate",
        "final_objective",
        "avg_final_objective",
        "baseline_final_objective",
        "avg_baseline_final_objective",
        "adam_baseline_final_objective",
        "avg_adam_baseline_final_objective",
        "no_oracle_final_objective",
        "avg_no_oracle_final_objective",
        "visible_gradient_final_objective",
        "avg_visible_gradient_final_objective",
        "final_ref_distance",
        "avg_final_ref_distance",
        "baseline_final_ref_distance",
        "avg_baseline_final_ref_distance",
        "adam_baseline_final_ref_distance",
        "avg_adam_baseline_final_ref_distance",
        "no_oracle_final_ref_distance",
        "avg_no_oracle_final_ref_distance",
        "visible_gradient_final_ref_distance",
        "avg_visible_gradient_final_ref_distance",
        "avg_path_len",
        "avg_shortest_dist",
        "avg_stretch",
        "window_size",
        "policy_loss",
        "value_loss",
        "entropy",
        "lr",
        "no_oracle_lr",
        "visible_gradient_lr",
    ]

    recent_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_baseline_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_adam_baseline_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_no_oracle_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_visible_gradient_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_path_len: deque[float] = deque(maxlen=config.running_avg_window)
    recent_shortest_dist: deque[float] = deque(maxlen=config.running_avg_window)
    recent_stretch: deque[float] = deque(maxlen=config.running_avg_window)
    recent_final_objective: deque[float] = deque(maxlen=config.running_avg_window)
    recent_baseline_final_objective: deque[float] = deque(maxlen=config.running_avg_window)
    recent_adam_baseline_final_objective: deque[float] = deque(maxlen=config.running_avg_window)
    recent_no_oracle_final_objective: deque[float] = deque(maxlen=config.running_avg_window)
    recent_visible_gradient_final_objective: deque[float] = deque(maxlen=config.running_avg_window)
    recent_final_ref_distance: deque[float] = deque(maxlen=config.running_avg_window)
    recent_baseline_final_ref_distance: deque[float] = deque(maxlen=config.running_avg_window)
    recent_adam_baseline_final_ref_distance: deque[float] = deque(maxlen=config.running_avg_window)
    recent_no_oracle_final_ref_distance: deque[float] = deque(maxlen=config.running_avg_window)
    recent_visible_gradient_final_ref_distance: deque[float] = deque(maxlen=config.running_avg_window)
    all_path_lengths: list[float] = []
    all_shortest_dists: list[float] = []
    all_final_ref_distances: list[float] = []
    all_baseline_final_ref_distances: list[float] = []
    all_adam_baseline_final_ref_distances: list[float] = []
    all_no_oracle_final_ref_distances: list[float] = []
    all_visible_gradient_final_ref_distances: list[float] = []
    success_path_lengths: list[float] = []
    failure_path_lengths: list[float] = []
    curriculum_episodes_since_update = 0
    curriculum_updates = 0

    with metric_csv_path.open("w", newline="", encoding="utf-8") as csv_file, metric_jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=metric_fields)
        writer.writeheader()

        total_steps = 0
        total_episodes = 0
        total_no_oracle_episodes = 0
        total_visible_gradient_episodes = 0
        update_index = 0
        last_update_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        early_stop_triggered = False
        early_stop_reason: str | None = None
        main_first_success_episode: int | None = None
        no_oracle_first_success_episode: int | None = None
        visible_gradient_first_success_episode: int | None = None
        objective_plateau_trackers: dict[str, dict[str, float | int | bool | None]] = {
            "rl_hidden_gradient": _init_objective_plateau_tracker()
        }
        if no_oracle_model is not None:
            objective_plateau_trackers["rl_no_oracle"] = _init_objective_plateau_tracker()
        if visible_gradient_model is not None:
            objective_plateau_trackers["rl_visible_oracle"] = _init_objective_plateau_tracker()
        spatial_search_training_trace = None
        if config.spatial_fixed_start_target:
            spatial_search_training_trace = _init_spatial_search_training_trace(
                config=config,
                env=env,
                no_oracle_model=no_oracle_model,
                visible_gradient_model=visible_gradient_model,
            )
        training_wall_start = time.perf_counter()

        progress = tqdm(total=config.train_steps, desc="Training", leave=False)

        while total_steps < config.train_steps:
            buffer.reset()
            if no_oracle_buffer is not None:
                no_oracle_buffer.reset()
            if visible_gradient_buffer is not None:
                visible_gradient_buffer.reset()

            for _ in range(config.rollout_len):
                token_t, dist_t, step_t = obs_to_tensors(obs, device)
                with torch.no_grad():
                    actions_t, logprob_t, values_t, next_recurrent_state = model.act(
                        token_t,
                        dist_t,
                        step_t,
                        hidden_state=recurrent_state,
                        deterministic=False,
                    )

                actions = actions_t.cpu().numpy()
                logprobs = logprob_t.cpu().numpy()
                values = values_t.cpu().numpy()
                recurrent_state_np = None
                if recurrent_state is not None:
                    recurrent_state_np = recurrent_state.cpu().numpy()

                next_obs, rewards, dones, infos = env.step(actions)
                normalized_rewards, reward_returns = normalize_rewards_with_running_returns(
                    rewards=rewards,
                    dones=dones,
                    returns_tracker=reward_returns,
                    returns_rms=reward_returns_rms,
                    gamma=config.gamma,
                    clip_abs=reward_norm_clip,
                )

                buffer.add(
                    obs=obs,
                    action=actions,
                    logprob=logprobs,
                    reward=normalized_rewards,
                    done=dones,
                    value=values,
                    hidden_state=recurrent_state_np,
                )

                if next_recurrent_state is not None:
                    recurrent_state = next_recurrent_state.clone()
                    done_mask = torch.as_tensor(dones, device=device, dtype=torch.bool)
                    recurrent_state[done_mask] = 0.0

                obs = next_obs
                step_increment = config.n_env
                total_steps += step_increment
                progress.update(step_increment)

                if (
                    no_oracle_model is not None
                    and no_oracle_env is not None
                    and no_oracle_buffer is not None
                    and no_oracle_obs is not None
                ):
                    no_token_t, no_dist_t, no_step_t = obs_to_tensors(no_oracle_obs, device)
                    with torch.no_grad():
                        no_actions_t, no_logprob_t, no_values_t, no_next_recurrent_state = no_oracle_model.act(
                            no_token_t,
                            no_dist_t,
                            no_step_t,
                            hidden_state=no_oracle_recurrent_state,
                            deterministic=False,
                        )

                    no_actions = no_actions_t.cpu().numpy()
                    no_logprobs = no_logprob_t.cpu().numpy()
                    no_values = no_values_t.cpu().numpy()
                    no_recurrent_state_np = None
                    if no_oracle_recurrent_state is not None:
                        no_recurrent_state_np = no_oracle_recurrent_state.cpu().numpy()

                    no_next_obs, no_rewards, no_dones, no_infos = no_oracle_env.step(no_actions)
                    assert no_oracle_reward_returns is not None
                    assert no_oracle_reward_returns_rms is not None
                    no_normalized_rewards, no_oracle_reward_returns = (
                        normalize_rewards_with_running_returns(
                            rewards=no_rewards,
                            dones=no_dones,
                            returns_tracker=no_oracle_reward_returns,
                            returns_rms=no_oracle_reward_returns_rms,
                            gamma=config.gamma,
                            clip_abs=reward_norm_clip,
                        )
                    )
                    no_oracle_buffer.add(
                        obs=no_oracle_obs,
                        action=no_actions,
                        logprob=no_logprobs,
                        reward=no_normalized_rewards,
                        done=no_dones,
                        value=no_values,
                        hidden_state=no_recurrent_state_np,
                    )
                    if no_next_recurrent_state is not None:
                        no_oracle_recurrent_state = no_next_recurrent_state.clone()
                        no_done_mask = torch.as_tensor(no_dones, device=device, dtype=torch.bool)
                        no_oracle_recurrent_state[no_done_mask] = 0.0
                    no_oracle_obs = no_next_obs

                    no_done_indices = np.where(no_dones)[0]
                    for no_env_index in no_done_indices:
                        no_info = no_infos[int(no_env_index)]
                        if not no_info.get("episode_done", False):
                            continue
                        total_no_oracle_episodes += 1
                        no_final_objective = float(no_info.get("final_objective", float("nan")))
                        no_final_ref_distance = float(no_info.get("final_ref_distance", float("nan")))
                        if np.isfinite(no_final_objective):
                            recent_no_oracle_final_objective.append(no_final_objective)
                        if no_oracle_env is not None:
                            no_success_threshold = no_oracle_env.success_threshold
                            no_oracle_success_val = (
                                1.0
                                if np.isfinite(no_final_objective)
                                and no_final_objective <= no_success_threshold
                                else 0.0
                            )
                            recent_no_oracle_success.append(no_oracle_success_val)
                            if no_oracle_success_val >= 0.5 and no_oracle_first_success_episode is None:
                                no_oracle_first_success_episode = int(total_no_oracle_episodes)
                        if np.isfinite(no_final_ref_distance):
                            recent_no_oracle_final_ref_distance.append(no_final_ref_distance)
                            all_no_oracle_final_ref_distances.append(no_final_ref_distance)

                if (
                    visible_gradient_model is not None
                    and visible_gradient_env is not None
                    and visible_gradient_buffer is not None
                    and visible_gradient_obs is not None
                ):
                    vg_token_t, vg_dist_t, vg_step_t = obs_to_tensors(visible_gradient_obs, device)
                    with torch.no_grad():
                        vg_actions_t, vg_logprob_t, vg_values_t, vg_next_recurrent_state = visible_gradient_model.act(
                            vg_token_t,
                            vg_dist_t,
                            vg_step_t,
                            hidden_state=visible_gradient_recurrent_state,
                            deterministic=False,
                        )

                    vg_actions = vg_actions_t.cpu().numpy()
                    vg_logprobs = vg_logprob_t.cpu().numpy()
                    vg_values = vg_values_t.cpu().numpy()
                    vg_recurrent_state_np = None
                    if visible_gradient_recurrent_state is not None:
                        vg_recurrent_state_np = visible_gradient_recurrent_state.cpu().numpy()

                    vg_next_obs, vg_rewards, vg_dones, vg_infos = visible_gradient_env.step(vg_actions)
                    assert visible_gradient_reward_returns is not None
                    assert visible_gradient_reward_returns_rms is not None
                    vg_normalized_rewards, visible_gradient_reward_returns = (
                        normalize_rewards_with_running_returns(
                            rewards=vg_rewards,
                            dones=vg_dones,
                            returns_tracker=visible_gradient_reward_returns,
                            returns_rms=visible_gradient_reward_returns_rms,
                            gamma=config.gamma,
                            clip_abs=reward_norm_clip,
                        )
                    )
                    visible_gradient_buffer.add(
                        obs=visible_gradient_obs,
                        action=vg_actions,
                        logprob=vg_logprobs,
                        reward=vg_normalized_rewards,
                        done=vg_dones,
                        value=vg_values,
                        hidden_state=vg_recurrent_state_np,
                    )
                    if vg_next_recurrent_state is not None:
                        visible_gradient_recurrent_state = vg_next_recurrent_state.clone()
                        vg_done_mask = torch.as_tensor(vg_dones, device=device, dtype=torch.bool)
                        visible_gradient_recurrent_state[vg_done_mask] = 0.0
                    visible_gradient_obs = vg_next_obs

                    vg_done_indices = np.where(vg_dones)[0]
                    for vg_env_index in vg_done_indices:
                        vg_info = vg_infos[int(vg_env_index)]
                        if not vg_info.get("episode_done", False):
                            continue
                        total_visible_gradient_episodes += 1
                        vg_final_objective = float(vg_info.get("final_objective", float("nan")))
                        vg_final_ref_distance = float(vg_info.get("final_ref_distance", float("nan")))
                        if np.isfinite(vg_final_objective):
                            recent_visible_gradient_final_objective.append(vg_final_objective)
                        if visible_gradient_env is not None:
                            vg_success_threshold = visible_gradient_env.success_threshold
                            vg_success_val = (
                                1.0
                                if np.isfinite(vg_final_objective)
                                and vg_final_objective <= vg_success_threshold
                                else 0.0
                            )
                            recent_visible_gradient_success.append(vg_success_val)
                            if vg_success_val >= 0.5 and visible_gradient_first_success_episode is None:
                                visible_gradient_first_success_episode = int(total_visible_gradient_episodes)
                        if np.isfinite(vg_final_ref_distance):
                            recent_visible_gradient_final_ref_distance.append(vg_final_ref_distance)
                            all_visible_gradient_final_ref_distances.append(vg_final_ref_distance)

                done_indices = np.where(dones)[0]
                for env_index in done_indices:
                    info = infos[int(env_index)]
                    if not info.get("episode_done", False):
                        continue

                    total_episodes += 1
                    success = 1.0 if bool(info.get("success", False)) else 0.0
                    episode_len = float(info.get("episode_len", 0))
                    shortest_dist = float(info.get("shortest_dist", 0))
                    final_objective = float(info.get("final_objective", float("nan")))
                    baseline_final_objective = float(
                        info.get("baseline_final_objective", float("nan"))
                    )
                    adam_baseline_final_objective = float(
                        info.get("adam_baseline_final_objective", float("nan"))
                    )
                    final_ref_distance = float(info.get("final_ref_distance", float("nan")))
                    baseline_final_ref_distance = float(
                        info.get("baseline_final_ref_distance", float("nan"))
                    )
                    adam_baseline_final_ref_distance = float(
                        info.get("adam_baseline_final_ref_distance", float("nan"))
                    )
                    recent_success.append(success)
                    if success >= 0.5 and main_first_success_episode is None:
                        main_first_success_episode = int(total_episodes)
                    if env.compute_episode_baselines:
                        success_threshold = env.success_threshold
                        baseline_success = (
                            1.0
                            if np.isfinite(baseline_final_objective)
                            and baseline_final_objective <= success_threshold
                            else 0.0
                        )
                        adam_baseline_success_val = (
                            1.0
                            if np.isfinite(adam_baseline_final_objective)
                            and adam_baseline_final_objective <= success_threshold
                            else 0.0
                        )
                        recent_baseline_success.append(baseline_success)
                        recent_adam_baseline_success.append(adam_baseline_success_val)
                    recent_path_len.append(episode_len)
                    recent_shortest_dist.append(shortest_dist)
                    if np.isfinite(final_objective):
                        recent_final_objective.append(final_objective)
                    if np.isfinite(baseline_final_objective):
                        recent_baseline_final_objective.append(baseline_final_objective)
                    if np.isfinite(adam_baseline_final_objective):
                        recent_adam_baseline_final_objective.append(adam_baseline_final_objective)
                    if np.isfinite(final_ref_distance):
                        recent_final_ref_distance.append(final_ref_distance)
                        all_final_ref_distances.append(final_ref_distance)
                    if np.isfinite(baseline_final_ref_distance):
                        recent_baseline_final_ref_distance.append(baseline_final_ref_distance)
                        all_baseline_final_ref_distances.append(baseline_final_ref_distance)
                    if np.isfinite(adam_baseline_final_ref_distance):
                        recent_adam_baseline_final_ref_distance.append(adam_baseline_final_ref_distance)
                        all_adam_baseline_final_ref_distances.append(adam_baseline_final_ref_distance)
                    all_path_lengths.append(episode_len)
                    all_shortest_dists.append(shortest_dist)
                    if success >= 0.5:
                        success_path_lengths.append(episode_len)
                    else:
                        failure_path_lengths.append(episode_len)
                    if "stretch" in info:
                        recent_stretch.append(float(info["stretch"]))

                    no_oracle_final_objective = (
                        float(recent_no_oracle_final_objective[-1])
                        if len(recent_no_oracle_final_objective) > 0
                        else float("nan")
                    )
                    no_oracle_final_ref_distance = (
                        float(recent_no_oracle_final_ref_distance[-1])
                        if len(recent_no_oracle_final_ref_distance) > 0
                        else float("nan")
                    )
                    visible_gradient_final_objective = (
                        float(recent_visible_gradient_final_objective[-1])
                        if len(recent_visible_gradient_final_objective) > 0
                        else float("nan")
                    )
                    visible_gradient_final_ref_distance = (
                        float(recent_visible_gradient_final_ref_distance[-1])
                        if len(recent_visible_gradient_final_ref_distance) > 0
                        else float("nan")
                    )
                    row = {
                        "episodes": int(total_episodes),
                        "steps": int(total_steps),
                        "success": float(success),
                        "success_rate": float(np.mean(recent_success)),
                        "success_threshold": float(env.success_threshold),
                        "avg_baseline_success_rate": (
                            float(np.mean(recent_baseline_success))
                            if len(recent_baseline_success) > 0
                            else float("nan")
                        ),
                        "avg_adam_baseline_success_rate": (
                            float(np.mean(recent_adam_baseline_success))
                            if len(recent_adam_baseline_success) > 0
                            else float("nan")
                        ),
                        "avg_no_oracle_success_rate": (
                            float(np.mean(recent_no_oracle_success))
                            if len(recent_no_oracle_success) > 0
                            else float("nan")
                        ),
                        "avg_visible_gradient_success_rate": (
                            float(np.mean(recent_visible_gradient_success))
                            if len(recent_visible_gradient_success) > 0
                            else float("nan")
                        ),
                        "final_objective": float(final_objective),
                        "avg_final_objective": (
                            float(np.mean(recent_final_objective))
                            if len(recent_final_objective) > 0
                            else float("nan")
                        ),
                        "baseline_final_objective": float(baseline_final_objective),
                        "avg_baseline_final_objective": (
                            float(np.mean(recent_baseline_final_objective))
                            if len(recent_baseline_final_objective) > 0
                            else float("nan")
                        ),
                        "adam_baseline_final_objective": float(adam_baseline_final_objective),
                        "avg_adam_baseline_final_objective": (
                            float(np.mean(recent_adam_baseline_final_objective))
                            if len(recent_adam_baseline_final_objective) > 0
                            else float("nan")
                        ),
                        "no_oracle_final_objective": float(no_oracle_final_objective),
                        "avg_no_oracle_final_objective": (
                            float(np.mean(recent_no_oracle_final_objective))
                            if len(recent_no_oracle_final_objective) > 0
                            else float("nan")
                        ),
                        "visible_gradient_final_objective": float(visible_gradient_final_objective),
                        "avg_visible_gradient_final_objective": (
                            float(np.mean(recent_visible_gradient_final_objective))
                            if len(recent_visible_gradient_final_objective) > 0
                            else float("nan")
                        ),
                        "final_ref_distance": float(final_ref_distance),
                        "avg_final_ref_distance": (
                            float(np.mean(recent_final_ref_distance))
                            if len(recent_final_ref_distance) > 0
                            else float("nan")
                        ),
                        "baseline_final_ref_distance": float(baseline_final_ref_distance),
                        "avg_baseline_final_ref_distance": (
                            float(np.mean(recent_baseline_final_ref_distance))
                            if len(recent_baseline_final_ref_distance) > 0
                            else float("nan")
                        ),
                        "adam_baseline_final_ref_distance": float(adam_baseline_final_ref_distance),
                        "avg_adam_baseline_final_ref_distance": (
                            float(np.mean(recent_adam_baseline_final_ref_distance))
                            if len(recent_adam_baseline_final_ref_distance) > 0
                            else float("nan")
                        ),
                        "no_oracle_final_ref_distance": float(no_oracle_final_ref_distance),
                        "avg_no_oracle_final_ref_distance": (
                            float(np.mean(recent_no_oracle_final_ref_distance))
                            if len(recent_no_oracle_final_ref_distance) > 0
                            else float("nan")
                        ),
                        "visible_gradient_final_ref_distance": float(visible_gradient_final_ref_distance),
                        "avg_visible_gradient_final_ref_distance": (
                            float(np.mean(recent_visible_gradient_final_ref_distance))
                            if len(recent_visible_gradient_final_ref_distance) > 0
                            else float("nan")
                        ),
                        "avg_path_len": float(np.mean(recent_path_len)),
                        "avg_shortest_dist": float(np.mean(recent_shortest_dist)),
                        "avg_stretch": (
                            float(np.mean(recent_stretch))
                            if len(recent_stretch) > 0
                            else float("nan")
                        ),
                        "window_size": int(len(recent_success)),
                        "policy_loss": float(last_update_stats["policy_loss"]),
                        "value_loss": float(last_update_stats["value_loss"]),
                        "entropy": float(last_update_stats["entropy"]),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "no_oracle_lr": (
                            float(no_oracle_optimizer.param_groups[0]["lr"])
                            if no_oracle_optimizer is not None
                            else float("nan")
                        ),
                        "visible_gradient_lr": (
                            float(visible_gradient_optimizer.param_groups[0]["lr"])
                            if visible_gradient_optimizer is not None
                            else float("nan")
                        ),
                    }

                    last_row = row
                    progress.set_postfix(
                        success_rate=f"{float(row['success_rate']):.2%}",
                        refresh=True,
                    )
                    curriculum_episodes_since_update += 1
                    if (
                        spatial_curriculum_enabled
                        and len(recent_success) >= config.running_avg_window
                        and curriculum_episodes_since_update >= config.running_avg_window
                    ):
                        current_success_rate = float(row["success_rate"])
                        if (
                            current_success_rate >= float(config.spatial_success_curriculum_trigger_rate)
                            and float(env.success_threshold)
                            > float(config.spatial_success_curriculum_min) + 1e-12
                        ):
                            previous_threshold = float(env.success_threshold)
                            new_threshold = _next_spatial_success_threshold(
                                previous_threshold,
                                decay=float(config.spatial_success_curriculum_decay),
                                minimum=float(config.spatial_success_curriculum_min),
                            )
                            if new_threshold < previous_threshold - 1e-12:
                                env.success_threshold = float(new_threshold)
                                if no_oracle_env is not None:
                                    no_oracle_env.success_threshold = float(new_threshold)
                                if visible_gradient_env is not None:
                                    visible_gradient_env.success_threshold = float(new_threshold)
                                curriculum_updates += 1
                                curriculum_episodes_since_update = 0
                                tqdm.write(
                                    "Spatial success curriculum update: "
                                    f"episodes={int(total_episodes)} "
                                    f"success_rate={current_success_rate:.4f} "
                                    f"threshold {previous_threshold:.6g} -> {new_threshold:.6g}"
                                )

                    should_sample_curve = save_every_episode or (
                        save_metrics_interval > 0 and total_episodes % save_metrics_interval == 0
                    )
                    if should_sample_curve:
                        if np.isfinite(float(row["avg_final_objective"])):
                            curve_metrics.append(
                                {
                                    "episodes": int(total_episodes),
                                    "success_rate": float(row["success_rate"]),
                                    "baseline_success_rate": float(
                                        row.get("avg_baseline_success_rate", float("nan"))
                                    ),
                                    "adam_baseline_success_rate": float(
                                        row.get("avg_adam_baseline_success_rate", float("nan"))
                                    ),
                                    "no_oracle_success_rate": float(
                                        row.get("avg_no_oracle_success_rate", float("nan"))
                                    ),
                                    "visible_gradient_success_rate": float(
                                        row.get("avg_visible_gradient_success_rate", float("nan"))
                                    ),
                                    "objective_value": float(row["avg_final_objective"]),
                                    "baseline_objective_value": float(
                                        row["avg_baseline_final_objective"]
                                    ),
                                    "adam_baseline_objective_value": float(
                                        row["avg_adam_baseline_final_objective"]
                                    ),
                                    "no_oracle_objective_value": float(
                                        row["avg_no_oracle_final_objective"]
                                    ),
                                    "visible_gradient_objective_value": float(
                                        row["avg_visible_gradient_final_objective"]
                                    ),
                                    "distance_value": float(row["avg_final_ref_distance"]),
                                    "baseline_distance_value": float(
                                        row["avg_baseline_final_ref_distance"]
                                    ),
                                    "adam_baseline_distance_value": float(
                                        row["avg_adam_baseline_final_ref_distance"]
                                    ),
                                    "no_oracle_distance_value": float(
                                        row["avg_no_oracle_final_ref_distance"]
                                    ),
                                    "visible_gradient_distance_value": float(
                                        row["avg_visible_gradient_final_ref_distance"]
                                    ),
                                }
                            )
                    writer.writerow(row)
                    jsonl_file.write(json.dumps(row) + "\n")

                    if bool(config.spatial_enable_objective_plateau_early_stop):
                        plateau_reached = []
                        plateau_method_labels: list[str] = []

                        main_plateau = _update_objective_plateau_tracker(
                            tracker=objective_plateau_trackers["rl_hidden_gradient"],
                            avg_objective=float(row["avg_final_objective"]),
                            episodes_completed=int(total_episodes),
                            warmup_episodes=int(config.spatial_objective_plateau_warmup_episodes),
                            patience_episodes=int(config.spatial_objective_plateau_patience_episodes),
                            min_delta=float(config.spatial_objective_plateau_min_delta),
                        )
                        plateau_reached.append(bool(main_plateau))
                        if main_plateau:
                            plateau_method_labels.append("hidden-gradient PPO")

                        if "rl_no_oracle" in objective_plateau_trackers:
                            no_oracle_plateau = _update_objective_plateau_tracker(
                                tracker=objective_plateau_trackers["rl_no_oracle"],
                                avg_objective=float(row["avg_no_oracle_final_objective"]),
                                episodes_completed=int(total_no_oracle_episodes),
                                warmup_episodes=int(config.spatial_objective_plateau_warmup_episodes),
                                patience_episodes=int(config.spatial_objective_plateau_patience_episodes),
                                min_delta=float(config.spatial_objective_plateau_min_delta),
                            )
                            plateau_reached.append(bool(no_oracle_plateau))
                            if no_oracle_plateau:
                                plateau_method_labels.append("no-oracle PPO")

                        if "rl_visible_oracle" in objective_plateau_trackers:
                            visible_plateau = _update_objective_plateau_tracker(
                                tracker=objective_plateau_trackers["rl_visible_oracle"],
                                avg_objective=float(row["avg_visible_gradient_final_objective"]),
                                episodes_completed=int(total_visible_gradient_episodes),
                                warmup_episodes=int(config.spatial_objective_plateau_warmup_episodes),
                                patience_episodes=int(config.spatial_objective_plateau_patience_episodes),
                                min_delta=float(config.spatial_objective_plateau_min_delta),
                            )
                            plateau_reached.append(bool(visible_plateau))
                            if visible_plateau:
                                plateau_method_labels.append("visible-gradient PPO")

                        if plateau_reached and all(plateau_reached):
                            early_stop_triggered = True
                            early_stop_reason = "ppo_objective_plateau"
                            tqdm.write(
                                "Early stopping PPO training: running average objective plateaued for "
                                + ", ".join(plateau_method_labels)
                                + f" (patience={int(config.spatial_objective_plateau_patience_episodes)}, "
                                f"min_delta={float(config.spatial_objective_plateau_min_delta):.3g})."
                            )
                            break

                    if _should_early_stop_spatial_search_training(
                        config=config,
                        main_success_seen=main_first_success_episode is not None,
                        no_oracle_model=no_oracle_model,
                        no_oracle_success_seen=no_oracle_first_success_episode is not None,
                        visible_gradient_model=visible_gradient_model,
                        visible_gradient_success_seen=visible_gradient_first_success_episode is not None,
                    ):
                        early_stop_triggered = True
                        early_stop_reason = "all_search_policies_first_success"
                        tqdm.write(
                            "Early stopping fixed-task spatial training: "
                            "all active RL search policies recorded a first successful episode."
                        )
                        break

                    if config.eval_interval_episodes > 0 and total_episodes % config.eval_interval_episodes == 0:
                        avg_path_len_all = float(np.mean(all_path_lengths)) if all_path_lengths else float("nan")
                        tqdm.write(
                            "episodes="
                            f"{total_episodes} steps={total_steps} "
                            f"success_rate={float(row['success_rate']):.4f} "
                            f"success_threshold={float(env.success_threshold):.6g} "
                            f"success_rate_gd={float(row.get('avg_baseline_success_rate', float('nan'))):.4f} "
                            f"success_rate_adam={float(row.get('avg_adam_baseline_success_rate', float('nan'))):.4f} "
                            f"success_rate_no_oracle={float(row.get('avg_no_oracle_success_rate', float('nan'))):.4f} "
                            f"success_rate_visible_gradient={float(row.get('avg_visible_gradient_success_rate', float('nan'))):.4f} "
                            f"avg_E(F(z))={float(row['avg_final_objective']):.4f} "
                            f"avg_E(F(z))_gd_baseline={float(row['avg_baseline_final_objective']):.4f} "
                            f"avg_E(F(z))_adam_baseline={float(row['avg_adam_baseline_final_objective']):.4f} "
                            f"avg_E(F(z))_no_oracle={float(row['avg_no_oracle_final_objective']):.4f} "
                            f"avg_E(F(z))_visible_gradient={float(row['avg_visible_gradient_final_objective']):.4f} "
                            f"avg_path_len={avg_path_len_all:.3f}"
                        )
                    if (
                        config.spatial_plot_interval_episodes > 0
                        and total_episodes % config.spatial_plot_interval_episodes == 0
                    ):
                        maybe_save_spatial_trajectory_plot(
                            model=model,
                            env=env,
                            device=device,
                            no_oracle_model=no_oracle_model,
                            visible_gradient_model=visible_gradient_model,
                            visible_gradient_env=visible_gradient_env,
                            output_path=run_dir
                            / f"spatial_trajectory_with_gradients_epi{int(total_episodes)}.png",
                            title=(
                                f"2D trajectory on energy landscape | D={config.spatial_hidden_dim}, "
                                f"mode={config.oracle_mode}, epi={int(total_episodes)}"
                            ),
                        )

                elapsed_training_sec = float(time.perf_counter() - training_wall_start)
                _record_spatial_search_training_step(
                    trace=spatial_search_training_trace,
                    method="rl_hidden_gradient",
                    obs=obs,
                    infos=infos,
                    dones=dones,
                    total_steps=total_steps,
                    episodes_completed=total_episodes,
                    elapsed_sec=elapsed_training_sec,
                )
                _record_spatial_search_training_step(
                    trace=spatial_search_training_trace,
                    method="rl_no_oracle",
                    obs=no_oracle_obs,
                    infos=no_infos if no_oracle_model is not None else None,
                    dones=no_dones if no_oracle_model is not None else None,
                    total_steps=total_steps,
                    episodes_completed=total_no_oracle_episodes,
                    elapsed_sec=elapsed_training_sec,
                )
                _record_spatial_search_training_step(
                    trace=spatial_search_training_trace,
                    method="rl_visible_oracle",
                    obs=visible_gradient_obs,
                    infos=vg_infos if visible_gradient_model is not None else None,
                    dones=vg_dones if visible_gradient_model is not None else None,
                    total_steps=total_steps,
                    episodes_completed=total_visible_gradient_episodes,
                    elapsed_sec=elapsed_training_sec,
                )

                if early_stop_triggered:
                    break

            csv_file.flush()
            jsonl_file.flush()

            if early_stop_triggered:
                break

            token_t, dist_t, step_t = obs_to_tensors(obs, device)
            with torch.no_grad():
                _, _, last_values_t, _ = model.act(
                    token_t,
                    dist_t,
                    step_t,
                    hidden_state=recurrent_state,
                    deterministic=False,
                )
            last_values = last_values_t.cpu().numpy()

            advantages, returns = compute_gae(
                rewards=buffer.rewards,
                dones=buffer.dones,
                values=buffer.values,
                last_values=last_values,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            last_update_stats = ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                advantages=advantages,
                returns=returns,
                hparams=hparams,
                device=device,
            )
            if lr_scheduler is not None:
                lr_scheduler.step()

            if (
                no_oracle_model is not None
                and no_oracle_optimizer is not None
                and no_oracle_buffer is not None
                and no_oracle_obs is not None
            ):
                no_token_t, no_dist_t, no_step_t = obs_to_tensors(no_oracle_obs, device)
                with torch.no_grad():
                    _, _, no_last_values_t, _ = no_oracle_model.act(
                        no_token_t,
                        no_dist_t,
                        no_step_t,
                        hidden_state=no_oracle_recurrent_state,
                        deterministic=False,
                    )
                no_last_values = no_last_values_t.cpu().numpy()
                no_advantages, no_returns = compute_gae(
                    rewards=no_oracle_buffer.rewards,
                    dones=no_oracle_buffer.dones,
                    values=no_oracle_buffer.values,
                    last_values=no_last_values,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                )
                ppo_update(
                    model=no_oracle_model,
                    optimizer=no_oracle_optimizer,
                    buffer=no_oracle_buffer,
                    advantages=no_advantages,
                    returns=no_returns,
                    hparams=hparams,
                    device=device,
                )
                if no_oracle_lr_scheduler is not None:
                    no_oracle_lr_scheduler.step()
            if (
                visible_gradient_model is not None
                and visible_gradient_optimizer is not None
                and visible_gradient_buffer is not None
                and visible_gradient_obs is not None
            ):
                vg_token_t, vg_dist_t, vg_step_t = obs_to_tensors(visible_gradient_obs, device)
                with torch.no_grad():
                    _, _, vg_last_values_t, _ = visible_gradient_model.act(
                        vg_token_t,
                        vg_dist_t,
                        vg_step_t,
                        hidden_state=visible_gradient_recurrent_state,
                        deterministic=False,
                    )
                vg_last_values = vg_last_values_t.cpu().numpy()
                vg_advantages, vg_returns = compute_gae(
                    rewards=visible_gradient_buffer.rewards,
                    dones=visible_gradient_buffer.dones,
                    values=visible_gradient_buffer.values,
                    last_values=vg_last_values,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                )
                ppo_update(
                    model=visible_gradient_model,
                    optimizer=visible_gradient_optimizer,
                    buffer=visible_gradient_buffer,
                    advantages=vg_advantages,
                    returns=vg_returns,
                    hparams=hparams,
                    device=device,
                )
                if visible_gradient_lr_scheduler is not None:
                    visible_gradient_lr_scheduler.step()
            update_index += 1

        progress.close()

    if config.enable_training_plots:
        hist_context = (
            f"D={config.spatial_hidden_dim}, visible={config.spatial_visible_dim}, "
            f"mode={config.oracle_mode}, sensing={config.sensing}"
        )
        plot_path_length_histograms(
            path_lengths=all_path_lengths,
            success_path_lengths=success_path_lengths,
            failure_path_lengths=failure_path_lengths,
            output_dir=run_dir,
            title_prefix=hist_context,
        )
        maybe_save_spatial_trajectory_plot(
            model=model,
            env=env,
            device=device,
            no_oracle_model=no_oracle_model,
            visible_gradient_model=visible_gradient_model,
            visible_gradient_env=visible_gradient_env,
            output_path=run_dir / "spatial_trajectory_with_gradients.png",
            title=(
                f"2D trajectory on energy landscape | D={config.spatial_hidden_dim}, "
                f"mode={config.oracle_mode}"
            ),
        )

    spatial_optimization_eval = evaluate_spatial_optimization_curves(
            config=config,
            run_dir=run_dir,
            device=device,
            hidden_gradient_model=model,
            hidden_gradient_env=env,
            no_oracle_model=no_oracle_model,
            no_oracle_env=no_oracle_env,
            visible_gradient_model=visible_gradient_model,
            visible_gradient_env=visible_gradient_env,
        )

    avg_path_len_all = float(np.mean(all_path_lengths)) if all_path_lengths else None
    avg_shortest_dist_all = float(np.mean(all_shortest_dists)) if all_shortest_dists else None
    avg_final_ref_distance_all = (
        float(np.mean(all_final_ref_distances)) if all_final_ref_distances else None
    )
    avg_baseline_final_ref_distance_all = (
        float(np.mean(all_baseline_final_ref_distances))
        if all_baseline_final_ref_distances
        else None
    )
    avg_adam_baseline_final_ref_distance_all = (
        float(np.mean(all_adam_baseline_final_ref_distances))
        if all_adam_baseline_final_ref_distances
        else None
    )
    avg_no_oracle_final_ref_distance_all = (
        float(np.mean(all_no_oracle_final_ref_distances))
        if all_no_oracle_final_ref_distances
        else None
    )
    avg_visible_gradient_final_ref_distance_all = (
        float(np.mean(all_visible_gradient_final_ref_distances))
        if all_visible_gradient_final_ref_distances
        else None
    )
    if avg_path_len_all is not None:
        print(f"Average path length over all episodes: {avg_path_len_all:.3f}")
    if avg_shortest_dist_all is not None:
        print(f"Average proxy step distance to reference minimum: {avg_shortest_dist_all:.3f}")
    if avg_final_ref_distance_all is not None:
        print(
            "Average final Euclidean distance to reference minimum over all episodes: "
            f"{avg_final_ref_distance_all:.3f}"
        )
    if avg_baseline_final_ref_distance_all is not None:
        print(
            "Average GD baseline final Euclidean distance to reference minimum over all episodes: "
            f"{avg_baseline_final_ref_distance_all:.3f}"
        )
    if avg_adam_baseline_final_ref_distance_all is not None:
        print(
            "Average Adam baseline final Euclidean distance to reference minimum over all episodes: "
            f"{avg_adam_baseline_final_ref_distance_all:.3f}"
        )
    if avg_no_oracle_final_ref_distance_all is not None:
        print(
            "Average no-oracle baseline final Euclidean distance to reference minimum over all episodes: "
            f"{avg_no_oracle_final_ref_distance_all:.3f}"
        )
    if avg_visible_gradient_final_ref_distance_all is not None:
        print(
            "Average PPO visible-gradient final Euclidean distance to reference minimum over all episodes: "
            f"{avg_visible_gradient_final_ref_distance_all:.3f}"
        )

    summary = {
        "run_dir": str(run_dir),
        "model_architecture": model_architecture,
        "final_objective_value": (
            float(last_row["avg_final_objective"])
            if last_row is not None
            else None
        ),
        "final_baseline_objective_value": (
            float(last_row["avg_baseline_final_objective"])
            if last_row is not None
            else None
        ),
        "final_adam_baseline_objective_value": (
            float(last_row["avg_adam_baseline_final_objective"])
            if last_row is not None
            else None
        ),
        "final_no_oracle_objective_value": (
            float(last_row["avg_no_oracle_final_objective"])
            if last_row is not None
            else None
        ),
        "final_visible_gradient_objective_value": (
            float(last_row["avg_visible_gradient_final_objective"])
            if last_row is not None
            else None
        ),
        "final_distance_to_ref_value": (
            float(last_row["avg_final_ref_distance"])
            if last_row is not None
            else None
        ),
        "final_baseline_distance_to_ref_value": (
            float(last_row["avg_baseline_final_ref_distance"])
            if last_row is not None
            else None
        ),
        "final_adam_baseline_distance_to_ref_value": (
            float(last_row["avg_adam_baseline_final_ref_distance"])
            if last_row is not None
            else None
        ),
        "final_no_oracle_distance_to_ref_value": (
            float(last_row["avg_no_oracle_final_ref_distance"])
            if last_row is not None
            else None
        ),
        "final_visible_gradient_distance_to_ref_value": (
            float(last_row["avg_visible_gradient_final_ref_distance"])
            if last_row is not None
            else None
        ),
        "avg_path_len_all_episodes": avg_path_len_all,
        "avg_shortest_dist_all_tasks": avg_shortest_dist_all,
        "num_points": len(curve_metrics),
        "running_avg_window": config.running_avg_window,
        "spatial_enable_baselines": bool(config.spatial_enable_baselines),
        "spatial_early_stop_on_all_methods_success": bool(config.spatial_early_stop_on_all_methods_success),
        "spatial_enable_objective_plateau_early_stop": bool(
            config.spatial_enable_objective_plateau_early_stop
        ),
        "spatial_objective_plateau_patience_episodes": int(
            config.spatial_objective_plateau_patience_episodes
        ),
        "spatial_objective_plateau_min_delta": float(config.spatial_objective_plateau_min_delta),
        "spatial_objective_plateau_warmup_episodes": int(
            config.spatial_objective_plateau_warmup_episodes
        ),
        "main_first_success_episode": main_first_success_episode,
        "no_oracle_first_success_episode": no_oracle_first_success_episode,
        "visible_gradient_first_success_episode": visible_gradient_first_success_episode,
        "ppo_step_scale": float(config.ppo_step_scale),
        "spatial_control_budget_scale": float(env.control_budget_scale),
        "spatial_base_step_size": float(env.base_step_size),
        "spatial_effective_step_size": float(env.step_size),
        "spatial_enable_success_curriculum": bool(config.spatial_enable_success_curriculum),
        "spatial_success_curriculum_start": float(config.spatial_success_curriculum_start),
        "spatial_success_curriculum_trigger_rate": float(config.spatial_success_curriculum_trigger_rate),
        "spatial_success_curriculum_decay": float(config.spatial_success_curriculum_decay),
        "spatial_success_curriculum_min": float(config.spatial_success_curriculum_min),
        "spatial_final_success_threshold": float(env.success_threshold),
        "spatial_success_curriculum_updates": int(curriculum_updates),
        "spatial_baseline_lr_gd": float(env.baseline_lr_gd),
        "spatial_baseline_lr_adam": float(env.baseline_lr_adam),
        "spatial_baseline_lr_tuning_enabled": bool(config.spatial_tune_baseline_lrs),
        "spatial_baseline_lr_tune_tasks": int(config.spatial_baseline_lr_tune_tasks),
        "spatial_baseline_lr_candidates": str(config.spatial_baseline_lr_candidates),
        "spatial_baseline_lr_tuning_path": (
            str(baseline_lr_tuning_path) if baseline_lr_tuning_path is not None else None
        ),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "final_no_oracle_lr": (
            float(no_oracle_optimizer.param_groups[0]["lr"])
            if no_oracle_optimizer is not None
            else None
        ),
        "final_visible_gradient_lr": (
            float(visible_gradient_optimizer.param_groups[0]["lr"])
            if visible_gradient_optimizer is not None
            else None
        ),
        "early_stop_triggered": bool(early_stop_triggered),
        "early_stop_reason": early_stop_reason,
        "objective_plateau_tracker": {
            method: {
                "best_objective": tracker.get("best_objective"),
                "best_episode": tracker.get("best_episode"),
                "last_objective": tracker.get("last_objective"),
                "plateau_trigger_episode": tracker.get("plateau_trigger_episode"),
            }
            for method, tracker in objective_plateau_trackers.items()
        },
        "updates": update_index,
        "steps": total_steps,
        "episodes": total_episodes,
        "no_oracle_episodes": (total_no_oracle_episodes if no_oracle_model is not None else None),
        "visible_gradient_episodes": (
            total_visible_gradient_episodes if visible_gradient_model is not None else None
        ),
        "spatial_search_training_trace_path": None,
        "main_first_success_steps": (
            int(
                spatial_search_training_trace["methods"]["rl_hidden_gradient"]["first_success_steps"]
            )
            if (
                spatial_search_training_trace is not None
                and spatial_search_training_trace["methods"]["rl_hidden_gradient"]["first_success_steps"]
                is not None
            )
            else None
        ),
        "main_first_success_wall_time_sec": (
            float(
                spatial_search_training_trace["methods"]["rl_hidden_gradient"][
                    "first_success_wall_time_sec"
                ]
            )
            if (
                spatial_search_training_trace is not None
                and spatial_search_training_trace["methods"]["rl_hidden_gradient"][
                    "first_success_wall_time_sec"
                ]
                is not None
            )
            else None
        ),
        "no_oracle_first_success_steps": (
            int(spatial_search_training_trace["methods"]["rl_no_oracle"]["first_success_steps"])
            if (
                spatial_search_training_trace is not None
                and "rl_no_oracle" in spatial_search_training_trace["methods"]
                and spatial_search_training_trace["methods"]["rl_no_oracle"]["first_success_steps"]
                is not None
            )
            else None
        ),
        "no_oracle_first_success_wall_time_sec": (
            float(
                spatial_search_training_trace["methods"]["rl_no_oracle"][
                    "first_success_wall_time_sec"
                ]
            )
            if (
                spatial_search_training_trace is not None
                and "rl_no_oracle" in spatial_search_training_trace["methods"]
                and spatial_search_training_trace["methods"]["rl_no_oracle"][
                    "first_success_wall_time_sec"
                ]
                is not None
            )
            else None
        ),
        "visible_gradient_first_success_steps": (
            int(
                spatial_search_training_trace["methods"]["rl_visible_oracle"][
                    "first_success_steps"
                ]
            )
            if (
                spatial_search_training_trace is not None
                and "rl_visible_oracle" in spatial_search_training_trace["methods"]
                and spatial_search_training_trace["methods"]["rl_visible_oracle"][
                    "first_success_steps"
                ]
                is not None
            )
            else None
        ),
        "visible_gradient_first_success_wall_time_sec": (
            float(
                spatial_search_training_trace["methods"]["rl_visible_oracle"][
                    "first_success_wall_time_sec"
                ]
            )
            if (
                spatial_search_training_trace is not None
                and "rl_visible_oracle" in spatial_search_training_trace["methods"]
                and spatial_search_training_trace["methods"]["rl_visible_oracle"][
                    "first_success_wall_time_sec"
                ]
                is not None
            )
            else None
        ),
        "spatial_optimization_curve_tasks": (
            int(spatial_optimization_eval["num_tasks"]) if spatial_optimization_eval is not None else None
        ),
        "spatial_optimization_curve_horizon": (
            int(spatial_optimization_eval["horizon"]) if spatial_optimization_eval is not None else None
        ),
        "spatial_optimization_curve_data_path": (
            str(spatial_optimization_eval["curve_data_path"])
            if spatial_optimization_eval is not None
            else None
        ),
        "spatial_optimization_curve_by_method_plot": (
            str(spatial_optimization_eval["by_method_plot_path"])
            if spatial_optimization_eval is not None
            else None
        ),
        "spatial_optimization_curve_mean_plot": (
            str(spatial_optimization_eval["summary_plot_path"])
            if spatial_optimization_eval is not None
            else None
        ),
    }

    if spatial_search_training_trace is not None:
        search_trace_path = run_dir / "spatial_search_training_trace.json"
        with search_trace_path.open("w", encoding="utf-8") as handle:
            json.dump(spatial_search_training_trace, handle, indent=2)
        summary["spatial_search_training_trace_path"] = str(search_trace_path)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    output = {
        "summary": summary,
        "metrics": curve_metrics,
        "config": asdict(config),
        "spatial_optimization_curves": spatial_optimization_eval,
        "spatial_baseline_lr_tuning": baseline_lr_tuning_result,
    }
    if return_artifacts:
        output["model"] = model
        output["env"] = env
        if spatial_search_training_trace is not None:
            output["spatial_search_training_trace"] = spatial_search_training_trace
        if no_oracle_model is not None:
            output["no_oracle_model"] = no_oracle_model
        if no_oracle_env is not None:
            output["no_oracle_env"] = no_oracle_env
        if visible_gradient_model is not None:
            output["visible_gradient_model"] = visible_gradient_model
        if visible_gradient_env is not None:
            output["visible_gradient_env"] = visible_gradient_env
    return output


def _parse_seed_values(seed_arg: str) -> list[int]:
    raw = str(seed_arg).strip()
    if not raw:
        raise ValueError("seed must be a non-empty integer or comma-separated list of integers")

    seeds: list[int] = []
    for piece in raw.split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            seeds.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid seed value: {token!r}") from exc

    if not seeds:
        raise ValueError("seed must include at least one integer")
    return seeds


def parse_args() -> tuple[TrainConfig, list[int]]:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Spatial convex-optimization RL training")
    parser.add_argument("--spatial_hidden_dim", type=int, default=defaults.spatial_hidden_dim)
    parser.add_argument("--spatial_visible_dim", type=int, default=defaults.spatial_visible_dim)
    parser.add_argument("--spatial_coord_limit", type=int, default=defaults.spatial_coord_limit)
    parser.add_argument("--spatial_token_dim", type=int, default=defaults.spatial_token_dim)
    parser.add_argument("--spatial_token_noise_std", type=float, default=defaults.spatial_token_noise_std)
    parser.add_argument(
        "--ppo_step_scale",
        type=float,
        default=defaults.ppo_step_scale,
        help="Multiplier applied to PPO step scale: ppo_step_scale * sigmoid(raw_step).",
    )
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
    parser.add_argument(
        "--spatial_success_threshold",
        type=float,
        default=defaults.spatial_success_threshold,
    )
    parser.add_argument(
        "--disable_spatial_success_curriculum",
        action="store_true",
        help=(
            "Disable spatial PPO success-threshold curriculum and keep a fixed "
            "--spatial_success_threshold."
        ),
    )
    parser.add_argument(
        "--spatial_success_curriculum_start",
        type=float,
        default=defaults.spatial_success_curriculum_start,
        help="Initial normalized success threshold for spatial PPO curriculum.",
    )
    parser.add_argument(
        "--spatial_success_curriculum_trigger_rate",
        type=float,
        default=defaults.spatial_success_curriculum_trigger_rate,
        help="Decrease threshold when running success rate reaches this value.",
    )
    parser.add_argument(
        "--spatial_success_curriculum_decay",
        type=float,
        default=defaults.spatial_success_curriculum_decay,
        help="Fractional decrease per curriculum step (e.g., 0.2 means multiply by 0.8).",
    )
    parser.add_argument(
        "--spatial_success_curriculum_min",
        type=float,
        default=defaults.spatial_success_curriculum_min,
        help="Lower bound on spatial curriculum success threshold.",
    )
    parser.add_argument(
        "--spatial_basis_complexity",
        type=int,
        default=defaults.spatial_basis_complexity,
    )
    parser.add_argument(
        "--F_type",
        dest="spatial_f_type",
        type=str,
        choices=["FOURIER", "MLP"],
        default=defaults.spatial_f_type,
    )
    parser.add_argument(
        "--spatial_policy_arch",
        type=str,
        choices=["mlp", "gru"],
        default=defaults.spatial_policy_arch,
        help="Policy/value backbone for spatial task.",
    )
    parser.add_argument(
        "--spatial_refresh_map_each_episode",
        action="store_true",
        default=defaults.spatial_refresh_map_each_episode,
    )
    parser.add_argument(
        "--spatial_fixed_start_target",
        action="store_true",
        default=defaults.spatial_fixed_start_target,
        help="For spatial task, sample one random start/target pair and reuse it every episode.",
    )
    parser.add_argument(
        "--spatial_plot_interval_episodes",
        type=int,
        default=defaults.spatial_plot_interval_episodes,
        help="For spatial task, save trajectory snapshots every N episodes (<=0 disables)",
    )
    parser.add_argument(
        "--spatial_disable_baselines",
        action="store_true",
        help="Disable spatial GD/Adam/no-oracle baselines and train only PPO with oracle.",
    )
    parser.add_argument(
        "--disable_spatial_baseline_lr_tuning",
        action="store_true",
        help="Disable automatic lr tuning for GD/Adam spatial baselines.",
    )
    parser.add_argument(
        "--spatial_baseline_lr_candidates",
        type=str,
        default=defaults.spatial_baseline_lr_candidates,
        help=(
            "Comma-separated base-lr candidates for tuning Adam spatial baseline. "
            "GD uses fixed candidates: 0.001,0.003,0.01,0.03,0.1,0.3."
        ),
    )
    parser.add_argument(
        "--spatial_baseline_lr_tune_tasks",
        type=int,
        default=defaults.spatial_baseline_lr_tune_tasks,
        help="Number of sampled tasks used to tune each spatial baseline base-lr.",
    )
    parser.add_argument(
        "--spatial_optimization_curve_tasks",
        type=int,
        default=defaults.spatial_optimization_curve_tasks,
        help="Number of sampled spatial tasks for post-training optimization-curve evaluation.",
    )
    parser.add_argument(
        "--disable_spatial_optimization_curve_eval",
        action="store_true",
        help="Skip post-training spatial optimization-curve evaluation plots/JSON.",
    )
    parser.add_argument(
        "--enable_spatial_objective_plateau_early_stop",
        action="store_true",
        help=(
            "Early-stop PPO training when the running average achieved objective for all active "
            "PPO methods has not improved for a configured patience window."
        ),
    )
    parser.add_argument(
        "--spatial_objective_plateau_patience_episodes",
        type=int,
        default=defaults.spatial_objective_plateau_patience_episodes,
        help="Patience in completed episodes before objective-plateau early stopping triggers.",
    )
    parser.add_argument(
        "--spatial_objective_plateau_min_delta",
        type=float,
        default=defaults.spatial_objective_plateau_min_delta,
        help="Minimum decrease in running average objective required to reset plateau patience.",
    )
    parser.add_argument(
        "--spatial_objective_plateau_warmup_episodes",
        type=int,
        default=defaults.spatial_objective_plateau_warmup_episodes,
        help="Do not evaluate objective-plateau early stopping before this many episodes.",
    )
    parser.add_argument("--n_env", type=int, default=defaults.n_env)
    parser.add_argument("--train_steps", type=int, default=defaults.train_steps)
    parser.add_argument("--rollout_len", type=int, default=defaults.rollout_len)
    parser.add_argument("--algo", type=str, default=defaults.algo)
    parser.add_argument("--sensing", type=str, choices=["S0", "S1"], default=defaults.sensing)
    parser.add_argument("--reward_noise_std", type=float, default=defaults.reward_noise_std)
    parser.add_argument(
        "--oracle_mode",
        type=str,
        choices=sorted(SPATIAL_ORACLE_MODES - {"visible_gradient"}),
        default=defaults.oracle_mode,
    )
    parser.add_argument("--running_avg_window", type=int, default=defaults.running_avg_window)
    parser.add_argument(
        "--save_metrics_interval_episodes",
        type=int,
        default=defaults.save_metrics_interval_episodes,
        help=(
            "Sample curve metrics every N episodes for compact summaries "
            "(<=0 samples every episode). metrics.csv/jsonl are always per-episode."
        ),
    )
    parser.add_argument("--eval_interval_episodes", type=int, default=defaults.eval_interval_episodes)
    parser.add_argument("--eval_episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--max_horizon", type=int, default=defaults.max_horizon)
    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument("--run_name", type=str, default=defaults.run_name)
    parser.add_argument(
        "--seed",
        type=str,
        default=str(defaults.seed),
        help="Single seed (e.g. 123) or comma-separated seeds (e.g. 123,42,22)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.device,
        help="Device for training (e.g. cpu, cuda, cuda:0). Falls back to CPU if CUDA unavailable.",
    )

    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["none", "constant", "linear", "cosine"],
        default=defaults.lr_scheduler,
        help="Learning-rate schedule stepped once per PPO update.",
    )
    parser.add_argument(
        "--lr_min_factor",
        type=float,
        default=defaults.lr_min_factor,
        help="Minimum LR multiplier for decay schedulers.",
    )
    parser.add_argument(
        "--lr_warmup_updates",
        type=int,
        default=defaults.lr_warmup_updates,
        help="Number of PPO updates for linear warmup at start.",
    )
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae_lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--clip_ratio", type=float, default=defaults.clip_ratio)
    parser.add_argument("--entropy_coef", type=float, default=defaults.entropy_coef)
    parser.add_argument("--value_coef", type=float, default=defaults.value_coef)
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--ppo_epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)

    parser.add_argument("--hidden_dim", type=int, default=defaults.hidden_dim)
    parser.add_argument(
        "--oracle_proj_dim",
        type=int,
        default=defaults.oracle_proj_dim,
        help="Oracle-token projection width before the policy trunk (0 disables projection).",
    )
    parser.add_argument("--token_embed_dim", type=int, default=defaults.token_embed_dim)
    parser.add_argument("--s1_step_penalty", type=float, default=defaults.s1_step_penalty)
    parser.add_argument(
        "--disable_training_plots",
        action="store_true",
        help="Skip automatic per-run matplotlib artifacts (path-length/trajectory PNGs).",
    )

    args = parser.parse_args()
    if int(args.oracle_proj_dim) < 0:
        raise ValueError("oracle_proj_dim must be >= 0")
    if int(args.spatial_objective_plateau_patience_episodes) < 1:
        raise ValueError("spatial_objective_plateau_patience_episodes must be >= 1")
    if int(args.spatial_objective_plateau_warmup_episodes) < 0:
        raise ValueError("spatial_objective_plateau_warmup_episodes must be >= 0")
    if float(args.spatial_objective_plateau_min_delta) < 0.0:
        raise ValueError("spatial_objective_plateau_min_delta must be >= 0")
    seed_values = _parse_seed_values(args.seed)
    config = TrainConfig(
        spatial_hidden_dim=args.spatial_hidden_dim,
        spatial_visible_dim=args.spatial_visible_dim,
        spatial_coord_limit=args.spatial_coord_limit,
        spatial_token_dim=args.spatial_token_dim,
        spatial_token_noise_std=args.spatial_token_noise_std,
        ppo_step_scale=args.ppo_step_scale,
        spatial_step_size=args.spatial_step_size,
        spatial_success_threshold=args.spatial_success_threshold,
        spatial_enable_success_curriculum=not args.disable_spatial_success_curriculum,
        spatial_success_curriculum_start=args.spatial_success_curriculum_start,
        spatial_success_curriculum_trigger_rate=args.spatial_success_curriculum_trigger_rate,
        spatial_success_curriculum_decay=args.spatial_success_curriculum_decay,
        spatial_success_curriculum_min=args.spatial_success_curriculum_min,
        spatial_basis_complexity=args.spatial_basis_complexity,
        spatial_f_type=args.spatial_f_type,
        spatial_policy_arch=args.spatial_policy_arch,
        spatial_refresh_map_each_episode=args.spatial_refresh_map_each_episode,
        spatial_fixed_start_target=args.spatial_fixed_start_target,
        spatial_plot_interval_episodes=args.spatial_plot_interval_episodes,
        spatial_enable_baselines=not args.spatial_disable_baselines,
        spatial_tune_baseline_lrs=not args.disable_spatial_baseline_lr_tuning,
        spatial_enable_objective_plateau_early_stop=args.enable_spatial_objective_plateau_early_stop,
        spatial_objective_plateau_patience_episodes=args.spatial_objective_plateau_patience_episodes,
        spatial_objective_plateau_min_delta=args.spatial_objective_plateau_min_delta,
        spatial_objective_plateau_warmup_episodes=args.spatial_objective_plateau_warmup_episodes,
        spatial_baseline_lr_candidates=args.spatial_baseline_lr_candidates,
        spatial_baseline_lr_tune_tasks=args.spatial_baseline_lr_tune_tasks,
        spatial_optimization_curve_tasks=args.spatial_optimization_curve_tasks,
        spatial_enable_optimization_curve_eval=not args.disable_spatial_optimization_curve_eval,
        n_env=args.n_env,
        train_steps=args.train_steps,
        rollout_len=args.rollout_len,
        algo=args.algo,
        sensing=args.sensing,
        reward_noise_std=args.reward_noise_std,
        oracle_mode=args.oracle_mode,
        running_avg_window=args.running_avg_window,
        save_metrics_interval_episodes=args.save_metrics_interval_episodes,
        eval_interval_episodes=args.eval_interval_episodes,
        eval_episodes=args.eval_episodes,
        max_horizon=args.max_horizon,
        logdir=args.logdir,
        run_name=args.run_name,
        seed=seed_values[0],
        device=args.device,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_min_factor=args.lr_min_factor,
        lr_warmup_updates=args.lr_warmup_updates,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        minibatches=args.minibatches,
        hidden_dim=args.hidden_dim,
        oracle_proj_dim=args.oracle_proj_dim,
        token_embed_dim=args.token_embed_dim,
        s1_step_penalty=args.s1_step_penalty,
        enable_training_plots=not args.disable_training_plots,
    )
    return config, seed_values


def main() -> None:
    config, seed_values = parse_args()
    if len(seed_values) == 1:
        output = run_training(config)
        print(json.dumps(output["summary"], indent=2))
        return

    summaries: list[dict] = []
    per_seed_spatial_curves: list[tuple[int, dict]] = []
    base_run_name = config.run_name.strip()
    for seed in seed_values:
        run_name = f"{base_run_name}_seed{seed}" if base_run_name else ""
        seed_config = replace(config, seed=int(seed), run_name=run_name)
        output = run_training(seed_config)
        summaries.append(output["summary"])
        curve_payload = output.get("spatial_optimization_curves")
        if curve_payload is not None:
            per_seed_spatial_curves.append((int(seed), curve_payload))

    aggregate = {
        "num_seeds": len(seed_values),
        "seeds": [int(s) for s in seed_values],
        "runs": summaries,
    }
    summary_stem = base_run_name or (
        f"spatial_{config.oracle_mode}_{config.sensing}_multi_seed"
    )

    if per_seed_spatial_curves:
        per_method_seed_curves: dict[str, list[np.ndarray]] = {}
        method_labels: dict[str, str] = {}
        for _, curve_payload in per_seed_spatial_curves:
            payload_methods = curve_payload.get("methods", {})
            labels = curve_payload.get("method_labels", {})
            for method_key, label in labels.items():
                method_labels[str(method_key)] = str(label)
            for method_key, stats in payload_methods.items():
                mean_curve = np.asarray(stats.get("mean_curve", []), dtype=np.float64).reshape(-1)
                if mean_curve.size < 2:
                    continue
                per_method_seed_curves.setdefault(method_key, []).append(mean_curve)

        overall_method_mean: dict[str, np.ndarray] = {}
        overall_method_std: dict[str, np.ndarray] = {}
        seed_counts: dict[str, int] = {}
        for method_key, curves in per_method_seed_curves.items():
            if not curves:
                continue
            min_len = min(curve.shape[0] for curve in curves)
            if min_len < 2:
                continue
            trimmed = np.stack([curve[:min_len] for curve in curves], axis=0)
            overall_method_mean[method_key] = np.nanmean(trimmed, axis=0).astype(np.float32)
            overall_method_std[method_key] = np.nanstd(trimmed, axis=0).astype(np.float32)
            seed_counts[method_key] = int(trimmed.shape[0])

        if overall_method_mean:
            overall_plot_path = Path(config.logdir) / f"{summary_stem}_spatial_optimization_curves_over_seeds.png"
            overall_json_path = Path(config.logdir) / f"{summary_stem}_spatial_optimization_curves_over_seeds.json"
            plot_spatial_optimization_curve_summary(
                method_mean_curves=overall_method_mean,
                method_std_curves=overall_method_std,
                output_path=overall_plot_path,
                title=(
                    f"Spatial optimization curves averaged over seeds (num_seeds={len(per_seed_spatial_curves)})"
                ),
                method_labels=method_labels if method_labels else SPATIAL_OPTIMIZATION_METHOD_LABELS,
                y_label="Normalized objective E(F(z))",
                x_label="Optimization step",
            )

            overall_payload = {
                "task": "spatial",
                "num_seeds": len(per_seed_spatial_curves),
                "seeds": [seed for seed, _ in per_seed_spatial_curves],
                "method_order": list(SPATIAL_OPTIMIZATION_METHOD_ORDER),
                "method_labels": (
                    method_labels if method_labels else dict(SPATIAL_OPTIMIZATION_METHOD_LABELS)
                ),
                "seed_counts_by_method": seed_counts,
                "methods": {
                    method_key: {
                        "mean_curve": [float(v) for v in overall_method_mean[method_key]],
                        "std_curve": [float(v) for v in overall_method_std[method_key]],
                    }
                    for method_key in overall_method_mean
                },
                "overall_plot_path": str(overall_plot_path),
            }
            with overall_json_path.open("w", encoding="utf-8") as handle:
                json.dump(overall_payload, handle, indent=2)
            aggregate["spatial_optimization_curves_over_seeds_plot"] = str(overall_plot_path)
            aggregate["spatial_optimization_curves_over_seeds_json"] = str(overall_json_path)

    aggregate_path = Path(config.logdir) / f"{summary_stem}_summary.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print(
        json.dumps(
            {
                "multi_seed_summary_path": str(aggregate_path),
                "num_seeds": len(seed_values),
                "seeds": [int(s) for s in seed_values],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
