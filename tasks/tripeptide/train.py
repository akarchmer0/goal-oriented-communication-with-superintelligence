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
from .energy import load_energy_surface
from .env import VectorizedTripeptideEnv
from .learned_lift import LearnedLiftOracle
from .lifting_map import LiftingMap
from .model import PolicyValueNet
from .oracle import ORACLE_MODES, SpatialOracle
from .plotting import (
    plot_path_length_histograms,
    plot_spatial_optimization_curve_summary,
    plot_spatial_optimization_curves_by_method,
    plot_spatial_trajectory_with_gradients,
)
from .ppo import PPOHyperParams, RolloutBuffer, compute_gae, ppo_update
from .sdp_oracle import solve_sdp_oracle, validate_sdp_solution


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


OPTIMIZATION_METHOD_ORDER = [
    "gd",
    "adam",
    "rl_no_oracle",
    "rl_visible_oracle",
    "rl_hidden_gradient",
]
GD_TUNING_LRS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)

OPTIMIZATION_METHOD_LABELS = {
    "gd": "GD",
    "adam": "Adam",
    "rl_no_oracle": "RL no oracle",
    "rl_visible_oracle": "RL visible oracle",
    "rl_hidden_gradient": "RL hidden gradient",
}


# ---------------------------------------------------------------------------
# Baseline rollout helpers (adapted for torus)
# ---------------------------------------------------------------------------

def _rollout_descent_curve(
    env: VectorizedTripeptideEnv,
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


def _rollout_adam_curve(
    env: VectorizedTripeptideEnv,
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
def _rollout_policy_curve(
    model: PolicyValueNet,
    env: VectorizedTripeptideEnv,
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
            token_t, dist_t, step_t, hidden_state=hidden_state, deterministic=True
        )
        action = action_t.squeeze(0).cpu().numpy()
        state = env._apply_action(state, action)
        curve[step + 1] = float(env._normalized_objective_value(state, env_index=env_index))
    return curve


def _compute_curve_stats(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if curves.ndim != 2 or curves.shape[0] == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    mean_curve = np.nanmean(curves, axis=0).astype(np.float32)
    std_curve = np.nanstd(curves, axis=0).astype(np.float32)
    return mean_curve, std_curve


# ---------------------------------------------------------------------------
# Baseline LR tuning
# ---------------------------------------------------------------------------

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
        raise ValueError("baseline_lr_candidates must include at least one positive value")
    return sorted(set(float(v) for v in values))


def _rollout_baseline_with_projection_stats(
    *,
    env: VectorizedTripeptideEnv,
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
        state = env._apply_baseline_optimizer_step(state, update)
        executed_steps += 1
        if env._is_success(state, env_index=env_index):
            break
    final_objective = float(env._objective_value(state, env_index=env_index))
    final_ref_distance = float(env._reference_distance(state, env_index=env_index))
    projection_rate = float(projected_steps / max(1, executed_steps))
    return final_objective, final_ref_distance, projection_rate


def _sample_tuning_task_snapshots(
    env: VectorizedTripeptideEnv,
    num_tasks: int,
) -> list[dict]:
    tasks: list[dict] = []
    for _ in range(max(1, int(num_tasks))):
        spec = env.sample_episode_spec(env_index=0)
        tasks.append({
            "start_xy": spec.source.copy().astype(np.float32),
            "horizon": int(spec.horizon),
        })
    return tasks


def _score_baseline_lr_candidate(
    env: VectorizedTripeptideEnv,
    method: str,
    candidate_lr: float,
    tasks: list[dict],
) -> dict[str, float]:
    final_objectives: list[float] = []
    final_ref_distances: list[float] = []
    projection_rates: list[float] = []
    successes: list[float] = []
    for task in tasks:
        start_xy = np.asarray(task["start_xy"], dtype=np.float32)
        horizon = int(task["horizon"])
        final_value_raw, final_ref_distance, projection_rate = _rollout_baseline_with_projection_stats(
            env=env, method=method, start_xy=start_xy, horizon=horizon,
            env_index=0, base_lr=candidate_lr,
        )
        final_value = float(env._normalized_objective_from_raw(float(final_value_raw), env_index=0))
        final_objectives.append(final_value)
        final_ref_distances.append(float(final_ref_distance))
        projection_rates.append(float(projection_rate))
        successes.append(1.0 if final_value <= float(env.success_threshold) else 0.0)

    return {
        "candidate_lr": float(candidate_lr),
        "avg_final_ref_distance": float(np.mean(final_ref_distances)) if final_ref_distances else float("inf"),
        "avg_final_objective": float(np.mean(final_objectives)) if final_objectives else float("inf"),
        "avg_projection_rate": float(np.mean(projection_rates)) if projection_rates else float("inf"),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
    }


def tune_baseline_learning_rates(
    *,
    env: VectorizedTripeptideEnv,
    candidate_lrs: list[float],
    num_tasks: int,
    seed: int,
) -> dict:
    tasks = _sample_tuning_task_snapshots(env=env, num_tasks=max(1, int(num_tasks)))
    methods = ("gd", "adam")
    per_method_scores: dict[str, list[dict[str, float]]] = {method: [] for method in methods}
    candidate_lrs_by_method: dict[str, list[float]] = {
        "gd": [float(v) for v in GD_TUNING_LRS],
        "adam": [float(v) for v in candidate_lrs],
    }
    best_lrs: dict[str, float] = {}

    for method in methods:
        for lr in candidate_lrs_by_method[method]:
            score = _score_baseline_lr_candidate(env=env, method=method, candidate_lr=float(lr), tasks=tasks)
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


# ---------------------------------------------------------------------------
# Trajectory baselines (for plotting)
# ---------------------------------------------------------------------------

def rollout_gradient_descent_baseline(
    env: VectorizedTripeptideEnv,
    start_xy: np.ndarray,
    horizon: int,
    base_lr: float | None = None,
) -> np.ndarray:
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


def rollout_adam_baseline(
    env: VectorizedTripeptideEnv,
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


def rollout_random_search_baseline(
    env: VectorizedTripeptideEnv,
    horizon: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random search: sample random points on the torus and keep the best."""
    local_rng = rng if rng is not None else np.random.default_rng(0)
    TWO_PI = 2.0 * np.pi
    best_state = local_rng.uniform(0.0, TWO_PI, size=env.visible_dim).astype(np.float32)
    best_obj = env._objective_value(best_state)
    trajectory = [best_state.copy()]
    for _ in range(int(horizon)):
        candidate = local_rng.uniform(0.0, TWO_PI, size=env.visible_dim).astype(np.float32)
        obj = env._objective_value(candidate)
        if obj < best_obj:
            best_state = candidate
            best_obj = obj
        trajectory.append(best_state.copy())
    return np.asarray(trajectory, dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_policy(
    model: PolicyValueNet,
    env: VectorizedTripeptideEnv,
    eval_episodes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    success_count = 0
    path_lengths: list[int] = []
    stretches: list[float] = []
    for _ in range(eval_episodes):
        spec = env.sample_episode_spec(env_index=0)
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
                token_t, dist_t, step_t, hidden_state=hidden_state, deterministic=True
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
def collect_trajectory(
    model: PolicyValueNet,
    env: VectorizedTripeptideEnv,
    device: torch.device,
    no_oracle_model: PolicyValueNet | None = None,
    visible_gradient_model: PolicyValueNet | None = None,
    visible_gradient_env: VectorizedTripeptideEnv | None = None,
) -> dict[str, np.ndarray] | None:
    model.eval()
    if no_oracle_model is not None:
        no_oracle_model.eval()
    if visible_gradient_model is not None:
        visible_gradient_model.eval()

    spec = env.sample_episode_spec(env_index=0)
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
            token_t, dist_t, step_t, hidden_state=hidden_state, deterministic=True
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
                token_t, dist_t, step_t, hidden_state=no_hidden_state, deterministic=True
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
                token_t, dist_t, step_t, hidden_state=visible_hidden_state, deterministic=True
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
        "baseline_trajectory_xy": rollout_gradient_descent_baseline(
            env=env, start_xy=spec.source, horizon=spec.horizon, base_lr=env.baseline_lr_gd
        ),
        "adam_baseline_trajectory_xy": rollout_adam_baseline(
            env=env, start_xy=spec.source, horizon=spec.horizon, base_lr=env.baseline_lr_adam
        ),
        "basin_hopping_trajectory_xy": None,
    }
    if no_oracle_trajectory_xy is not None and no_oracle_trajectory_xy.shape[0] >= 2:
        result["no_oracle_trajectory_xy"] = no_oracle_trajectory_xy
    if visible_gradient_trajectory_xy is not None and visible_gradient_trajectory_xy.shape[0] >= 2:
        result["visible_gradient_trajectory_xy"] = visible_gradient_trajectory_xy
    return result


def maybe_save_trajectory_plot(
    model: PolicyValueNet,
    env: VectorizedTripeptideEnv,
    device: torch.device,
    output_path: Path,
    title: str,
    no_oracle_model: PolicyValueNet | None = None,
    visible_gradient_model: PolicyValueNet | None = None,
    visible_gradient_env: VectorizedTripeptideEnv | None = None,
) -> None:
    trace = collect_trajectory(
        model, env, device,
        no_oracle_model=no_oracle_model,
        visible_gradient_model=visible_gradient_model,
        visible_gradient_env=visible_gradient_env,
    )
    if trace is None:
        return

    grid_x, grid_y, grid_energy = env.energy_landscape_grid(resolution=150, env_index=0)
    grid_energy_normalized = np.clip(
        grid_energy / max(float(env.max_objective_env[0]), 1e-8), 0.0, 1.0
    ).astype(np.float32)

    plot_spatial_trajectory_with_gradients(
        trajectory_xy=trace["trajectory_xy"],
        gradient_xy=trace["gradient_xy"],
        move_vectors_xy=trace["move_vectors_xy"],
        target_xy=trace["target_xy"],
        baseline_trajectory_xy=trace["baseline_trajectory_xy"],
        adam_baseline_trajectory_xy=trace["adam_baseline_trajectory_xy"],
        basin_hopping_trajectory_xy=trace.get("basin_hopping_trajectory_xy"),
        no_oracle_trajectory_xy=trace.get("no_oracle_trajectory_xy"),
        visible_gradient_trajectory_xy=trace.get("visible_gradient_trajectory_xy"),
        output_path=output_path,
        title=title,
        landscape_x=grid_x,
        landscape_y=grid_y,
        landscape_energy=grid_energy_normalized,
        landscape_label="Normalized E(phi,psi)",
    )


def evaluate_optimization_curves(
    *,
    config: TrainConfig,
    run_dir: Path,
    device: torch.device,
    hidden_gradient_model: PolicyValueNet,
    hidden_gradient_env: VectorizedTripeptideEnv,
    no_oracle_model: PolicyValueNet | None,
    no_oracle_env: VectorizedTripeptideEnv | None,
    visible_gradient_model: PolicyValueNet | None,
    visible_gradient_env: VectorizedTripeptideEnv | None,
) -> dict | None:
    if not config.enable_optimization_curve_eval:
        return None

    num_tasks = max(1, int(config.optimization_curve_tasks))
    horizon = max(1, int(config.max_horizon))
    method_curves_lists: dict[str, list[np.ndarray]] = {
        method_key: [] for method_key in OPTIMIZATION_METHOD_ORDER
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
        spec = hidden_gradient_env.sample_episode_spec(env_index=0)
        start_xy = spec.source.copy().astype(np.float32)

        method_curves_lists["gd"].append(
            _rollout_descent_curve(env=hidden_gradient_env, start_xy=start_xy, horizon=horizon,
                                   env_index=0, base_lr=hidden_gradient_env.baseline_lr_gd)
        )
        method_curves_lists["adam"].append(
            _rollout_adam_curve(env=hidden_gradient_env, start_xy=start_xy, horizon=horizon,
                               env_index=0, base_lr=hidden_gradient_env.baseline_lr_adam)
        )

        if config.oracle_mode == "convex_gradient":
            method_curves_lists["rl_hidden_gradient"].append(
                _rollout_policy_curve(model=hidden_gradient_model, env=hidden_gradient_env,
                                      device=device, start_xy=start_xy, horizon=horizon, env_index=0)
            )

        if no_oracle_model is not None and no_oracle_env is not None:
            method_curves_lists["rl_no_oracle"].append(
                _rollout_policy_curve(model=no_oracle_model, env=no_oracle_env,
                                      device=device, start_xy=start_xy, horizon=horizon, env_index=0)
            )

        if visible_gradient_model is not None and visible_gradient_env is not None:
            method_curves_lists["rl_visible_oracle"].append(
                _rollout_policy_curve(model=visible_gradient_model, env=visible_gradient_env,
                                      device=device, start_xy=start_xy, horizon=horizon, env_index=0)
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
    for method_key in OPTIMIZATION_METHOD_ORDER:
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

    by_method_plot_path = run_dir / "optimization_curves_by_method.png"
    summary_plot_path = run_dir / "optimization_curves_mean_std.png"
    plot_spatial_optimization_curves_by_method(
        method_curves=method_curves, output_path=by_method_plot_path,
        title=f"Tripeptide optimization curves per method (seed={config.seed}, tasks={num_tasks})",
        method_labels=OPTIMIZATION_METHOD_LABELS,
    )
    plot_spatial_optimization_curve_summary(
        method_mean_curves=method_means, method_std_curves=method_stds,
        output_path=summary_plot_path,
        title=f"Tripeptide optimization mean +/- std (seed={config.seed}, tasks={num_tasks})",
        method_labels=OPTIMIZATION_METHOD_LABELS,
    )

    methods_payload: dict[str, dict] = {}
    for method_key in OPTIMIZATION_METHOD_ORDER:
        if method_key not in method_curves:
            continue
        methods_payload[method_key] = {
            "label": OPTIMIZATION_METHOD_LABELS.get(method_key, method_key),
            "mean_curve": [float(v) for v in method_means[method_key]],
            "std_curve": [float(v) for v in method_stds[method_key]],
            "task_curves": method_curves[method_key].astype(np.float64).tolist(),
        }

    curve_data_path = run_dir / "optimization_curves.json"
    curve_data = {
        "task": "tripeptide",
        "seed": int(config.seed),
        "num_tasks": int(num_tasks),
        "horizon": int(horizon),
        "baseline_lrs": {
            "gd": float(hidden_gradient_env.baseline_lr_gd),
            "adam": float(hidden_gradient_env.baseline_lr_adam),
        },
        "steps": list(range(horizon + 1)),
        "method_order": list(OPTIMIZATION_METHOD_ORDER),
        "available_methods": list(method_curves.keys()),
        "missing_methods": missing_methods,
        "methods": methods_payload,
    }
    with curve_data_path.open("w", encoding="utf-8") as handle:
        json.dump(curve_data, handle, indent=2)

    return {
        "num_tasks": int(num_tasks),
        "horizon": int(horizon),
        "available_methods": list(method_curves.keys()),
        "missing_methods": missing_methods,
        "method_order": list(OPTIMIZATION_METHOD_ORDER),
        "method_labels": dict(OPTIMIZATION_METHOD_LABELS),
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


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Synchronize envs (no torus_offset since tripeptide doesn't use torus rotation)
# ---------------------------------------------------------------------------

def _synchronize_envs(
    reference_env: VectorizedTripeptideEnv,
    target_env: VectorizedTripeptideEnv,
) -> None:
    if reference_env.n_env != target_env.n_env:
        raise ValueError("Cannot synchronize envs with different n_env")
    target_env.max_objective = float(reference_env.max_objective)
    target_env.max_objective_env[:] = reference_env.max_objective_env.astype(np.float32)
    target_env.current_xy[:] = reference_env.current_xy.astype(np.float32)
    target_env.initial_xy[:] = reference_env.initial_xy.astype(np.float32)
    target_env.steps[:] = reference_env.steps.astype(np.int32)
    target_env.horizons[:] = reference_env.horizons.astype(np.int32)
    target_env.initial_dist[:] = reference_env.initial_dist.astype(np.int32)
    target_env.initial_objective[:] = reference_env.initial_objective.astype(np.float32)
    target_env.completed_episodes = int(reference_env.completed_episodes)
    target_env.success_threshold = float(reference_env.success_threshold)
    target_env.s_star[:] = reference_env.s_star.astype(np.float32)
    target_env.reference_min_xy_env[:] = reference_env.reference_min_xy_env.astype(np.float32)


# ---------------------------------------------------------------------------
# Resolve device
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        warnings.warn("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(config: TrainConfig, return_artifacts: bool = False) -> dict:
    if config.algo != "ppo":
        raise ValueError("Only --algo ppo is implemented")
    if config.running_avg_window < 1:
        raise ValueError("running_avg_window must be >= 1")
    if config.oracle_mode not in ORACLE_MODES:
        raise ValueError(f"oracle_mode={config.oracle_mode!r} is invalid")
    if config.oracle_mode == "visible_gradient":
        raise ValueError(
            "oracle_mode='visible_gradient' is reserved for the parallel baseline; "
            "run main PPO with oracle_mode='convex_gradient'."
        )

    set_global_seed(config.seed)
    device = _resolve_device(config.device)
    run_dir = config.resolve_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    # Load energy surface and compute oracle target s*
    energy_surface = load_energy_surface(
        config.energy_json,
        use_synthetic=config.use_synthetic_fallback,
        d=config.synthetic_d,
        K=config.synthetic_K,
        n_minima=config.synthetic_n_minima,
        seed=config.seed,
    )

    learned_lift_oracle: LearnedLiftOracle | None = None

    if config.learned_lift:
        # --- Learned lifting map: encoder F + ICNN decoder G -----------------
        D = config.learned_lift_D
        enc_hidden = [int(x) for x in config.learned_lift_encoder_hidden.split(",") if x.strip()]
        dec_hidden = [int(x) for x in config.learned_lift_decoder_hidden.split(",") if x.strip()]
        print(f"  Training learned lift: D={D}, encoder={enc_hidden}, decoder={dec_hidden}")
        learned_lift_oracle = LearnedLiftOracle.train_from_energy_surface(
            energy_surface=energy_surface,
            d=energy_surface.d,
            D=D,
            encoder_hidden_dims=tuple(enc_hidden),
            decoder_hidden_dims=tuple(dec_hidden),
            n_train=config.learned_lift_n_train,
            n_epochs=config.learned_lift_n_epochs,
            lr=config.learned_lift_lr,
            seed=config.seed + 7,
        )
        # Use a dummy Fourier lifting map for baseline gradient computations;
        # the RL agent uses the learned oracle instead.
        lifting_map = LiftingMap(d=energy_surface.d, K_map=config.K_map)
        hidden_dim = D
        s_star_sdp = learned_lift_oracle.s_star
        sdp_bound = float("nan")
        sdp_status = "learned_lift"
    else:
        lifting_map = LiftingMap(d=energy_surface.d, K_map=config.K_map)
        if config.use_simple_s_star:
            # Norm-ball oracle: s* = -R * c / ||c||
            # R = sqrt(N_freq) since ||F(z)||^2 = N_freq for any z on the torus
            c_vec = lifting_map.energy_as_linear(energy_surface)
            c_norm = float(np.linalg.norm(c_vec))
            R = float(np.sqrt(lifting_map.N_freq))
            s_star_sdp = (-R * c_vec / max(c_norm, 1e-12)).astype(np.float32)
            sdp_bound = float(np.dot(c_vec, s_star_sdp.astype(np.float64)) + energy_surface.c0)
            sdp_status = "simple_norm_ball"
            print(f"  Using simple norm-ball oracle: R={R:.2f}, ||c||={c_norm:.4f}")
            print(f"  Norm-ball energy bound: {sdp_bound:.4f}")
            validate_sdp_solution(s_star_sdp, lifting_map, energy_surface)
        else:
            s_star_sdp, sdp_bound, sdp_status = solve_sdp_oracle(
                lifting_map, energy_surface, K_relax=config.K_relax
            )
            validate_sdp_solution(s_star_sdp, lifting_map, energy_surface)
        hidden_dim = lifting_map.D

    token_dim = config.spatial_token_dim

    oracle = SpatialOracle(
        hidden_dim=hidden_dim,
        token_dim=token_dim,
        mode=config.oracle_mode,
        seed=config.seed + 29,
        token_noise_std=config.token_noise_std,
    )
    env = VectorizedTripeptideEnv(
        lifting_map=lifting_map,
        energy_surface=energy_surface,
        s_star_sdp=s_star_sdp,
        oracle=oracle,
        n_env=config.n_env,
        sensing=config.sensing,
        max_horizon=config.max_horizon,
        seed=config.seed + 41,
        s1_step_penalty=config.s1_step_penalty,
        reward_noise_std=config.reward_noise_std,
        step_size=config.step_size,
        ppo_step_scale=config.ppo_step_scale,
        success_threshold=config.success_threshold,
        compute_episode_baselines=config.enable_baselines,
        lattice_rl=config.lattice_rl,
        lattice_granularity=config.lattice_granularity,
        learned_lift_oracle=learned_lift_oracle,
    )
    action_dim = env.action_dim
    action_space_type = "discrete" if config.lattice_rl else "continuous"

    no_oracle_env: VectorizedTripeptideEnv | None = None
    no_oracle_model: PolicyValueNet | None = None
    no_oracle_optimizer: torch.optim.Optimizer | None = None
    no_oracle_buffer: RolloutBuffer | None = None
    no_oracle_obs: dict[str, np.ndarray] | None = None
    no_oracle_recurrent_state: torch.Tensor | None = None
    visible_gradient_env: VectorizedTripeptideEnv | None = None
    visible_gradient_model: PolicyValueNet | None = None
    visible_gradient_optimizer: torch.optim.Optimizer | None = None
    visible_gradient_buffer: RolloutBuffer | None = None
    visible_gradient_obs: dict[str, np.ndarray] | None = None
    visible_gradient_recurrent_state: torch.Tensor | None = None
    baseline_lr_tuning_result: dict | None = None

    if config.enable_baselines and config.oracle_mode != "no_oracle":
        no_oracle_oracle = SpatialOracle(
            hidden_dim=hidden_dim, token_dim=token_dim, mode="no_oracle",
            seed=config.seed + 131, token_noise_std=config.token_noise_std,
        )
        no_oracle_env = VectorizedTripeptideEnv(
            lifting_map=lifting_map, energy_surface=energy_surface, s_star_sdp=s_star_sdp,
            oracle=no_oracle_oracle, n_env=config.n_env,
            sensing=config.sensing, max_horizon=config.max_horizon,
            seed=config.seed + 149, s1_step_penalty=config.s1_step_penalty,
            reward_noise_std=config.reward_noise_std, step_size=config.step_size,
            ppo_step_scale=config.ppo_step_scale, success_threshold=config.success_threshold,
            compute_episode_baselines=config.enable_baselines,
            lattice_rl=config.lattice_rl, lattice_granularity=config.lattice_granularity,
            learned_lift_oracle=learned_lift_oracle,
        )
        if config.oracle_mode == "convex_gradient":
            visible_gradient_oracle = SpatialOracle(
                hidden_dim=hidden_dim, token_dim=config.visible_dim, mode="visible_gradient",
                seed=config.seed + 167, token_noise_std=config.token_noise_std,
            )
            visible_gradient_env = VectorizedTripeptideEnv(
                lifting_map=lifting_map, energy_surface=energy_surface, s_star_sdp=s_star_sdp,
                oracle=visible_gradient_oracle, n_env=config.n_env,
                sensing=config.sensing, max_horizon=config.max_horizon,
                seed=config.seed + 173, s1_step_penalty=config.s1_step_penalty,
                reward_noise_std=config.reward_noise_std, step_size=config.step_size,
                ppo_step_scale=config.ppo_step_scale, success_threshold=config.success_threshold,
                compute_episode_baselines=config.enable_baselines,
                lattice_rl=config.lattice_rl, lattice_granularity=config.lattice_granularity,
                learned_lift_oracle=learned_lift_oracle,
            )
        if no_oracle_env is not None:
            _synchronize_envs(env, no_oracle_env)
        if visible_gradient_env is not None:
            _synchronize_envs(env, visible_gradient_env)

    if config.tune_baseline_lrs:
        candidate_lrs = _parse_baseline_lr_candidates(config.baseline_lr_candidates)
        baseline_lr_tuning_result = tune_baseline_learning_rates(
            env=env, candidate_lrs=candidate_lrs,
            num_tasks=config.baseline_lr_tune_tasks, seed=config.seed + 809,
        )
        best_lrs = baseline_lr_tuning_result["best_lrs"]
        env.set_baseline_learning_rates(gd=float(best_lrs["gd"]), adam=float(best_lrs["adam"]))
        if no_oracle_env is not None:
            no_oracle_env.set_baseline_learning_rates(gd=float(best_lrs["gd"]), adam=float(best_lrs["adam"]))
        if visible_gradient_env is not None:
            visible_gradient_env.set_baseline_learning_rates(gd=float(best_lrs["gd"]), adam=float(best_lrs["adam"]))
        for env_index in range(env.n_env):
            env._reset_env(env_index)
        if no_oracle_env is not None:
            for env_index in range(no_oracle_env.n_env):
                no_oracle_env._reset_env(env_index)
        if visible_gradient_env is not None:
            for env_index in range(visible_gradient_env.n_env):
                visible_gradient_env._reset_env(env_index)
        tuning_path = run_dir / "baseline_lr_tuning.json"
        with tuning_path.open("w", encoding="utf-8") as handle:
            json.dump(baseline_lr_tuning_result, handle, indent=2)
        print(f"Baseline lr tuning selected: gd={best_lrs['gd']:.6g}, adam={best_lrs['adam']:.6g}")

    if no_oracle_env is not None:
        _synchronize_envs(env, no_oracle_env)
    if visible_gradient_env is not None:
        _synchronize_envs(env, visible_gradient_env)

    model_architecture = str(config.policy_arch)
    model = PolicyValueNet(
        token_feature_dim=env.token_feature_dim, oracle_token_dim=env.oracle_token_dim,
        action_dim=action_dim, hidden_dim=config.hidden_dim,
        oracle_proj_dim=config.oracle_proj_dim, architecture=model_architecture,
        action_space_type=action_space_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    updates_total = int(np.ceil(config.train_steps / max(1, config.n_env * config.rollout_len)))
    lr_scheduler = _build_lr_scheduler(optimizer, config, total_updates=updates_total)

    no_oracle_lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    visible_gradient_lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    if no_oracle_env is not None:
        no_oracle_model = PolicyValueNet(
            token_feature_dim=no_oracle_env.token_feature_dim, oracle_token_dim=no_oracle_env.oracle_token_dim,
            action_dim=action_dim, hidden_dim=config.hidden_dim,
            oracle_proj_dim=config.oracle_proj_dim, architecture=model_architecture,
            action_space_type=action_space_type,
        ).to(device)
        no_oracle_optimizer = torch.optim.Adam(no_oracle_model.parameters(), lr=config.lr)
        no_oracle_lr_scheduler = _build_lr_scheduler(no_oracle_optimizer, config, total_updates=updates_total)
    if visible_gradient_env is not None:
        visible_gradient_model = PolicyValueNet(
            token_feature_dim=visible_gradient_env.token_feature_dim,
            oracle_token_dim=visible_gradient_env.oracle_token_dim,
            action_dim=action_dim, hidden_dim=config.hidden_dim,
            oracle_proj_dim=config.oracle_proj_dim, architecture=model_architecture,
            action_space_type=action_space_type,
        ).to(device)
        visible_gradient_optimizer = torch.optim.Adam(visible_gradient_model.parameters(), lr=config.lr)
        visible_gradient_lr_scheduler = _build_lr_scheduler(visible_gradient_optimizer, config, total_updates=updates_total)

    hparams = PPOHyperParams(
        gamma=config.gamma, gae_lambda=config.gae_lambda, clip_ratio=config.clip_ratio,
        entropy_coef=config.entropy_coef, value_coef=config.value_coef,
        max_grad_norm=config.max_grad_norm, ppo_epochs=config.ppo_epochs,
        minibatches=config.minibatches,
    )

    buffer = RolloutBuffer(
        rollout_len=config.rollout_len, n_env=config.n_env,
        token_feature_dim=env.token_feature_dim, action_dim=action_dim,
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
            rollout_len=config.rollout_len, n_env=config.n_env,
            token_feature_dim=no_oracle_env.token_feature_dim, action_dim=action_dim,
            action_dtype=action_space_type,
        )
        no_oracle_obs = no_oracle_env.get_obs()
        assert no_oracle_model is not None
        no_oracle_recurrent_state = no_oracle_model.initial_state(batch_size=config.n_env, device=device)
        no_oracle_reward_returns_rms = RunningMeanStd()
        no_oracle_reward_returns = np.zeros(config.n_env, dtype=np.float64)
    if visible_gradient_env is not None:
        visible_gradient_buffer = RolloutBuffer(
            rollout_len=config.rollout_len, n_env=config.n_env,
            token_feature_dim=visible_gradient_env.token_feature_dim, action_dim=action_dim,
            action_dtype=action_space_type,
        )
        visible_gradient_obs = visible_gradient_env.get_obs()
        assert visible_gradient_model is not None
        visible_gradient_recurrent_state = visible_gradient_model.initial_state(
            batch_size=config.n_env, device=device
        )
        visible_gradient_reward_returns_rms = RunningMeanStd()
        visible_gradient_reward_returns = np.zeros(config.n_env, dtype=np.float64)

    curve_metrics: list[dict] = []
    save_metrics_interval = int(config.save_metrics_interval_episodes)
    save_every_episode = save_metrics_interval <= 0
    last_row: dict[str, float | int] | None = None
    metric_csv_path = run_dir / "metrics.csv"
    metric_jsonl_path = run_dir / "metrics.jsonl"
    metric_fields = [
        "episodes", "steps", "success", "success_rate",
        "avg_baseline_success_rate", "avg_adam_baseline_success_rate",
        "avg_no_oracle_success_rate", "avg_visible_gradient_success_rate",
        "final_objective", "avg_final_objective",
        "baseline_final_objective", "avg_baseline_final_objective",
        "adam_baseline_final_objective", "avg_adam_baseline_final_objective",
        "no_oracle_final_objective", "avg_no_oracle_final_objective",
        "visible_gradient_final_objective", "avg_visible_gradient_final_objective",
        "final_ref_distance", "avg_final_ref_distance",
        "avg_path_len", "avg_shortest_dist", "avg_stretch",
        "window_size", "policy_loss", "value_loss", "entropy", "lr",
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
    all_path_lengths: list[float] = []
    all_shortest_dists: list[float] = []
    success_path_lengths: list[float] = []
    failure_path_lengths: list[float] = []

    with metric_csv_path.open("w", newline="", encoding="utf-8") as csv_file, \
         metric_jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=metric_fields)
        writer.writeheader()

        total_steps = 0
        total_episodes = 0
        total_no_oracle_episodes = 0
        total_visible_gradient_episodes = 0
        update_index = 0
        last_update_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

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
                        token_t, dist_t, step_t, hidden_state=recurrent_state, deterministic=False
                    )
                actions = actions_t.cpu().numpy()
                logprobs = logprob_t.cpu().numpy()
                values = values_t.cpu().numpy()
                recurrent_state_np = None
                if recurrent_state is not None:
                    recurrent_state_np = recurrent_state.cpu().numpy()

                next_obs, rewards, dones, infos = env.step(actions)
                normalized_rewards, reward_returns = normalize_rewards_with_running_returns(
                    rewards=rewards, dones=dones, returns_tracker=reward_returns,
                    returns_rms=reward_returns_rms, gamma=config.gamma, clip_abs=reward_norm_clip,
                )
                buffer.add(obs=obs, action=actions, logprob=logprobs, reward=normalized_rewards,
                           done=dones, value=values, hidden_state=recurrent_state_np)

                if next_recurrent_state is not None:
                    recurrent_state = next_recurrent_state.clone()
                    done_mask = torch.as_tensor(dones, device=device, dtype=torch.bool)
                    recurrent_state[done_mask] = 0.0
                obs = next_obs
                step_increment = config.n_env
                total_steps += step_increment
                progress.update(step_increment)

                # No-oracle model step
                if (no_oracle_model is not None and no_oracle_env is not None
                        and no_oracle_buffer is not None and no_oracle_obs is not None):
                    no_token_t, no_dist_t, no_step_t = obs_to_tensors(no_oracle_obs, device)
                    with torch.no_grad():
                        no_actions_t, no_logprob_t, no_values_t, no_next_rs = no_oracle_model.act(
                            no_token_t, no_dist_t, no_step_t,
                            hidden_state=no_oracle_recurrent_state, deterministic=False
                        )
                    no_actions = no_actions_t.cpu().numpy()
                    no_logprobs = no_logprob_t.cpu().numpy()
                    no_values = no_values_t.cpu().numpy()
                    no_rs_np = no_oracle_recurrent_state.cpu().numpy() if no_oracle_recurrent_state is not None else None
                    no_next_obs, no_rewards, no_dones, no_infos = no_oracle_env.step(no_actions)
                    assert no_oracle_reward_returns is not None
                    assert no_oracle_reward_returns_rms is not None
                    no_norm_rewards, no_oracle_reward_returns = normalize_rewards_with_running_returns(
                        rewards=no_rewards, dones=no_dones, returns_tracker=no_oracle_reward_returns,
                        returns_rms=no_oracle_reward_returns_rms, gamma=config.gamma, clip_abs=reward_norm_clip,
                    )
                    no_oracle_buffer.add(obs=no_oracle_obs, action=no_actions, logprob=no_logprobs,
                                         reward=no_norm_rewards, done=no_dones, value=no_values,
                                         hidden_state=no_rs_np)
                    if no_next_rs is not None:
                        no_oracle_recurrent_state = no_next_rs.clone()
                        no_oracle_recurrent_state[torch.as_tensor(no_dones, device=device, dtype=torch.bool)] = 0.0
                    no_oracle_obs = no_next_obs
                    for no_env_idx in np.where(no_dones)[0]:
                        no_info = no_infos[int(no_env_idx)]
                        if not no_info.get("episode_done", False):
                            continue
                        total_no_oracle_episodes += 1
                        no_fo = float(no_info.get("final_objective", float("nan")))
                        if np.isfinite(no_fo):
                            recent_no_oracle_final_objective.append(no_fo)
                            no_succ = 1.0 if no_fo <= env.success_threshold else 0.0
                            recent_no_oracle_success.append(no_succ)

                # Visible-gradient model step
                if (visible_gradient_model is not None and visible_gradient_env is not None
                        and visible_gradient_buffer is not None and visible_gradient_obs is not None):
                    vg_token_t, vg_dist_t, vg_step_t = obs_to_tensors(visible_gradient_obs, device)
                    with torch.no_grad():
                        vg_actions_t, vg_logprob_t, vg_values_t, vg_next_rs = visible_gradient_model.act(
                            vg_token_t, vg_dist_t, vg_step_t,
                            hidden_state=visible_gradient_recurrent_state, deterministic=False
                        )
                    vg_actions = vg_actions_t.cpu().numpy()
                    vg_logprobs = vg_logprob_t.cpu().numpy()
                    vg_values = vg_values_t.cpu().numpy()
                    vg_rs_np = visible_gradient_recurrent_state.cpu().numpy() if visible_gradient_recurrent_state is not None else None
                    vg_next_obs, vg_rewards, vg_dones, vg_infos = visible_gradient_env.step(vg_actions)
                    assert visible_gradient_reward_returns is not None
                    assert visible_gradient_reward_returns_rms is not None
                    vg_norm_rewards, visible_gradient_reward_returns = normalize_rewards_with_running_returns(
                        rewards=vg_rewards, dones=vg_dones, returns_tracker=visible_gradient_reward_returns,
                        returns_rms=visible_gradient_reward_returns_rms, gamma=config.gamma, clip_abs=reward_norm_clip,
                    )
                    visible_gradient_buffer.add(obs=visible_gradient_obs, action=vg_actions, logprob=vg_logprobs,
                                                 reward=vg_norm_rewards, done=vg_dones, value=vg_values,
                                                 hidden_state=vg_rs_np)
                    if vg_next_rs is not None:
                        visible_gradient_recurrent_state = vg_next_rs.clone()
                        visible_gradient_recurrent_state[torch.as_tensor(vg_dones, device=device, dtype=torch.bool)] = 0.0
                    visible_gradient_obs = vg_next_obs
                    for vg_env_idx in np.where(vg_dones)[0]:
                        vg_info = vg_infos[int(vg_env_idx)]
                        if not vg_info.get("episode_done", False):
                            continue
                        total_visible_gradient_episodes += 1
                        vg_fo = float(vg_info.get("final_objective", float("nan")))
                        if np.isfinite(vg_fo):
                            recent_visible_gradient_final_objective.append(vg_fo)
                            vg_succ = 1.0 if vg_fo <= env.success_threshold else 0.0
                            recent_visible_gradient_success.append(vg_succ)

                # Process main env episode completions
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
                    baseline_fo = float(info.get("baseline_final_objective", float("nan")))
                    adam_fo = float(info.get("adam_baseline_final_objective", float("nan")))
                    final_ref_distance = float(info.get("final_ref_distance", float("nan")))

                    recent_success.append(success)
                    if env.compute_episode_baselines:
                        bl_succ = 1.0 if np.isfinite(baseline_fo) and baseline_fo <= env.success_threshold else 0.0
                        adam_succ = 1.0 if np.isfinite(adam_fo) and adam_fo <= env.success_threshold else 0.0
                        recent_baseline_success.append(bl_succ)
                        recent_adam_baseline_success.append(adam_succ)
                    recent_path_len.append(episode_len)
                    recent_shortest_dist.append(shortest_dist)
                    if np.isfinite(final_objective):
                        recent_final_objective.append(final_objective)
                    if np.isfinite(baseline_fo):
                        recent_baseline_final_objective.append(baseline_fo)
                    if np.isfinite(adam_fo):
                        recent_adam_baseline_final_objective.append(adam_fo)
                    if np.isfinite(final_ref_distance):
                        recent_final_ref_distance.append(final_ref_distance)
                    all_path_lengths.append(episode_len)
                    all_shortest_dists.append(shortest_dist)
                    if success >= 0.5:
                        success_path_lengths.append(episode_len)
                    else:
                        failure_path_lengths.append(episode_len)
                    if "stretch" in info:
                        recent_stretch.append(float(info["stretch"]))

                    row = {
                        "episodes": int(total_episodes),
                        "steps": int(total_steps),
                        "success": float(success),
                        "success_rate": float(np.mean(recent_success)),
                        "avg_baseline_success_rate": float(np.mean(recent_baseline_success)) if recent_baseline_success else float("nan"),
                        "avg_adam_baseline_success_rate": float(np.mean(recent_adam_baseline_success)) if recent_adam_baseline_success else float("nan"),
                        "avg_no_oracle_success_rate": float(np.mean(recent_no_oracle_success)) if recent_no_oracle_success else float("nan"),
                        "avg_visible_gradient_success_rate": float(np.mean(recent_visible_gradient_success)) if recent_visible_gradient_success else float("nan"),
                        "final_objective": float(final_objective),
                        "avg_final_objective": float(np.mean(recent_final_objective)) if recent_final_objective else float("nan"),
                        "baseline_final_objective": float(baseline_fo),
                        "avg_baseline_final_objective": float(np.mean(recent_baseline_final_objective)) if recent_baseline_final_objective else float("nan"),
                        "adam_baseline_final_objective": float(adam_fo),
                        "avg_adam_baseline_final_objective": float(np.mean(recent_adam_baseline_final_objective)) if recent_adam_baseline_final_objective else float("nan"),
                        "no_oracle_final_objective": float(recent_no_oracle_final_objective[-1]) if recent_no_oracle_final_objective else float("nan"),
                        "avg_no_oracle_final_objective": float(np.mean(recent_no_oracle_final_objective)) if recent_no_oracle_final_objective else float("nan"),
                        "visible_gradient_final_objective": float(recent_visible_gradient_final_objective[-1]) if recent_visible_gradient_final_objective else float("nan"),
                        "avg_visible_gradient_final_objective": float(np.mean(recent_visible_gradient_final_objective)) if recent_visible_gradient_final_objective else float("nan"),
                        "final_ref_distance": float(final_ref_distance),
                        "avg_final_ref_distance": float(np.mean(recent_final_ref_distance)) if recent_final_ref_distance else float("nan"),
                        "avg_path_len": float(np.mean(recent_path_len)),
                        "avg_shortest_dist": float(np.mean(recent_shortest_dist)),
                        "avg_stretch": float(np.mean(recent_stretch)) if recent_stretch else float("nan"),
                        "window_size": int(len(recent_success)),
                        "policy_loss": float(last_update_stats["policy_loss"]),
                        "value_loss": float(last_update_stats["value_loss"]),
                        "entropy": float(last_update_stats["entropy"]),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                    last_row = row
                    progress.set_postfix(success_rate=f"{float(row['success_rate']):.2%}", refresh=True)

                    should_sample_curve = save_every_episode or (
                        save_metrics_interval > 0 and total_episodes % save_metrics_interval == 0
                    )
                    if should_sample_curve and np.isfinite(float(row["avg_final_objective"])):
                        curve_metrics.append({
                            "episodes": int(total_episodes),
                            "success_rate": float(row["success_rate"]),
                            "avg_final_objective": float(row["avg_final_objective"]),
                        })

                    writer.writerow(row)
                    csv_file.flush()
                    jsonl_file.write(json.dumps(row) + "\n")
                    jsonl_file.flush()

            # PPO update for main model
            token_t, dist_t, step_t = obs_to_tensors(obs, device)
            with torch.no_grad():
                _, last_values_t, _ = model.forward(token_t, dist_t, step_t)
            last_values = last_values_t.cpu().numpy()
            advantages, returns = compute_gae(
                rewards=buffer.rewards, dones=buffer.dones, values=buffer.values,
                last_values=last_values, gamma=config.gamma, gae_lambda=config.gae_lambda,
            )
            last_update_stats = ppo_update(
                model=model, optimizer=optimizer, buffer=buffer,
                advantages=advantages, returns=returns, hparams=hparams, device=device,
            )
            if lr_scheduler is not None:
                lr_scheduler.step()

            # PPO update for no-oracle model
            if (no_oracle_model is not None and no_oracle_env is not None
                    and no_oracle_buffer is not None and no_oracle_obs is not None
                    and no_oracle_optimizer is not None):
                no_token_t, no_dist_t, no_step_t = obs_to_tensors(no_oracle_obs, device)
                with torch.no_grad():
                    _, no_last_values_t, _ = no_oracle_model.forward(no_token_t, no_dist_t, no_step_t)
                no_last_values = no_last_values_t.cpu().numpy()
                no_advantages, no_returns = compute_gae(
                    rewards=no_oracle_buffer.rewards, dones=no_oracle_buffer.dones,
                    values=no_oracle_buffer.values, last_values=no_last_values,
                    gamma=config.gamma, gae_lambda=config.gae_lambda,
                )
                ppo_update(model=no_oracle_model, optimizer=no_oracle_optimizer,
                           buffer=no_oracle_buffer, advantages=no_advantages, returns=no_returns,
                           hparams=hparams, device=device)
                if no_oracle_lr_scheduler is not None:
                    no_oracle_lr_scheduler.step()

            # PPO update for visible-gradient model
            if (visible_gradient_model is not None and visible_gradient_env is not None
                    and visible_gradient_buffer is not None and visible_gradient_obs is not None
                    and visible_gradient_optimizer is not None):
                vg_token_t, vg_dist_t, vg_step_t = obs_to_tensors(visible_gradient_obs, device)
                with torch.no_grad():
                    _, vg_last_values_t, _ = visible_gradient_model.forward(vg_token_t, vg_dist_t, vg_step_t)
                vg_last_values = vg_last_values_t.cpu().numpy()
                vg_advantages, vg_returns = compute_gae(
                    rewards=visible_gradient_buffer.rewards, dones=visible_gradient_buffer.dones,
                    values=visible_gradient_buffer.values, last_values=vg_last_values,
                    gamma=config.gamma, gae_lambda=config.gae_lambda,
                )
                ppo_update(model=visible_gradient_model, optimizer=visible_gradient_optimizer,
                           buffer=visible_gradient_buffer, advantages=vg_advantages, returns=vg_returns,
                           hparams=hparams, device=device)
                if visible_gradient_lr_scheduler is not None:
                    visible_gradient_lr_scheduler.step()
            update_index += 1

        progress.close()

    if config.enable_training_plots:
        plot_path_length_histograms(
            path_lengths=all_path_lengths,
            success_path_lengths=success_path_lengths,
            failure_path_lengths=failure_path_lengths,
            output_dir=run_dir,
            title_prefix=f"Tripeptide K_map={config.K_map}, {config.oracle_mode}, {config.sensing}",
        )
        maybe_save_trajectory_plot(
            model=model, env=env, device=device,
            no_oracle_model=no_oracle_model,
            visible_gradient_model=visible_gradient_model,
            visible_gradient_env=visible_gradient_env,
            output_path=run_dir / "trajectory_with_gradients.png",
            title=f"Tripeptide | K_map={config.K_map}, {config.oracle_mode}",
        )

    optimization_eval = evaluate_optimization_curves(
        config=config, run_dir=run_dir, device=device,
        hidden_gradient_model=model, hidden_gradient_env=env,
        no_oracle_model=no_oracle_model, no_oracle_env=no_oracle_env,
        visible_gradient_model=visible_gradient_model, visible_gradient_env=visible_gradient_env,
    )

    summary = {
        "run_dir": str(run_dir),
        "task": "tripeptide",
        "K_map": int(config.K_map),
        "K_relax": int(config.K_relax),
        "hidden_dim": int(hidden_dim),
        "final_objective_value": float(last_row["avg_final_objective"]) if last_row is not None else None,
        "final_baseline_objective_value": float(last_row["avg_baseline_final_objective"]) if last_row is not None else None,
        "final_adam_baseline_objective_value": float(last_row["avg_adam_baseline_final_objective"]) if last_row is not None else None,
        "updates": update_index,
        "steps": total_steps,
        "episodes": total_episodes,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    output = {
        "summary": summary,
        "metrics": curve_metrics,
        "config": asdict(config),
        "optimization_curves": optimization_eval,
        "baseline_lr_tuning": baseline_lr_tuning_result,
    }
    if return_artifacts:
        output["model"] = model
        output["env"] = env
        output["no_oracle_model"] = no_oracle_model
        output["no_oracle_env"] = no_oracle_env
        output["visible_gradient_model"] = visible_gradient_model
        output["visible_gradient_env"] = visible_gradient_env
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_seed_values(seed_arg: str) -> list[int]:
    raw = str(seed_arg).strip()
    if not raw:
        raise ValueError("seed must be non-empty")
    seeds: list[int] = []
    for piece in raw.split(","):
        token = piece.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("seed must include at least one integer")
    return seeds


def parse_args() -> tuple[TrainConfig, list[int]]:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Tripeptide RL training")
    parser.add_argument("--K_map", type=int, default=defaults.K_map)
    parser.add_argument("--K_relax", type=int, default=defaults.K_relax)
    parser.add_argument("--energy_json", type=str, default=defaults.energy_json)
    parser.add_argument("--use_synthetic_fallback", action="store_true")
    parser.add_argument("--token_noise_std", type=float, default=defaults.token_noise_std)
    parser.add_argument("--ppo_step_scale", type=float, default=defaults.ppo_step_scale)
    parser.add_argument("--step_size", type=float, default=defaults.step_size)
    parser.add_argument("--success_threshold", type=float, default=defaults.success_threshold)
    parser.add_argument("--policy_arch", type=str, choices=["mlp", "gru"], default=defaults.policy_arch)
    parser.add_argument("--disable_baselines", action="store_true")
    parser.add_argument("--disable_baseline_lr_tuning", action="store_true")
    parser.add_argument("--baseline_lr_candidates", type=str, default=defaults.baseline_lr_candidates)
    parser.add_argument("--baseline_lr_tune_tasks", type=int, default=defaults.baseline_lr_tune_tasks)
    parser.add_argument("--optimization_curve_tasks", type=int, default=defaults.optimization_curve_tasks)
    parser.add_argument("--disable_optimization_curve_eval", action="store_true")
    parser.add_argument("--n_env", type=int, default=defaults.n_env)
    parser.add_argument("--train_steps", type=int, default=defaults.train_steps)
    parser.add_argument("--rollout_len", type=int, default=defaults.rollout_len)
    parser.add_argument("--algo", type=str, default=defaults.algo)
    parser.add_argument("--sensing", type=str, choices=["S0", "S1"], default=defaults.sensing)
    parser.add_argument("--reward_noise_std", type=float, default=defaults.reward_noise_std)
    parser.add_argument("--oracle_mode", type=str,
                        choices=sorted(ORACLE_MODES - {"visible_gradient"}),
                        default=defaults.oracle_mode)
    parser.add_argument("--running_avg_window", type=int, default=defaults.running_avg_window)
    parser.add_argument("--save_metrics_interval_episodes", type=int, default=defaults.save_metrics_interval_episodes)
    parser.add_argument("--eval_interval_episodes", type=int, default=defaults.eval_interval_episodes)
    parser.add_argument("--eval_episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--max_horizon", type=int, default=defaults.max_horizon)
    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument("--run_name", type=str, default=defaults.run_name)
    parser.add_argument("--seed", type=str, default=str(defaults.seed))
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--lr_scheduler", type=str, choices=["none", "constant", "linear", "cosine"],
                        default=defaults.lr_scheduler)
    parser.add_argument("--lr_min_factor", type=float, default=defaults.lr_min_factor)
    parser.add_argument("--lr_warmup_updates", type=int, default=defaults.lr_warmup_updates)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae_lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--clip_ratio", type=float, default=defaults.clip_ratio)
    parser.add_argument("--entropy_coef", type=float, default=defaults.entropy_coef)
    parser.add_argument("--value_coef", type=float, default=defaults.value_coef)
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--ppo_epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)
    parser.add_argument("--hidden_dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--oracle_proj_dim", type=int, default=defaults.oracle_proj_dim)
    parser.add_argument("--s1_step_penalty", type=float, default=defaults.s1_step_penalty)
    parser.add_argument("--disable_training_plots", action="store_true")
    parser.add_argument("--simple_s_star", action="store_true",
                        help="Use norm-ball oracle s*=-R*c/||c|| instead of SDP relaxation.")
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
    parser.add_argument(
        "--learned_lift",
        action="store_true",
        help="Use a learned encoder (MLP) + decoder (ICNN) instead of Fourier lifting map + SDP.",
    )
    parser.add_argument("--learned_lift_D", type=int, default=defaults.learned_lift_D,
                        help="Hidden dimension D for the learned lifting map.")
    parser.add_argument("--learned_lift_encoder_hidden", type=str,
                        default=defaults.learned_lift_encoder_hidden,
                        help="Comma-separated encoder hidden layer sizes.")
    parser.add_argument("--learned_lift_decoder_hidden", type=str,
                        default=defaults.learned_lift_decoder_hidden,
                        help="Comma-separated ICNN decoder hidden layer sizes.")
    parser.add_argument("--learned_lift_n_train", type=int, default=defaults.learned_lift_n_train)
    parser.add_argument("--learned_lift_n_epochs", type=int, default=defaults.learned_lift_n_epochs)
    parser.add_argument("--learned_lift_lr", type=float, default=defaults.learned_lift_lr)

    args = parser.parse_args()
    seed_values = _parse_seed_values(args.seed)
    config = TrainConfig(
        K_map=args.K_map,
        K_relax=args.K_relax,
        energy_json=args.energy_json,
        use_synthetic_fallback=args.use_synthetic_fallback,
        token_noise_std=args.token_noise_std,
        ppo_step_scale=args.ppo_step_scale,
        step_size=args.step_size,
        success_threshold=args.success_threshold,
        policy_arch=args.policy_arch,
        enable_baselines=not args.disable_baselines,
        tune_baseline_lrs=not args.disable_baseline_lr_tuning,
        baseline_lr_candidates=args.baseline_lr_candidates,
        baseline_lr_tune_tasks=args.baseline_lr_tune_tasks,
        optimization_curve_tasks=args.optimization_curve_tasks,
        enable_optimization_curve_eval=not args.disable_optimization_curve_eval,
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
        s1_step_penalty=args.s1_step_penalty,
        enable_training_plots=not args.disable_training_plots,
        use_simple_s_star=args.simple_s_star,
        lattice_rl=args.lattice_RL,
        lattice_granularity=args.lattice_granularity,
        learned_lift=args.learned_lift,
        learned_lift_D=args.learned_lift_D,
        learned_lift_encoder_hidden=args.learned_lift_encoder_hidden,
        learned_lift_decoder_hidden=args.learned_lift_decoder_hidden,
        learned_lift_n_train=args.learned_lift_n_train,
        learned_lift_n_epochs=args.learned_lift_n_epochs,
        learned_lift_lr=args.learned_lift_lr,
    )
    return config, seed_values


def main() -> None:
    config, seed_values = parse_args()
    if len(seed_values) == 1:
        output = run_training(config)
        print(json.dumps(output["summary"], indent=2))
        return

    summaries: list[dict] = []
    base_run_name = config.run_name.strip()
    for seed in seed_values:
        run_name = f"{base_run_name}_seed{seed}" if base_run_name else ""
        seed_config = replace(config, seed=int(seed), run_name=run_name)
        output = run_training(seed_config)
        summaries.append(output["summary"])

    aggregate = {
        "num_seeds": len(seed_values),
        "seeds": [int(s) for s in seed_values],
        "runs": summaries,
    }
    summary_stem = base_run_name or f"tripeptide_{config.oracle_mode}_{config.sensing}_multi_seed"
    aggregate_path = Path(config.logdir) / f"{summary_stem}_summary.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(json.dumps({
        "multi_seed_summary_path": str(aggregate_path),
        "num_seeds": len(seed_values),
        "seeds": [int(s) for s in seed_values],
    }, indent=2))


if __name__ == "__main__":
    main()
