import argparse
import csv
import json
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from v2.config import TrainConfig
from v2.distances import INF_DISTANCE, precompute_distance_pool
from v2.env import VectorizedGraphEnv
from v2.graph import build_directed_regular_graph
from v2.model import PolicyValueNet
from v2.oracle import ORACLE_MODES, SPATIAL_ORACLE_MODES, Oracle, SpatialOracle
from v2.plotting import plot_path_length_histograms, plot_spatial_trajectory_with_gradients
from v2.ppo import PPOHyperParams, RolloutBuffer, compute_gae, ppo_update
from v2.spatial_env import VectorizedSpatialEnv


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
    env: VectorizedGraphEnv | VectorizedSpatialEnv,
    eval_episodes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    success_count = 0
    path_lengths: list[int] = []
    stretches: list[float] = []

    for _ in range(eval_episodes):
        if isinstance(env, VectorizedGraphEnv):
            spec = env.sample_episode_spec()
            node = spec.source
            steps_taken = 0
            success = False
            hidden_state = model.initial_state(batch_size=1, device=device)

            for step in range(spec.horizon):
                if env.oracle.mode == "no_oracle":
                    token_id = env.null_token_id
                elif step < spec.message_tokens.shape[0]:
                    token_id = int(spec.message_tokens[step])
                else:
                    token_id = env.null_token_id

                token_features = np.zeros((1, env.token_feature_dim), dtype=np.float32)
                token_features[0, token_id] = 1.0

                dist_value = int(env.dist_pool[spec.target_index, node])
                if dist_value >= INF_DISTANCE:
                    dist_feature = 1.0
                else:
                    dist_feature = float(min(dist_value, env.n) / env.n)

                step_fraction = float(step / max(1, spec.horizon))
                token_t = torch.tensor(token_features, dtype=torch.float32, device=device)
                dist_t = torch.tensor([dist_feature], dtype=torch.float32, device=device)
                step_t = torch.tensor([step_fraction], dtype=torch.float32, device=device)

                logits, _, hidden_state = model.forward(
                    token_t,
                    dist_t,
                    step_t,
                    hidden_state=hidden_state,
                )
                action = int(torch.argmax(logits, dim=-1).item())

                node = int(env.out_neighbors[node, action])
                steps_taken += 1
                if node == spec.target_node:
                    success = True
                    break
        else:
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
                objective_value = env._objective_value(state, env_index=0)
                dist_feature = float(np.clip(objective_value / env.max_objective, 0.0, 1.0))
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
                if env._objective_value(state, env_index=0) <= env.success_threshold:
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
) -> dict[str, np.ndarray] | None:
    model.eval()
    if no_oracle_model is not None:
        no_oracle_model.eval()

    spec = env.sample_episode_spec(
        env_index=0,
        refresh_map=env.refresh_map_each_episode,
    )
    state = spec.source.copy()
    hidden_state = model.initial_state(batch_size=1, device=device)

    trajectory = [state.astype(np.float32)]
    gradient_xy: list[np.ndarray] = []
    move_vectors: list[np.ndarray] = []

    for step in range(spec.horizon):
        token_features = env._obs_token_features(state, env_index=0)[None, :]
        z_value = env._objective_value(state, env_index=0)
        z_feature = float(np.clip(z_value / env.max_objective, 0.0, 1.0))
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

        if env._objective_value(state, env_index=0) <= env.success_threshold:
            break

    no_oracle_trajectory_xy: np.ndarray | None = None
    if no_oracle_model is not None:
        no_state = spec.source.copy()
        no_hidden_state = no_oracle_model.initial_state(batch_size=1, device=device)
        no_trajectory = [no_state.astype(np.float32)]
        for step in range(spec.horizon):
            token_features = env._obs_token_features(no_state, env_index=0)[None, :]
            token_features[:, : env.oracle_token_dim] = 0.0
            z_value = env._objective_value(no_state, env_index=0)
            z_feature = float(np.clip(z_value / env.max_objective, 0.0, 1.0))
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

            if env._objective_value(no_state, env_index=0) <= env.success_threshold:
                break
        no_oracle_trajectory_xy = np.asarray(no_trajectory, dtype=np.float32)

    model.train()
    if no_oracle_model is not None:
        no_oracle_model.train()
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
        ),
        "sgd_baseline_trajectory_xy": rollout_spatial_gradient_descent_baseline(
            env=env,
            start_xy=spec.source,
            horizon=spec.horizon,
            gradient_noise_std=env.sgd_gradient_noise_std,
            noise_rng=env.baseline_rng,
        ),
    }
    if no_oracle_trajectory_xy is not None and no_oracle_trajectory_xy.shape[0] >= 2:
        result["no_oracle_trajectory_xy"] = no_oracle_trajectory_xy
    return result


def rollout_spatial_gradient_descent_baseline(
    env: VectorizedSpatialEnv,
    start_xy: np.ndarray,
    horizon: int,
    gradient_noise_std: float = 0.0,
    noise_rng: np.random.Generator | None = None,
) -> np.ndarray:
    # Baseline: local descent in z-space using the true 2D gradient of E(F(z)).
    # SGD variant is the same update with additive Gaussian noise on the 2D gradient.
    state = start_xy.astype(np.float32).copy()
    trajectory = [state.astype(np.float32)]
    noise_std = float(max(0.0, gradient_noise_std))
    rng = noise_rng if noise_rng is not None else env.baseline_rng

    for step in range(int(horizon)):
        grad_xy = env._gradient_xy(state, env_index=0)
        if noise_std > 0.0:
            grad_xy = grad_xy + rng.normal(0.0, noise_std, size=grad_xy.shape).astype(np.float32)
        step_direction = -grad_xy
        step_scale = env._cosine_annealed_step_scale(step, int(horizon))
        next_state = env._apply_action(state, step_direction, step_scale_override=step_scale)
        trajectory.append(next_state.astype(np.float32))
        state = next_state

        if env._objective_value(state, env_index=0) <= env.success_threshold:
            break

    return np.asarray(trajectory, dtype=np.float32)


def maybe_save_spatial_trajectory_plot(
    model: PolicyValueNet,
    env: VectorizedSpatialEnv,
    device: torch.device,
    output_path: Path,
    title: str,
    no_oracle_model: PolicyValueNet | None = None,
) -> None:
    spatial_trace = collect_spatial_trajectory(model, env, device, no_oracle_model=no_oracle_model)
    if spatial_trace is None:
        return

    grid_x, grid_y, grid_energy = env.energy_landscape_grid(resolution=150, env_index=0)
    plot_spatial_trajectory_with_gradients(
        trajectory_xy=spatial_trace["trajectory_xy"],
        gradient_xy=spatial_trace["gradient_xy"],
        move_vectors_xy=spatial_trace["move_vectors_xy"],
        target_xy=spatial_trace["target_xy"],
        baseline_trajectory_xy=spatial_trace["baseline_trajectory_xy"],
        sgd_baseline_trajectory_xy=spatial_trace["sgd_baseline_trajectory_xy"],
        no_oracle_trajectory_xy=spatial_trace.get("no_oracle_trajectory_xy"),
        output_path=output_path,
        title=title,
        landscape_x=grid_x,
        landscape_y=grid_y,
        landscape_energy=grid_energy,
    )


@torch.no_grad()
def estimate_cipher_accuracy(
    model: PolicyValueNet,
    env: VectorizedGraphEnv,
    device: torch.device,
    samples_per_token: int = 400,
) -> float | None:
    true_inverse = env.oracle.true_inverse_mapping
    if true_inverse is None:
        return None

    model.eval()
    inferred = np.zeros(env.oracle.sigma_size, dtype=np.int64)

    for token in range(env.oracle.sigma_size):
        action_counts = np.zeros(env.d, dtype=np.int64)
        for _ in range(samples_per_token):
            target_index = int(env.rng.integers(0, env.t_pool_size))
            dist_row = env.dist_pool[target_index]
            reachable = np.where((dist_row > 0) & (dist_row < INF_DISTANCE))[0]
            if reachable.size == 0:
                continue

            node = int(env.rng.choice(reachable))
            dist_value = int(dist_row[node])
            horizon = max(1, min(2 * max(1, dist_value), env.max_horizon))
            k = int(env.rng.integers(0, horizon))

            dist_feature = float(min(dist_value, env.n) / env.n)
            step_fraction = float(k / max(1, horizon))

            token_features = np.zeros((1, env.token_feature_dim), dtype=np.float32)
            token_features[0, token] = 1.0
            token_t = torch.tensor(token_features, dtype=torch.float32, device=device)
            dist_t = torch.tensor([dist_feature], dtype=torch.float32, device=device)
            step_t = torch.tensor([step_fraction], dtype=torch.float32, device=device)

            logits, _, _ = model.forward(token_t, dist_t, step_t)
            action = int(torch.argmax(logits, dim=-1).item())
            action_counts[action] += 1

        inferred[token] = int(np.argmax(action_counts))

    model.train()
    return float(np.mean(inferred == true_inverse))


def find_episodes_to_threshold(metrics: list[dict], threshold: float = 0.8) -> int | None:
    for item in metrics:
        if item["success_rate"] >= threshold:
            return int(item["episodes"])
    return None


def run_training(config: TrainConfig) -> dict:
    if config.algo != "ppo":
        raise ValueError("Only --algo ppo is implemented in this prototype")
    if config.device != "cpu":
        raise ValueError("This prototype is CPU-only; use --device cpu")
    if config.running_avg_window < 1:
        raise ValueError("running_avg_window must be >= 1")
    if config.task == "graph" and config.oracle_mode not in ORACLE_MODES:
        raise ValueError(
            f"oracle_mode={config.oracle_mode!r} is invalid for task='graph'; "
            f"choose one of {sorted(ORACLE_MODES)}"
        )
    if config.task == "spatial" and config.oracle_mode not in SPATIAL_ORACLE_MODES:
        raise ValueError(
            f"oracle_mode={config.oracle_mode!r} is invalid for task='spatial'; "
            f"choose one of {sorted(SPATIAL_ORACLE_MODES)}"
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

    set_global_seed(config.seed)
    device = torch.device("cpu")

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

    if config.task == "graph":
        graph = build_directed_regular_graph(n=config.n, d=config.d, seed=config.seed)
        target_nodes, dist_pool = precompute_distance_pool(
            graph.rev_neighbors,
            n=config.n,
            t_pool_size=config.t_pool,
            seed=config.seed + 17,
        )
        valid_pair_mask = (dist_pool > 0) & (dist_pool < INF_DISTANCE)
        valid_st_pairs = int(np.sum(valid_pair_mask, dtype=np.int64))
        print(
            "Valid (s,t) pairs in sampled target pool: "
            f"{valid_st_pairs} (targets={target_nodes.shape[0]}, n={config.n})"
        )

        oracle = Oracle(
            d=config.d,
            mode=config.oracle_mode,
            sigma_size=config.sigma_size,
            seed=config.seed + 29,
            lie_prob=config.lie_prob,
            fst_k=config.fst_k,
        )
        env: VectorizedGraphEnv | VectorizedSpatialEnv = VectorizedGraphEnv(
            out_neighbors=graph.out_neighbors,
            target_nodes=target_nodes,
            dist_pool=dist_pool,
            oracle=oracle,
            n_env=config.n_env,
            sensing=config.sensing,
            max_horizon=config.max_horizon,
            seed=config.seed + 41,
            s1_step_penalty=config.s1_step_penalty,
            reward_noise_std=config.reward_noise_std,
        )
        action_dim = config.d
        action_space_type = "discrete"
    elif config.task == "spatial":
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
            sgd_gradient_noise_std=config.spatial_sgd_gradient_noise_std,
            success_threshold=config.spatial_success_threshold,
            basis_complexity=config.spatial_basis_complexity,
            f_type=config.spatial_f_type,
            refresh_map_each_episode=config.spatial_refresh_map_each_episode,
        )
        action_dim = env.action_dim
        action_space_type = "continuous"

        if config.oracle_mode != "no_oracle":
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
                sgd_gradient_noise_std=config.spatial_sgd_gradient_noise_std,
                success_threshold=config.spatial_success_threshold,
                basis_complexity=config.spatial_basis_complexity,
                f_type=config.spatial_f_type,
                refresh_map_each_episode=config.spatial_refresh_map_each_episode,
            )
    else:
        raise ValueError("task must be 'graph' or 'spatial'")

    if config.task == "spatial":
        model_architecture = str(config.spatial_policy_arch)
    else:
        model_architecture = "gru" if config.oracle_mode == "fst_cipher" else "mlp"
    model = PolicyValueNet(
        token_feature_dim=env.token_feature_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        architecture=model_architecture,
        action_space_type=action_space_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    updates_total = int(np.ceil(config.train_steps / max(1, config.n_env * config.rollout_len)))
    lr_scheduler = _build_lr_scheduler(optimizer, config, total_updates=updates_total)

    no_oracle_lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    if no_oracle_env is not None:
        no_oracle_model = PolicyValueNet(
            token_feature_dim=no_oracle_env.token_feature_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            architecture=model_architecture,
            action_space_type=action_space_type,
        ).to(device)
        no_oracle_optimizer = torch.optim.Adam(no_oracle_model.parameters(), lr=config.lr)
        no_oracle_lr_scheduler = _build_lr_scheduler(
            no_oracle_optimizer,
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

    curve_metrics: list[dict] = []
    save_metrics_interval = int(config.save_metrics_interval_episodes)
    save_every_episode = save_metrics_interval <= 0
    episodes_to_80: int | None = None
    last_row: dict[str, float | int] | None = None
    metric_csv_path = run_dir / "metrics.csv"
    metric_jsonl_path = run_dir / "metrics.jsonl"
    metric_fields = [
        "episodes",
        "steps",
        "success",
        "success_rate",
        "avg_baseline_success_rate",
        "avg_sgd_baseline_success_rate",
        "avg_no_oracle_success_rate",
        "final_objective",
        "avg_final_objective",
        "baseline_final_objective",
        "avg_baseline_final_objective",
        "sgd_baseline_final_objective",
        "avg_sgd_baseline_final_objective",
        "no_oracle_final_objective",
        "avg_no_oracle_final_objective",
        "final_ref_distance",
        "avg_final_ref_distance",
        "baseline_final_ref_distance",
        "avg_baseline_final_ref_distance",
        "sgd_baseline_final_ref_distance",
        "avg_sgd_baseline_final_ref_distance",
        "no_oracle_final_ref_distance",
        "avg_no_oracle_final_ref_distance",
        "avg_path_len",
        "avg_shortest_dist",
        "avg_stretch",
        "window_size",
        "policy_loss",
        "value_loss",
        "entropy",
        "lr",
        "no_oracle_lr",
    ]

    recent_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_baseline_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_sgd_baseline_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_no_oracle_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_path_len: deque[float] = deque(maxlen=config.running_avg_window)
    recent_shortest_dist: deque[float] = deque(maxlen=config.running_avg_window)
    recent_stretch: deque[float] = deque(maxlen=config.running_avg_window)
    recent_final_objective: deque[float] = deque(maxlen=config.running_avg_window)
    recent_baseline_final_objective: deque[float] = deque(maxlen=config.running_avg_window)
    recent_sgd_baseline_final_objective: deque[float] = deque(maxlen=config.running_avg_window)
    recent_no_oracle_final_objective: deque[float] = deque(maxlen=config.running_avg_window)
    recent_final_ref_distance: deque[float] = deque(maxlen=config.running_avg_window)
    recent_baseline_final_ref_distance: deque[float] = deque(maxlen=config.running_avg_window)
    recent_sgd_baseline_final_ref_distance: deque[float] = deque(maxlen=config.running_avg_window)
    recent_no_oracle_final_ref_distance: deque[float] = deque(maxlen=config.running_avg_window)
    all_path_lengths: list[float] = []
    all_shortest_dists: list[float] = []
    all_final_ref_distances: list[float] = []
    all_baseline_final_ref_distances: list[float] = []
    all_sgd_baseline_final_ref_distances: list[float] = []
    all_no_oracle_final_ref_distances: list[float] = []
    success_path_lengths: list[float] = []
    failure_path_lengths: list[float] = []

    with metric_csv_path.open("w", newline="", encoding="utf-8") as csv_file, metric_jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=metric_fields)
        writer.writeheader()

        total_steps = 0
        total_episodes = 0
        total_no_oracle_episodes = 0
        update_index = 0
        last_update_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        progress = tqdm(total=config.train_steps, desc="Training", leave=False)

        while total_steps < config.train_steps:
            buffer.reset()
            if no_oracle_buffer is not None:
                no_oracle_buffer.reset()

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
                        if (
                            config.task == "spatial"
                            and no_oracle_env is not None
                            and isinstance(no_oracle_env, VectorizedSpatialEnv)
                        ):
                            no_success_threshold = no_oracle_env.success_threshold
                            no_oracle_success_val = (
                                1.0
                                if np.isfinite(no_final_objective)
                                and no_final_objective <= no_success_threshold
                                else 0.0
                            )
                            recent_no_oracle_success.append(no_oracle_success_val)
                        if np.isfinite(no_final_ref_distance):
                            recent_no_oracle_final_ref_distance.append(no_final_ref_distance)
                            all_no_oracle_final_ref_distances.append(no_final_ref_distance)

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
                    sgd_baseline_final_objective = float(
                        info.get("sgd_baseline_final_objective", float("nan"))
                    )
                    final_ref_distance = float(info.get("final_ref_distance", float("nan")))
                    baseline_final_ref_distance = float(
                        info.get("baseline_final_ref_distance", float("nan"))
                    )
                    sgd_baseline_final_ref_distance = float(
                        info.get("sgd_baseline_final_ref_distance", float("nan"))
                    )
                    recent_success.append(success)
                    if config.task == "spatial" and isinstance(env, VectorizedSpatialEnv):
                        success_threshold = env.success_threshold
                        baseline_success = (
                            1.0
                            if np.isfinite(baseline_final_objective)
                            and baseline_final_objective <= success_threshold
                            else 0.0
                        )
                        sgd_baseline_success_val = (
                            1.0
                            if np.isfinite(sgd_baseline_final_objective)
                            and sgd_baseline_final_objective <= success_threshold
                            else 0.0
                        )
                        recent_baseline_success.append(baseline_success)
                        recent_sgd_baseline_success.append(sgd_baseline_success_val)
                    recent_path_len.append(episode_len)
                    recent_shortest_dist.append(shortest_dist)
                    if np.isfinite(final_objective):
                        recent_final_objective.append(final_objective)
                    if np.isfinite(baseline_final_objective):
                        recent_baseline_final_objective.append(baseline_final_objective)
                    if np.isfinite(sgd_baseline_final_objective):
                        recent_sgd_baseline_final_objective.append(sgd_baseline_final_objective)
                    if np.isfinite(final_ref_distance):
                        recent_final_ref_distance.append(final_ref_distance)
                        all_final_ref_distances.append(final_ref_distance)
                    if np.isfinite(baseline_final_ref_distance):
                        recent_baseline_final_ref_distance.append(baseline_final_ref_distance)
                        all_baseline_final_ref_distances.append(baseline_final_ref_distance)
                    if np.isfinite(sgd_baseline_final_ref_distance):
                        recent_sgd_baseline_final_ref_distance.append(sgd_baseline_final_ref_distance)
                        all_sgd_baseline_final_ref_distances.append(sgd_baseline_final_ref_distance)
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
                    row = {
                        "episodes": int(total_episodes),
                        "steps": int(total_steps),
                        "success": float(success),
                        "success_rate": float(np.mean(recent_success)),
                        "avg_baseline_success_rate": (
                            float(np.mean(recent_baseline_success))
                            if len(recent_baseline_success) > 0
                            else float("nan")
                        ),
                        "avg_sgd_baseline_success_rate": (
                            float(np.mean(recent_sgd_baseline_success))
                            if len(recent_sgd_baseline_success) > 0
                            else float("nan")
                        ),
                        "avg_no_oracle_success_rate": (
                            float(np.mean(recent_no_oracle_success))
                            if len(recent_no_oracle_success) > 0
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
                        "sgd_baseline_final_objective": float(sgd_baseline_final_objective),
                        "avg_sgd_baseline_final_objective": (
                            float(np.mean(recent_sgd_baseline_final_objective))
                            if len(recent_sgd_baseline_final_objective) > 0
                            else float("nan")
                        ),
                        "no_oracle_final_objective": float(no_oracle_final_objective),
                        "avg_no_oracle_final_objective": (
                            float(np.mean(recent_no_oracle_final_objective))
                            if len(recent_no_oracle_final_objective) > 0
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
                        "sgd_baseline_final_ref_distance": float(sgd_baseline_final_ref_distance),
                        "avg_sgd_baseline_final_ref_distance": (
                            float(np.mean(recent_sgd_baseline_final_ref_distance))
                            if len(recent_sgd_baseline_final_ref_distance) > 0
                            else float("nan")
                        ),
                        "no_oracle_final_ref_distance": float(no_oracle_final_ref_distance),
                        "avg_no_oracle_final_ref_distance": (
                            float(np.mean(recent_no_oracle_final_ref_distance))
                            if len(recent_no_oracle_final_ref_distance) > 0
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
                    }

                    last_row = row
                    progress.set_postfix(
                        success_rate=f"{float(row['success_rate']):.2%}",
                        refresh=True,
                    )
                    if (
                        config.task == "graph"
                        and episodes_to_80 is None
                        and float(row["success_rate"]) >= 0.8
                    ):
                        episodes_to_80 = int(total_episodes)

                    should_sample_curve = save_every_episode or (
                        save_metrics_interval > 0 and total_episodes % save_metrics_interval == 0
                    )
                    if should_sample_curve:
                        if config.task == "graph":
                            curve_metrics.append(
                                {
                                    "episodes": int(total_episodes),
                                    "success_rate": float(row["success_rate"]),
                                }
                            )
                        elif np.isfinite(float(row["avg_final_objective"])):
                            curve_metrics.append(
                                {
                                    "episodes": int(total_episodes),
                                    "success_rate": float(row["success_rate"]),
                                    "baseline_success_rate": float(
                                        row.get("avg_baseline_success_rate", float("nan"))
                                    ),
                                    "sgd_baseline_success_rate": float(
                                        row.get("avg_sgd_baseline_success_rate", float("nan"))
                                    ),
                                    "no_oracle_success_rate": float(
                                        row.get("avg_no_oracle_success_rate", float("nan"))
                                    ),
                                    "objective_value": float(row["avg_final_objective"]),
                                    "baseline_objective_value": float(
                                        row["avg_baseline_final_objective"]
                                    ),
                                    "sgd_baseline_objective_value": float(
                                        row["avg_sgd_baseline_final_objective"]
                                    ),
                                    "no_oracle_objective_value": float(
                                        row["avg_no_oracle_final_objective"]
                                    ),
                                    "distance_value": float(row["avg_final_ref_distance"]),
                                    "baseline_distance_value": float(
                                        row["avg_baseline_final_ref_distance"]
                                    ),
                                    "sgd_baseline_distance_value": float(
                                        row["avg_sgd_baseline_final_ref_distance"]
                                    ),
                                    "no_oracle_distance_value": float(
                                        row["avg_no_oracle_final_ref_distance"]
                                    ),
                                }
                            )
                    writer.writerow(row)
                    jsonl_file.write(json.dumps(row) + "\n")

                    if config.eval_interval_episodes > 0 and total_episodes % config.eval_interval_episodes == 0:
                        avg_path_len_all = float(np.mean(all_path_lengths)) if all_path_lengths else float("nan")
                        avg_shortest_dist_all = (
                            float(np.mean(all_shortest_dists)) if all_shortest_dists else float("nan")
                        )
                        if config.task == "graph":
                            tqdm.write(
                                "episodes="
                                f"{total_episodes} steps={total_steps} "
                                f"success_rate={float(row['success_rate']):.4f} "
                                f"avg_path_len={avg_path_len_all:.3f} "
                                f"avg_shortest_dist={avg_shortest_dist_all:.3f}"
                            )
                        else:
                            tqdm.write(
                                "episodes="
                                f"{total_episodes} steps={total_steps} "
                                f"success_rate={float(row['success_rate']):.4f} "
                                f"success_rate_gd={float(row.get('avg_baseline_success_rate', float('nan'))):.4f} "
                                f"success_rate_sgd={float(row.get('avg_sgd_baseline_success_rate', float('nan'))):.4f} "
                                f"success_rate_no_oracle={float(row.get('avg_no_oracle_success_rate', float('nan'))):.4f} "
                                f"avg_E(F(z))={float(row['avg_final_objective']):.4f} "
                                f"avg_E(F(z))_gd_baseline={float(row['avg_baseline_final_objective']):.4f} "
                                f"avg_E(F(z))_sgd_baseline={float(row['avg_sgd_baseline_final_objective']):.4f} "
                                f"avg_E(F(z))_no_oracle={float(row['avg_no_oracle_final_objective']):.4f} "
                                f"avg_dist_ref={float(row['avg_final_ref_distance']):.4f} "
                                f"avg_dist_ref_gd_baseline={float(row['avg_baseline_final_ref_distance']):.4f} "
                                f"avg_dist_ref_sgd_baseline={float(row['avg_sgd_baseline_final_ref_distance']):.4f} "
                                f"avg_dist_ref_no_oracle={float(row['avg_no_oracle_final_ref_distance']):.4f} "
                                f"avg_path_len={avg_path_len_all:.3f} "
                                f"avg_shortest_dist={avg_shortest_dist_all:.3f}"
                            )
                    if (
                        config.task == "spatial"
                        and isinstance(env, VectorizedSpatialEnv)
                        and config.spatial_plot_interval_episodes > 0
                        and total_episodes % config.spatial_plot_interval_episodes == 0
                    ):
                        maybe_save_spatial_trajectory_plot(
                            model=model,
                            env=env,
                            device=device,
                            no_oracle_model=no_oracle_model,
                            output_path=run_dir
                            / f"spatial_trajectory_with_gradients_epi{int(total_episodes)}.png",
                            title=(
                                f"2D trajectory on energy landscape | D={config.spatial_hidden_dim}, "
                                f"mode={config.oracle_mode}, epi={int(total_episodes)}"
                            ),
                        )

            csv_file.flush()
            jsonl_file.flush()

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
            update_index += 1

        progress.close()

    if config.task == "graph":
        hist_context = f"n={config.n}, mode={config.oracle_mode}, sensing={config.sensing}"
    else:
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
    if isinstance(env, VectorizedSpatialEnv):
        maybe_save_spatial_trajectory_plot(
            model=model,
            env=env,
            device=device,
            no_oracle_model=no_oracle_model,
            output_path=run_dir / "spatial_trajectory_with_gradients.png",
            title=(
                f"2D trajectory on energy landscape | D={config.spatial_hidden_dim}, "
                f"mode={config.oracle_mode}"
            ),
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
    avg_sgd_baseline_final_ref_distance_all = (
        float(np.mean(all_sgd_baseline_final_ref_distances))
        if all_sgd_baseline_final_ref_distances
        else None
    )
    avg_no_oracle_final_ref_distance_all = (
        float(np.mean(all_no_oracle_final_ref_distances))
        if all_no_oracle_final_ref_distances
        else None
    )
    if avg_path_len_all is not None:
        print(f"Average path length over all episodes: {avg_path_len_all:.3f}")
    if avg_shortest_dist_all is not None:
        if config.task == "graph":
            print(f"Average shortest-path distance (s->t) over all tasks: {avg_shortest_dist_all:.3f}")
        else:
            print(f"Average proxy step distance to reference minimum: {avg_shortest_dist_all:.3f}")
    if config.task == "spatial" and avg_final_ref_distance_all is not None:
        print(
            "Average final Euclidean distance to reference minimum over all episodes: "
            f"{avg_final_ref_distance_all:.3f}"
        )
    if config.task == "spatial" and avg_baseline_final_ref_distance_all is not None:
        print(
            "Average GD baseline final Euclidean distance to reference minimum over all episodes: "
            f"{avg_baseline_final_ref_distance_all:.3f}"
        )
    if config.task == "spatial" and avg_sgd_baseline_final_ref_distance_all is not None:
        print(
            "Average SGD baseline final Euclidean distance to reference minimum over all episodes: "
            f"{avg_sgd_baseline_final_ref_distance_all:.3f}"
        )
    if config.task == "spatial" and avg_no_oracle_final_ref_distance_all is not None:
        print(
            "Average no-oracle baseline final Euclidean distance to reference minimum over all episodes: "
            f"{avg_no_oracle_final_ref_distance_all:.3f}"
        )

    final_cipher_accuracy = None
    if (
        config.task == "graph"
        and config.diagnostic_cipher
        and isinstance(env, VectorizedGraphEnv)
        and config.sigma_size == config.d
    ):
        final_cipher_accuracy = estimate_cipher_accuracy(model, env, device)

    summary = {
        "run_dir": str(run_dir),
        "model_architecture": model_architecture,
        "episodes_to_80": (episodes_to_80 if config.task == "graph" else None),
        "final_success_rate": (
            float(last_row["success_rate"])
            if (config.task == "graph" and last_row is not None)
            else None
        ),
        "final_objective_value": (
            float(last_row["avg_final_objective"])
            if (config.task == "spatial" and last_row is not None)
            else None
        ),
        "final_baseline_objective_value": (
            float(last_row["avg_baseline_final_objective"])
            if (config.task == "spatial" and last_row is not None)
            else None
        ),
        "final_sgd_baseline_objective_value": (
            float(last_row["avg_sgd_baseline_final_objective"])
            if (config.task == "spatial" and last_row is not None)
            else None
        ),
        "final_no_oracle_objective_value": (
            float(last_row["avg_no_oracle_final_objective"])
            if (config.task == "spatial" and last_row is not None)
            else None
        ),
        "final_distance_to_ref_value": (
            float(last_row["avg_final_ref_distance"])
            if (config.task == "spatial" and last_row is not None)
            else None
        ),
        "final_baseline_distance_to_ref_value": (
            float(last_row["avg_baseline_final_ref_distance"])
            if (config.task == "spatial" and last_row is not None)
            else None
        ),
        "final_sgd_baseline_distance_to_ref_value": (
            float(last_row["avg_sgd_baseline_final_ref_distance"])
            if (config.task == "spatial" and last_row is not None)
            else None
        ),
        "final_no_oracle_distance_to_ref_value": (
            float(last_row["avg_no_oracle_final_ref_distance"])
            if (config.task == "spatial" and last_row is not None)
            else None
        ),
        "avg_path_len_all_episodes": avg_path_len_all,
        "avg_shortest_dist_all_tasks": avg_shortest_dist_all,
        "num_points": len(curve_metrics),
        "running_avg_window": config.running_avg_window,
        "final_cipher_accuracy": final_cipher_accuracy,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "final_no_oracle_lr": (
            float(no_oracle_optimizer.param_groups[0]["lr"])
            if no_oracle_optimizer is not None
            else None
        ),
        "updates": update_index,
        "steps": total_steps,
        "episodes": total_episodes,
        "no_oracle_episodes": (total_no_oracle_episodes if no_oracle_model is not None else None),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "summary": summary,
        "metrics": curve_metrics,
        "config": asdict(config),
    }


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
    parser = argparse.ArgumentParser(description="Goal-oriented communication RL prototype")
    parser.add_argument("--task", type=str, choices=["graph", "spatial"], default=defaults.task)
    parser.add_argument("--n", type=int, default=defaults.n)
    parser.add_argument("--d", type=int, default=defaults.d)
    parser.add_argument("--t_pool", type=int, default=defaults.t_pool)
    parser.add_argument("--spatial_hidden_dim", type=int, default=defaults.spatial_hidden_dim)
    parser.add_argument("--spatial_visible_dim", type=int, default=defaults.spatial_visible_dim)
    parser.add_argument("--spatial_coord_limit", type=int, default=defaults.spatial_coord_limit)
    parser.add_argument("--spatial_token_dim", type=int, default=defaults.spatial_token_dim)
    parser.add_argument("--spatial_token_noise_std", type=float, default=defaults.spatial_token_noise_std)
    parser.add_argument("--spatial_step_size", type=float, default=defaults.spatial_step_size)
    parser.add_argument(
        "--spatial_sgd_gradient_noise_std",
        type=float,
        default=defaults.spatial_sgd_gradient_noise_std,
        help="Std-dev of additive Gaussian noise on the 2D gradient for the SGD baseline.",
    )
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
        "--spatial_plot_interval_episodes",
        type=int,
        default=defaults.spatial_plot_interval_episodes,
        help="For spatial task, save trajectory snapshots every N episodes (<=0 disables)",
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
        choices=sorted(ORACLE_MODES | SPATIAL_ORACLE_MODES),
        default=defaults.oracle_mode,
    )
    parser.add_argument("--sigma_size", type=int, default=defaults.sigma_size)
    parser.add_argument("--fst_k", type=int, default=defaults.fst_k)
    parser.add_argument("--lie_prob", type=float, default=defaults.lie_prob)
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
    parser.add_argument("--device", type=str, default=defaults.device)

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
    parser.add_argument("--token_embed_dim", type=int, default=defaults.token_embed_dim)
    parser.add_argument("--s1_step_penalty", type=float, default=defaults.s1_step_penalty)
    parser.add_argument("--no_diagnostic_cipher", action="store_true")

    args = parser.parse_args()
    seed_values = _parse_seed_values(args.seed)
    config = TrainConfig(
        task=args.task,
        n=args.n,
        d=args.d,
        t_pool=args.t_pool,
        spatial_hidden_dim=args.spatial_hidden_dim,
        spatial_visible_dim=args.spatial_visible_dim,
        spatial_coord_limit=args.spatial_coord_limit,
        spatial_token_dim=args.spatial_token_dim,
        spatial_token_noise_std=args.spatial_token_noise_std,
        spatial_step_size=args.spatial_step_size,
        spatial_sgd_gradient_noise_std=args.spatial_sgd_gradient_noise_std,
        spatial_success_threshold=args.spatial_success_threshold,
        spatial_basis_complexity=args.spatial_basis_complexity,
        spatial_f_type=args.spatial_f_type,
        spatial_policy_arch=args.spatial_policy_arch,
        spatial_refresh_map_each_episode=args.spatial_refresh_map_each_episode,
        spatial_plot_interval_episodes=args.spatial_plot_interval_episodes,
        n_env=args.n_env,
        train_steps=args.train_steps,
        rollout_len=args.rollout_len,
        algo=args.algo,
        sensing=args.sensing,
        reward_noise_std=args.reward_noise_std,
        oracle_mode=args.oracle_mode,
        sigma_size=args.sigma_size,
        fst_k=args.fst_k,
        lie_prob=args.lie_prob,
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
        token_embed_dim=args.token_embed_dim,
        s1_step_penalty=args.s1_step_penalty,
        diagnostic_cipher=not args.no_diagnostic_cipher,
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
    summary_stem = base_run_name or (
        f"{config.task}_{config.oracle_mode}_{config.sensing}_multi_seed"
    )
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
