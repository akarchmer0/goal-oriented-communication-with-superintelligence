import argparse
import csv
import json
import warnings
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import TrainConfig
from .distances import INF_DISTANCE, precompute_distance_pool
from .env import VectorizedGraphEnv
from .graph import build_directed_regular_graph
from .model import PolicyValueNet
from .oracle import ORACLE_MODES, Oracle
from .plotting import plot_path_length_histograms
from .ppo import PPOHyperParams, RolloutBuffer, compute_gae, ppo_update


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
    env: VectorizedGraphEnv,
    eval_episodes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    success_count = 0
    path_lengths: list[int] = []
    stretches: list[float] = []

    for _ in range(eval_episodes):
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


def _resolve_device(device_str: str) -> torch.device:
    """Resolve device from config; use CUDA if requested and available."""
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        warnings.warn("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def run_training(config: TrainConfig, return_artifacts: bool = False) -> dict:
    # Force graph task
    config = replace(config, task="graph")

    if config.algo != "ppo":
        raise ValueError("Only --algo ppo is implemented in this prototype")
    if config.running_avg_window < 1:
        raise ValueError("running_avg_window must be >= 1")
    if config.oracle_mode not in ORACLE_MODES:
        raise ValueError(
            f"oracle_mode={config.oracle_mode!r} is invalid for task='graph'; "
            f"choose one of {sorted(ORACLE_MODES)}"
        )
    if config.lr_scheduler not in {"none", "constant", "linear", "cosine"}:
        raise ValueError(
            "lr_scheduler must be one of {'none', 'constant', 'linear', 'cosine'}"
        )
    if config.lr_min_factor <= 0.0 or config.lr_min_factor > 1.0:
        raise ValueError("lr_min_factor must be in (0, 1]")
    if config.lr_warmup_updates < 0:
        raise ValueError("lr_warmup_updates must be >= 0")

    set_global_seed(config.seed)
    device = _resolve_device(config.device)

    run_dir = config.resolve_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    # Build graph environment
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
    env = VectorizedGraphEnv(
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
        "avg_path_len",
        "avg_shortest_dist",
        "avg_stretch",
        "window_size",
        "policy_loss",
        "value_loss",
        "entropy",
        "lr",
    ]

    recent_success: deque[float] = deque(maxlen=config.running_avg_window)
    recent_path_len: deque[float] = deque(maxlen=config.running_avg_window)
    recent_shortest_dist: deque[float] = deque(maxlen=config.running_avg_window)
    recent_stretch: deque[float] = deque(maxlen=config.running_avg_window)
    all_path_lengths: list[float] = []
    all_shortest_dists: list[float] = []
    success_path_lengths: list[float] = []
    failure_path_lengths: list[float] = []

    with metric_csv_path.open("w", newline="", encoding="utf-8") as csv_file, metric_jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=metric_fields)
        writer.writeheader()

        total_steps = 0
        total_episodes = 0
        update_index = 0
        last_update_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        progress = tqdm(total=config.train_steps, desc="Training", leave=False)

        while total_steps < config.train_steps:
            buffer.reset()

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

                done_indices = np.where(dones)[0]
                for env_index in done_indices:
                    info = infos[int(env_index)]
                    if not info.get("episode_done", False):
                        continue

                    total_episodes += 1
                    success = 1.0 if bool(info.get("success", False)) else 0.0
                    episode_len = float(info.get("episode_len", 0))
                    shortest_dist = float(info.get("shortest_dist", 0))
                    recent_success.append(success)
                    recent_path_len.append(episode_len)
                    recent_shortest_dist.append(shortest_dist)
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
                    }

                    last_row = row
                    progress.set_postfix(
                        success_rate=f"{float(row['success_rate']):.2%}",
                        refresh=True,
                    )
                    if episodes_to_80 is None and float(row["success_rate"]) >= 0.8:
                        episodes_to_80 = int(total_episodes)

                    should_sample_curve = save_every_episode or (
                        save_metrics_interval > 0 and total_episodes % save_metrics_interval == 0
                    )
                    if should_sample_curve:
                        curve_metrics.append(
                            {
                                "episodes": int(total_episodes),
                                "success_rate": float(row["success_rate"]),
                            }
                        )
                    writer.writerow(row)
                    jsonl_file.write(json.dumps(row) + "\n")

                    if config.eval_interval_episodes > 0 and total_episodes % config.eval_interval_episodes == 0:
                        avg_path_len_all = float(np.mean(all_path_lengths)) if all_path_lengths else float("nan")
                        avg_shortest_dist_all = (
                            float(np.mean(all_shortest_dists)) if all_shortest_dists else float("nan")
                        )
                        tqdm.write(
                            "episodes="
                            f"{total_episodes} steps={total_steps} "
                            f"success_rate={float(row['success_rate']):.4f} "
                            f"avg_path_len={avg_path_len_all:.3f} "
                            f"avg_shortest_dist={avg_shortest_dist_all:.3f}"
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
            update_index += 1

        progress.close()

    if config.enable_training_plots:
        hist_context = f"n={config.n}, mode={config.oracle_mode}, sensing={config.sensing}"
        plot_path_length_histograms(
            path_lengths=all_path_lengths,
            success_path_lengths=success_path_lengths,
            failure_path_lengths=failure_path_lengths,
            output_dir=run_dir,
            title_prefix=hist_context,
        )

    avg_path_len_all = float(np.mean(all_path_lengths)) if all_path_lengths else None
    avg_shortest_dist_all = float(np.mean(all_shortest_dists)) if all_shortest_dists else None
    if avg_path_len_all is not None:
        print(f"Average path length over all episodes: {avg_path_len_all:.3f}")
    if avg_shortest_dist_all is not None:
        print(f"Average shortest-path distance (s->t) over all tasks: {avg_shortest_dist_all:.3f}")

    final_cipher_accuracy = None
    if (
        config.diagnostic_cipher
        and config.sigma_size == config.d
    ):
        final_cipher_accuracy = estimate_cipher_accuracy(model, env, device)

    summary = {
        "run_dir": str(run_dir),
        "model_architecture": model_architecture,
        "episodes_to_80": episodes_to_80,
        "final_success_rate": (
            float(last_row["success_rate"])
            if last_row is not None
            else None
        ),
        "avg_path_len_all_episodes": avg_path_len_all,
        "avg_shortest_dist_all_tasks": avg_shortest_dist_all,
        "num_points": len(curve_metrics),
        "running_avg_window": config.running_avg_window,
        "final_cipher_accuracy": final_cipher_accuracy,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
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
    }
    if return_artifacts:
        output["model"] = model
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
    parser = argparse.ArgumentParser(description="Graph navigation RL training")
    parser.add_argument("--n", type=int, default=defaults.n)
    parser.add_argument("--d", type=int, default=defaults.d)
    parser.add_argument("--t_pool", type=int, default=defaults.t_pool)
    parser.add_argument("--n_env", type=int, default=defaults.n_env)
    parser.add_argument("--train_steps", type=int, default=defaults.train_steps)
    parser.add_argument("--rollout_len", type=int, default=defaults.rollout_len)
    parser.add_argument("--algo", type=str, default=defaults.algo)
    parser.add_argument("--sensing", type=str, choices=["S0", "S1"], default=defaults.sensing)
    parser.add_argument("--reward_noise_std", type=float, default=defaults.reward_noise_std)
    parser.add_argument(
        "--oracle_mode",
        type=str,
        choices=sorted(ORACLE_MODES),
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
    parser.add_argument("--token_embed_dim", type=int, default=defaults.token_embed_dim)
    parser.add_argument("--s1_step_penalty", type=float, default=defaults.s1_step_penalty)
    parser.add_argument("--no_diagnostic_cipher", action="store_true")
    parser.add_argument(
        "--disable_training_plots",
        action="store_true",
        help="Skip automatic per-run matplotlib artifacts (path-length PNGs).",
    )

    args = parser.parse_args()
    seed_values = _parse_seed_values(args.seed)
    config = TrainConfig(
        task="graph",
        n=args.n,
        d=args.d,
        t_pool=args.t_pool,
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
        f"graph_{config.oracle_mode}_{config.sensing}_multi_seed"
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
