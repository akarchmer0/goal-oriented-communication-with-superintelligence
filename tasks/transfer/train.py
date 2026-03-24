"""Transfer experiment: meta-learn lifting map, train PPO, evaluate on alanine.

Pipeline:
  1. Meta-train neural lifting map F: T^d -> R^D  (offline, no RL).
  2. Freeze F.  For each random surface, fit a PSD quadratic Q in F-space.
  3. Train PPO policy with oracle = nabla_s Q = 2As + b.
  4. Load alanine dipeptide, fit quadratic, evaluate zero-shot.

Usage:
    python -m tasks.transfer.train --D 32 --meta_steps 3000 --train_steps 300000
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tasks.alanine_dipeptide.energy import EnergySurface
from tasks.alanine_dipeptide.model import PolicyValueNet
from tasks.alanine_dipeptide.ppo import PPOHyperParams, RolloutBuffer, compute_gae, ppo_update

from .eval_env import TransferEvalEnv
from .lifting_net import LiftingNet, n_diagonal_params, n_quadratic_params
from .meta_train import meta_train_lifting
from .train_env import VectorizedTransferTrainEnv, fit_quadratic_for_surface


# ──────────────────────────────────────────────────────────────────────
# Reward normalisation
# ──────────────────────────────────────────────────────────────────────


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
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        self.mean = float(new_mean)
        self.var = float(max(m2 / total_count, 0.0))
        self.count = float(total_count)


def normalize_rewards(
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
    normalised = (rewards_f64 / scale).astype(np.float32)
    if clip_abs > 0.0:
        normalised = np.clip(normalised, -clip_abs, clip_abs).astype(np.float32)
    updated_returns = updated_returns * (1.0 - dones_f64)
    return normalised, updated_returns


# ──────────────────────────────────────────────────────────────────────
# LR scheduler
# ──────────────────────────────────────────────────────────────────────


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    mode: str,
    min_factor: float,
    warmup_updates: int,
    total_updates: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if mode == "none":
        return None
    decay_updates = max(1, total_updates - warmup_updates)

    def _lambda(u: int) -> float:
        if warmup_updates > 0 and u < warmup_updates:
            return max(1e-8, float(u + 1) / float(warmup_updates))
        if mode == "constant":
            return 1.0
        t = float(np.clip((u - warmup_updates) / float(decay_updates), 0.0, 1.0))
        if mode == "linear":
            return float(min_factor + (1.0 - min_factor) * (1.0 - t))
        if mode == "cosine":
            return float(
                min_factor + 0.5 * (1.0 - min_factor) * (1.0 + np.cos(np.pi * t))
            )
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lambda)


# ──────────────────────────────────────────────────────────────────────
# obs helpers
# ──────────────────────────────────────────────────────────────────────


def obs_to_tensors(
    obs: dict[str, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(obs["token_features"], dtype=torch.float32, device=device),
        torch.as_tensor(obs["dist"], dtype=torch.float32, device=device),
        torch.as_tensor(obs["step_frac"], dtype=torch.float32, device=device),
    )


# ──────────────────────────────────────────────────────────────────────
# PPO training loop
# ──────────────────────────────────────────────────────────────────────


def train_on_random_surfaces(
    env: VectorizedTransferTrainEnv,
    *,
    train_steps: int,
    hidden_dim: int = 64,
    oracle_proj_dim: int = 64,
    rollout_len: int = 64,
    ppo_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    minibatches: int = 4,
    lr: float = 3e-4,
    lr_scheduler: str = "cosine",
    lr_min_factor: float = 0.1,
    lr_warmup_updates: int = 0,
    eval_interval_episodes: int = 200,
    running_avg_window: int = 100,
    device_str: str = "cpu",
    lattice_rl: bool = False,
) -> dict:
    """Train PPO on the transfer training env.  Returns model + stats."""
    device = torch.device(device_str)
    n_env = env.n_env
    action_space_type = "discrete" if lattice_rl else "continuous"

    model = PolicyValueNet(
        token_feature_dim=env.token_feature_dim,
        oracle_token_dim=env.oracle_token_dim,
        action_dim=env.action_dim,
        hidden_dim=hidden_dim,
        oracle_proj_dim=oracle_proj_dim,
        architecture="mlp",
        action_space_type=action_space_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    updates_total = int(np.ceil(train_steps / max(1, n_env * rollout_len)))
    scheduler = build_lr_scheduler(
        optimizer, lr_scheduler, lr_min_factor, lr_warmup_updates, updates_total
    )

    hparams = PPOHyperParams(
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        minibatches=minibatches,
    )

    buffer = RolloutBuffer(
        rollout_len=rollout_len,
        n_env=n_env,
        token_feature_dim=env.token_feature_dim,
        action_dim=env.action_dim,
        action_dtype=action_space_type,
    )

    obs = env.get_obs()
    recurrent_state = model.initial_state(batch_size=n_env, device=device)
    reward_rms = RunningMeanStd()
    reward_returns = np.zeros(n_env, dtype=np.float64)

    recent_success: deque[float] = deque(maxlen=running_avg_window)
    recent_path_len: deque[float] = deque(maxlen=running_avg_window)
    recent_final_obj: deque[float] = deque(maxlen=running_avg_window)

    total_steps = 0
    total_episodes = 0
    last_log_episodes = 0
    pbar = tqdm(total=train_steps, desc="PPO training", unit="step")

    for _ in range(updates_total):
        # ── Collect rollout ──
        for _ in range(rollout_len):
            token_t, dist_t, step_t = obs_to_tensors(obs, device)
            with torch.no_grad():
                action_t, log_prob_t, value_t, new_recurrent = model.act(
                    token_t, dist_t, step_t, hidden_state=recurrent_state
                )

            actions_np = action_t.cpu().numpy()
            next_obs, rewards, dones, infos = env.step(actions_np)

            norm_rewards, reward_returns = normalize_rewards(
                rewards, dones, reward_returns, reward_rms, gamma
            )

            buffer.add(
                obs=obs,
                action=actions_np,
                logprob=log_prob_t.cpu().numpy(),
                reward=norm_rewards,
                done=dones,
                value=value_t.cpu().numpy(),
            )

            obs = next_obs
            recurrent_state = new_recurrent
            total_steps += n_env

            for info in infos:
                if info.get("episode_done"):
                    total_episodes += 1
                    recent_success.append(1.0 if info["success"] else 0.0)
                    recent_path_len.append(float(info["episode_len"]))
                    recent_final_obj.append(float(info["final_objective"]))

            if recurrent_state is not None:
                for i in range(n_env):
                    if dones[i]:
                        recurrent_state[:, i, :] = 0.0

        # ── Bootstrap & GAE ──
        with torch.no_grad():
            token_t, dist_t, step_t = obs_to_tensors(obs, device)
            _, _, next_value, _ = model.act(
                token_t, dist_t, step_t, hidden_state=recurrent_state
            )

        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.dones,
            buffer.values,
            next_value.cpu().numpy(),
            hparams.gamma,
            hparams.gae_lambda,
        )

        ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            advantages=advantages,
            returns=returns,
            hparams=hparams,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()
        buffer.reset()

        # ── Progress ──
        pbar.update(n_env * rollout_len)
        if recent_success and total_episodes > last_log_episodes + eval_interval_episodes:
            last_log_episodes = total_episodes
            sr = float(np.mean(recent_success))
            pl = float(np.mean(recent_path_len)) if recent_path_len else 0.0
            fo = float(np.mean(recent_final_obj)) if recent_final_obj else 0.0
            pbar.set_postfix(ep=total_episodes, sr=f"{sr:.3f}", path=f"{pl:.1f}", obj=f"{fo:.4f}")
            tqdm.write(
                f"episodes={total_episodes} steps={total_steps} "
                f"success_rate={sr:.4f} avg_path_len={pl:.1f} avg_final_obj={fo:.4f}"
            )

        if total_steps >= train_steps:
            break

    pbar.close()
    return {
        "model": model,
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "final_success_rate": float(np.mean(recent_success)) if recent_success else 0.0,
        "final_avg_path_len": float(np.mean(recent_path_len)) if recent_path_len else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_transfer(
    model: PolicyValueNet,
    env: TransferEvalEnv,
    eval_episodes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    successes = 0
    final_energies: list[float] = []
    final_distances: list[float] = []
    path_lengths: list[int] = []

    for _ in range(eval_episodes):
        spec = env.sample_episode_spec()
        state = spec.source.copy()
        if env.lattice_rl:
            state = env._snap_to_lattice(state)
        hidden_state = model.initial_state(batch_size=1, device=device)

        for step in range(spec.horizon):
            tf = env._obs_token_features(state)[None, :]
            dist_val = env._normalized_objective_value(state)
            step_frac = float(step / max(1, spec.horizon))
            token_t = torch.tensor(tf, dtype=torch.float32, device=device)
            dist_t = torch.tensor([dist_val], dtype=torch.float32, device=device)
            step_t = torch.tensor([step_frac], dtype=torch.float32, device=device)
            action_t, _, _, hidden_state = model.act(
                token_t, dist_t, step_t, hidden_state=hidden_state, deterministic=True
            )
            if env.lattice_rl:
                state = env._apply_action(state, action_t.item())
            else:
                state = env._apply_action(state, action_t.squeeze(0).cpu().numpy())
            if env._is_success(state):
                successes += 1
                path_lengths.append(step + 1)
                break
        else:
            path_lengths.append(spec.horizon)

        final_energies.append(env._energy_value(state))
        final_distances.append(env._reference_distance(state))

    model.train()
    return {
        "success_rate": float(successes / max(1, eval_episodes)),
        "avg_final_energy": float(np.mean(final_energies)),
        "avg_final_distance": float(np.mean(final_distances)),
        "avg_path_length": float(np.mean(path_lengths)),
        "median_final_energy": float(np.median(final_energies)),
    }


def evaluate_baselines(
    env: TransferEvalEnv, eval_episodes: int
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    runners = [
        ("gd_energy", env.rollout_gd_baseline, env.baseline_lr_gd),
        ("adam_energy", env.rollout_adam_baseline, env.baseline_lr_adam),
        ("gd_surrogate", env.rollout_surrogate_gd_baseline, env.baseline_lr_gd),
    ]
    for name, runner, lr_val in runners:
        successes = 0
        final_energies: list[float] = []
        final_distances: list[float] = []
        for _ in range(eval_episodes):
            spec = env.sample_episode_spec()
            energy, dist, success = runner(
                start_xy=spec.source, horizon=spec.horizon, lr=lr_val
            )
            final_energies.append(energy)
            final_distances.append(dist)
            if success:
                successes += 1
        results[name] = {
            "success_rate": float(successes / max(1, eval_episodes)),
            "avg_final_energy": float(np.mean(final_energies)),
            "avg_final_distance": float(np.mean(final_distances)),
            "median_final_energy": float(np.median(final_energies)),
            "lr": lr_val,
        }
    return results


# ──────────────────────────────────────────────────────────────────────
# Baseline LR tuning
# ──────────────────────────────────────────────────────────────────────

GD_TUNING_LRS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
ADAM_TUNING_LRS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.05)


def tune_eval_baseline_lrs(
    env: TransferEvalEnv, num_tasks: int = 200
) -> dict[str, float]:
    tasks = []
    for _ in range(num_tasks):
        spec = env.sample_episode_spec()
        tasks.append({"start_xy": spec.source.copy(), "horizon": spec.horizon})

    best_lrs: dict[str, float] = {}
    for method, lrs, runner_fn in [
        ("gd", GD_TUNING_LRS, env.rollout_gd_baseline),
        ("adam", ADAM_TUNING_LRS, env.rollout_adam_baseline),
    ]:
        scores = []
        for lr_val in lrs:
            energies = []
            for task in tasks:
                energy, _, _ = runner_fn(
                    start_xy=task["start_xy"], horizon=task["horizon"], lr=lr_val
                )
                energies.append(energy)
            avg_e = float(np.mean(energies))
            scores.append((avg_e, lr_val))
            print(f"    {method} lr={lr_val:.4f}: avg_energy={avg_e:.4f}")
        best = min(scores, key=lambda x: x[0])
        best_lrs[method] = best[1]
        print(f"  Best {method} lr: {best_lrs[method]:.4f}")
    return best_lrs


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────


def run_transfer(
    *,
    # Lifting net / meta-training
    D: int = 32,
    net_hidden_dims: tuple[int, ...] = (128, 128),
    meta_steps: int = 3000,
    meta_lr: float = 1e-3,
    meta_n_surfaces: int = 16,
    meta_n_fit: int = 512,
    meta_n_test: int = 256,
    meta_reg_lambda: float = 1e-4,
    meta_psd_eps: float = 1e-6,
    diagonal: bool = False,
    # Training surfaces
    K_energy: int = 10,
    visible_dim: int = 2,
    amplitude_scale: float = 5.0,
    freq_sparsity: int = 0,
    # PPO training env
    n_env: int = 32,
    refresh_surface_each_episode: bool = False,
    n_fit_points: int = 1024,
    ppo_reg_lambda: float = 1e-4,
    ppo_psd_eps: float = 1e-6,
    lattice_rl: bool = False,
    lattice_granularity: int = 20,
    # PPO
    train_steps: int = 300_000,
    hidden_dim: int = 64,
    oracle_proj_dim: int = 64,
    rollout_len: int = 64,
    ppo_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    minibatches: int = 4,
    lr: float = 3e-4,
    lr_scheduler: str = "cosine",
    lr_min_factor: float = 0.1,
    lr_warmup_updates: int = 0,
    # Env params
    sensing: str = "S0",
    max_horizon: int = 60,
    step_size: float = 0.3,
    ppo_step_scale: float = 1.0,
    success_threshold: float = 0.01,
    s1_step_penalty: float = -0.01,
    reward_noise_std: float = 0.0,
    # Eval
    eval_energy_json: str = "tasks/alanine_dipeptide/fourier_coefficients.json",
    eval_episodes: int = 500,
    eval_max_horizon: int = 120,
    eval_success_threshold: float = 0.05,
    eval_seed: int = 999,
    tune_lrs: bool = True,
    lr_tune_tasks: int = 200,
    run_baselines: bool = True,
    # Misc
    seed: int = 0,
    logdir: str = "runs",
    run_name: str = "",
    device: str = "cpu",
    eval_interval_episodes: int = 200,
    running_avg_window: int = 100,
) -> dict:
    """Full pipeline: meta-train F -> PPO train -> alanine eval."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    d = visible_dim
    p = n_diagonal_params(D) if diagonal else n_quadratic_params(D)
    mode_str = "diagonal" if diagonal else "full PSD"
    print(f"Lifting: d={d}, D={D}, mode={mode_str}, n_params={p}")

    # ── Phase 1: Meta-train the lifting map F ────────────────────────
    print("\n" + "=" * 70)
    print(f"Phase 1: Meta-training lifting net F  (D={D}, {meta_steps} steps)")
    print("=" * 70)

    t0 = time.time()
    lifting_net, meta_stats = meta_train_lifting(
        visible_dim=d,
        lifting_dim=D,
        net_hidden_dims=net_hidden_dims,
        K_energy=K_energy,
        amplitude_scale=amplitude_scale,
        freq_sparsity=freq_sparsity,
        n_surfaces_per_batch=meta_n_surfaces,
        n_fit=meta_n_fit,
        n_test=meta_n_test,
        reg_lambda=meta_reg_lambda,
        psd_eps=meta_psd_eps,
        diagonal=diagonal,
        meta_lr=meta_lr,
        meta_steps=meta_steps,
        seed=seed,
        device=device,
    )
    meta_time = time.time() - t0
    print(
        f"\nMeta-training done in {meta_time:.1f}s  "
        f"(loss={meta_stats['final_loss']:.4f}, R²={meta_stats['final_r2']:.4f})"
    )

    # Freeze F
    lifting_net.eval()
    for param in lifting_net.parameters():
        param.requires_grad_(False)

    # ── Phase 2: PPO training on random surfaces ─────────────────────
    print("\n" + "=" * 70)
    print(f"Phase 2: PPO training ({train_steps} steps, {n_env} envs)")
    print("=" * 70)

    train_env = VectorizedTransferTrainEnv(
        lifting_net=lifting_net,
        n_env=n_env,
        K_energy=K_energy,
        max_horizon=max_horizon,
        seed=seed + 41,
        sensing=sensing,
        s1_step_penalty=s1_step_penalty,
        reward_noise_std=reward_noise_std,
        step_size=step_size,
        ppo_step_scale=ppo_step_scale,
        success_threshold=success_threshold,
        amplitude_scale=amplitude_scale,
        refresh_surface_each_episode=refresh_surface_each_episode,
        n_fit_points=n_fit_points,
        reg_lambda=ppo_reg_lambda,
        psd_eps=ppo_psd_eps,
        diagonal=diagonal,
        freq_sparsity=freq_sparsity,
        lattice_rl=lattice_rl,
        lattice_granularity=lattice_granularity,
    )

    t0 = time.time()
    train_result = train_on_random_surfaces(
        env=train_env,
        train_steps=train_steps,
        hidden_dim=hidden_dim,
        oracle_proj_dim=oracle_proj_dim,
        rollout_len=rollout_len,
        ppo_epochs=ppo_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        minibatches=minibatches,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        lr_warmup_updates=lr_warmup_updates,
        eval_interval_episodes=eval_interval_episodes,
        running_avg_window=running_avg_window,
        device_str=device,
        lattice_rl=lattice_rl,
    )
    train_time = time.time() - t0
    model = train_result["model"]
    print(
        f"\nPPO training done in {train_time:.1f}s  "
        f"(episodes={train_result['total_episodes']}, "
        f"sr={train_result['final_success_rate']:.4f})"
    )

    # ── Phase 3: Load alanine + fit quadratic ────────────────────────
    print("\n" + "=" * 70)
    print("Phase 3: Loading alanine dipeptide + fitting quadratic")
    print("=" * 70)

    energy_surface = EnergySurface.from_json(eval_energy_json)
    print(f"  d={energy_surface.d}, K_energy={energy_surface.K_energy}")
    print(f"  global_min_energy={energy_surface.global_min_energy:.4f}")

    if energy_surface.d != d:
        raise ValueError(
            f"Alanine has d={energy_surface.d} but training used d={d}"
        )

    coeff_eval, b_eval, c_eval = fit_quadratic_for_surface(
        lifting_net,
        energy_surface,
        n_fit=max(n_fit_points, 4096),
        reg_lambda=ppo_reg_lambda,
        psd_eps=ppo_psd_eps,
        seed=eval_seed + 42,
        diagonal=diagonal,
    )

    eval_env = TransferEvalEnv(
        lifting_net=lifting_net,
        A=coeff_eval,
        b=b_eval,
        c=c_eval,
        energy_surface=energy_surface,
        step_size=step_size,
        ppo_step_scale=ppo_step_scale,
        max_horizon=eval_max_horizon,
        success_threshold=eval_success_threshold,
        seed=eval_seed,
        sensing=sensing,
        s1_step_penalty=s1_step_penalty,
        diagonal=diagonal,
        lattice_rl=lattice_rl,
        lattice_granularity=lattice_granularity,
    )

    # ── Phase 4: Quadratic fit quality diagnostic ────────────────────
    print("\n" + "=" * 70)
    print("Phase 4: Quadratic fit quality (diagnostic)")
    print("=" * 70)

    fit_result = eval_env.fit_quadratic_quality()
    print(f"  R²  = {fit_result['r2']:.4f}")
    print(f"  MAE = {fit_result['mae']:.4f}")
    print(f"  MSE = {fit_result['mse']:.4f}")

    # ── Phase 5: Tune eval baseline LRs ──────────────────────────────
    if tune_lrs and run_baselines:
        print("\n" + "=" * 70)
        print(f"Phase 5: Tuning eval baseline LRs ({lr_tune_tasks} tasks)")
        print("=" * 70)
        best_lrs = tune_eval_baseline_lrs(eval_env, num_tasks=lr_tune_tasks)
        eval_env.baseline_lr_gd = best_lrs["gd"]
        eval_env.baseline_lr_adam = best_lrs["adam"]

    # ── Phase 6: Zero-shot evaluation ────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Phase 6: Zero-shot evaluation ({eval_episodes} episodes)")
    print("=" * 70)

    rl_results = evaluate_transfer(
        model=model,
        env=eval_env,
        eval_episodes=eval_episodes,
        device=torch.device(device),
    )
    print(f"\n  RL (zero-shot):")
    print(f"    success_rate     = {rl_results['success_rate']:.4f}")
    print(f"    avg_final_energy = {rl_results['avg_final_energy']:.4f}")
    print(f"    med_final_energy = {rl_results['median_final_energy']:.4f}")
    print(f"    avg_final_dist   = {rl_results['avg_final_distance']:.4f}")
    print(f"    avg_path_length  = {rl_results['avg_path_length']:.1f}")

    baseline_results: dict[str, dict[str, float]] = {}
    if run_baselines:
        print(
            f"\n  Running baselines "
            f"(gd_lr={eval_env.baseline_lr_gd:.4f}, "
            f"adam_lr={eval_env.baseline_lr_adam:.4f})..."
        )
        baseline_results = evaluate_baselines(eval_env, eval_episodes)
        for name, res in baseline_results.items():
            print(f"\n  {name} (lr={res['lr']:.4f}):")
            print(f"    success_rate     = {res['success_rate']:.4f}")
            print(f"    avg_final_energy = {res['avg_final_energy']:.4f}")
            print(f"    med_final_energy = {res['median_final_energy']:.4f}")
            print(f"    avg_final_dist   = {res['avg_final_distance']:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    diag_tag = "_diag" if diagonal else ""
    sparse_tag = f"_sp{freq_sparsity}" if freq_sparsity > 0 else ""
    default_name = f"transfer_D{D}_K{K_energy}{diag_tag}{sparse_tag}_meta{meta_steps}_seed{seed}"
    rn = run_name or default_name
    run_dir = Path(logdir) / rn
    run_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "transfer_config": {
            "D": D,
            "d": d,
            "K_energy": K_energy,
            "diagonal": diagonal,
            "freq_sparsity": freq_sparsity,
            "net_hidden_dims": list(net_hidden_dims),
            "meta_steps": meta_steps,
            "meta_lr": meta_lr,
            "meta_n_surfaces": meta_n_surfaces,
            "meta_n_fit": meta_n_fit,
            "meta_n_test": meta_n_test,
            "n_env": n_env,
            "n_fit_points": n_fit_points,
            "amplitude_scale": amplitude_scale,
            "lattice_rl": lattice_rl,
            "lattice_granularity": lattice_granularity,
        },
        "meta_training": {
            "meta_time_sec": meta_time,
            "final_loss": meta_stats["final_loss"],
            "final_r2": meta_stats["final_r2"],
        },
        "ppo_training": {
            "train_steps": train_steps,
            "train_time_sec": train_time,
            "total_episodes": train_result["total_episodes"],
            "final_success_rate": train_result["final_success_rate"],
            "final_avg_path_len": train_result["final_avg_path_len"],
        },
        "eval_config": {
            "eval_energy_json": eval_energy_json,
            "eval_episodes": eval_episodes,
            "eval_max_horizon": eval_max_horizon,
            "eval_success_threshold": eval_success_threshold,
            "eval_seed": eval_seed,
            "baseline_lr_gd": eval_env.baseline_lr_gd,
            "baseline_lr_adam": eval_env.baseline_lr_adam,
        },
        "quadratic_fit": fit_result,
        "rl_zero_shot": rl_results,
        "baselines": baseline_results,
    }

    results_path = run_dir / "transfer_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save lifting net + policy
    torch.save(lifting_net.state_dict(), run_dir / "lifting_net.pt")
    torch.save(model.state_dict(), run_dir / "policy.pt")

    return output


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Transfer: meta-learn F, train PPO, evaluate on alanine"
    )

    # Lifting / meta-training
    mg = parser.add_argument_group("Lifting net / meta-training")
    mg.add_argument("--D", type=int, default=32, help="Lifting dimension")
    mg.add_argument(
        "--net_hidden_dims",
        type=str,
        default="128,128",
        help="Comma-separated hidden layer sizes for F",
    )
    mg.add_argument("--meta_steps", type=int, default=3000)
    mg.add_argument("--meta_lr", type=float, default=1e-3)
    mg.add_argument("--meta_n_surfaces", type=int, default=16)
    mg.add_argument("--meta_n_fit", type=int, default=512)
    mg.add_argument("--meta_n_test", type=int, default=256)
    mg.add_argument("--meta_reg_lambda", type=float, default=1e-4)
    mg.add_argument("--meta_psd_eps", type=float, default=1e-6)
    mg.add_argument(
        "--diagonal", action="store_true",
        help="Use diagonal quadratic (2D+1 params) instead of full PSD (D(D+1)/2+D+1 params)",
    )

    # Training surfaces
    sg = parser.add_argument_group("Training surfaces")
    sg.add_argument("--K_energy", type=int, default=10)
    sg.add_argument("--visible_dim", type=int, default=2)
    sg.add_argument("--amplitude_scale", type=float, default=5.0)
    sg.add_argument(
        "--freq_sparsity", type=int, default=0,
        help="Max nonzero components per frequency vector (0=dense, 1=axis-aligned, etc.)",
    )

    # PPO training env
    eg = parser.add_argument_group("PPO training env")
    eg.add_argument("--n_env", type=int, default=32)
    eg.add_argument("--refresh_surface_each_episode", action="store_true")
    eg.add_argument("--n_fit_points", type=int, default=1024)
    eg.add_argument("--ppo_reg_lambda", type=float, default=1e-4)
    eg.add_argument("--ppo_psd_eps", type=float, default=1e-6)
    eg.add_argument(
        "--lattice_RL",
        action="store_true",
        help="Enable lattice-grid discrete PPO: the RL agent selects among moves to adjacent lattice nodes.",
    )
    eg.add_argument(
        "--lattice_granularity",
        type=int,
        default=20,
        help="Number of lattice nodes per dimension (default: 20).",
    )

    # PPO
    tr = parser.add_argument_group("PPO training")
    tr.add_argument("--train_steps", type=int, default=300_000)
    tr.add_argument("--rollout_len", type=int, default=64)
    tr.add_argument("--ppo_epochs", type=int, default=4)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--gae_lambda", type=float, default=0.95)
    tr.add_argument("--clip_ratio", type=float, default=0.2)
    tr.add_argument("--entropy_coef", type=float, default=0.01)
    tr.add_argument("--value_coef", type=float, default=0.5)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--lr_scheduler", type=str, default="cosine")
    tr.add_argument("--lr_min_factor", type=float, default=0.1)
    tr.add_argument("--lr_warmup_updates", type=int, default=0)
    tr.add_argument("--max_grad_norm", type=float, default=0.5)
    tr.add_argument("--hidden_dim", type=int, default=64)
    tr.add_argument("--oracle_proj_dim", type=int, default=64)
    tr.add_argument("--minibatches", type=int, default=4)
    tr.add_argument("--sensing", type=str, default="S0")
    tr.add_argument("--max_horizon", type=int, default=60)
    tr.add_argument("--step_size", type=float, default=0.3)
    tr.add_argument("--ppo_step_scale", type=float, default=1.0)
    tr.add_argument("--success_threshold", type=float, default=0.01)
    tr.add_argument("--s1_step_penalty", type=float, default=-0.01)
    tr.add_argument("--reward_noise_std", type=float, default=0.0)
    tr.add_argument("--running_avg_window", type=int, default=100)
    tr.add_argument("--eval_interval_episodes", type=int, default=200)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--logdir", type=str, default="runs")
    tr.add_argument("--run_name", type=str, default="")
    tr.add_argument("--device", type=str, default="cpu")

    # Eval
    ev = parser.add_argument_group("Transfer evaluation")
    ev.add_argument(
        "--eval_energy_json",
        type=str,
        default="tasks/alanine_dipeptide/fourier_coefficients.json",
    )
    ev.add_argument("--eval_episodes", type=int, default=500)
    ev.add_argument("--eval_max_horizon", type=int, default=120)
    ev.add_argument("--eval_success_threshold", type=float, default=0.05)
    ev.add_argument("--eval_seed", type=int, default=999)
    ev.add_argument("--tune_lrs", action="store_true", default=True)
    ev.add_argument("--no_tune_lrs", action="store_false", dest="tune_lrs")
    ev.add_argument("--lr_tune_tasks", type=int, default=200)
    ev.add_argument("--no_eval_baselines", action="store_true")

    args = parser.parse_args()

    net_hidden = tuple(int(x) for x in args.net_hidden_dims.split(",") if x.strip())

    run_transfer(
        D=args.D,
        net_hidden_dims=net_hidden,
        meta_steps=args.meta_steps,
        meta_lr=args.meta_lr,
        meta_n_surfaces=args.meta_n_surfaces,
        meta_n_fit=args.meta_n_fit,
        meta_n_test=args.meta_n_test,
        meta_reg_lambda=args.meta_reg_lambda,
        meta_psd_eps=args.meta_psd_eps,
        diagonal=args.diagonal,
        K_energy=args.K_energy,
        visible_dim=args.visible_dim,
        amplitude_scale=args.amplitude_scale,
        freq_sparsity=args.freq_sparsity,
        n_env=args.n_env,
        refresh_surface_each_episode=args.refresh_surface_each_episode,
        n_fit_points=args.n_fit_points,
        ppo_reg_lambda=args.ppo_reg_lambda,
        ppo_psd_eps=args.ppo_psd_eps,
        lattice_rl=args.lattice_RL,
        lattice_granularity=args.lattice_granularity,
        train_steps=args.train_steps,
        hidden_dim=args.hidden_dim,
        oracle_proj_dim=args.oracle_proj_dim,
        rollout_len=args.rollout_len,
        ppo_epochs=args.ppo_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        minibatches=args.minibatches,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_min_factor=args.lr_min_factor,
        lr_warmup_updates=args.lr_warmup_updates,
        sensing=args.sensing,
        max_horizon=args.max_horizon,
        step_size=args.step_size,
        ppo_step_scale=args.ppo_step_scale,
        success_threshold=args.success_threshold,
        s1_step_penalty=args.s1_step_penalty,
        reward_noise_std=args.reward_noise_std,
        eval_energy_json=args.eval_energy_json,
        eval_episodes=args.eval_episodes,
        eval_max_horizon=args.eval_max_horizon,
        eval_success_threshold=args.eval_success_threshold,
        eval_seed=args.eval_seed,
        tune_lrs=args.tune_lrs,
        lr_tune_tasks=args.lr_tune_tasks,
        run_baselines=not args.no_eval_baselines,
        seed=args.seed,
        logdir=args.logdir,
        run_name=args.run_name,
        device=args.device,
        eval_interval_episodes=args.eval_interval_episodes,
        running_avg_window=args.running_avg_window,
    )


if __name__ == "__main__":
    main()
