"""Vectorized RL environment for alanine dipeptide torus optimization.

Mirrors the interface of tasks.spatial.spatial_env.VectorizedSpatialEnv but
operates on the 2D torus [0, 2pi)^2 with a fixed Fourier-series energy surface.
"""

from dataclasses import dataclass

import numpy as np

from .alanine_energy import (
    ALANINE_GLOBAL_MIN,
    alanine_energy,
    alanine_energy_grid,
    alanine_gradient,
    lifting_map_dim,
    lifting_map_eval,
    lifting_map_jacobian,
)
from .oracle import SpatialOracle

TWO_PI = 2.0 * np.pi


@dataclass
class AlanineEpisodeSpec:
    target_min_xy: np.ndarray
    source: np.ndarray
    shortest_dist: int
    horizon: int
    initial_objective: float


class VectorizedAlanineEnv:
    def __init__(
        self,
        K_max: int,
        oracle: SpatialOracle,
        n_env: int,
        sensing: str,
        max_horizon: int,
        seed: int,
        s1_step_penalty: float = -0.01,
        reward_noise_std: float = 0.0,
        step_size: float = 0.3,
        ppo_step_scale: float = 1.0,
        baseline_lr_gd: float | None = None,
        baseline_lr_adam: float | None = None,
        success_threshold: float = 0.01,
        compute_episode_baselines: bool = True,
    ):
        if sensing not in {"S0", "S1"}:
            raise ValueError("sensing must be S0 or S1")
        if reward_noise_std < 0.0:
            raise ValueError("reward_noise_std must be >= 0")
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0")
        if not np.isfinite(float(ppo_step_scale)) or float(ppo_step_scale) <= 0.0:
            raise ValueError("ppo_step_scale must be finite and > 0")
        if success_threshold <= 0.0:
            raise ValueError("success_threshold must be > 0")

        self.K_max = int(K_max)
        self.hidden_dim = lifting_map_dim(self.K_max)
        self.visible_dim = 2
        self.oracle = oracle
        self.n_env = int(n_env)
        self.sensing = sensing
        self.max_horizon = int(max_horizon)
        self.s1_step_penalty = float(s1_step_penalty)
        self.reward_noise_std = float(reward_noise_std)
        self.base_step_size = float(step_size)
        self.control_budget_scale = 1.0
        self.step_size = float(self.base_step_size)
        self.ppo_step_scale = float(ppo_step_scale)
        gd_lr = float(self.step_size) if baseline_lr_gd is None else float(baseline_lr_gd)
        adam_lr = float(self.step_size) if baseline_lr_adam is None else float(baseline_lr_adam)
        if gd_lr <= 0.0 or adam_lr <= 0.0:
            raise ValueError("baseline learning rates must be > 0")
        self.baseline_lr_gd = gd_lr
        self.baseline_lr_adam = adam_lr
        self.success_threshold = float(success_threshold)
        self.compute_episode_baselines = bool(compute_episode_baselines)
        self.rng = np.random.default_rng(seed)
        self.baseline_rng = np.random.default_rng(int(seed) + 101_003)
        self.objective_scale_rng = np.random.default_rng(int(seed) + 202_007)

        self.action_dim = self.visible_dim + 1  # direction (2D) + step scale (1D)
        self.oracle_token_dim = int(self.oracle.token_dim)
        self.token_feature_dim = self.oracle_token_dim + self.visible_dim

        # Per-env random torus rotation offset, sampled fresh each episode.
        # The energy surface is E((z - offset) % 2π), so the global minimum
        # appears at (ALANINE_GLOBAL_MIN + offset) % 2π in the agent's frame.
        self.torus_offset = np.zeros((self.n_env, 2), dtype=np.float32)

        # Target: the known global minimum, shifted per-env by torus_offset
        self.reference_min_xy = ALANINE_GLOBAL_MIN.astype(np.float32).copy()
        self.reference_min_xy_env = np.tile(
            self.reference_min_xy, (self.n_env, 1)
        ).astype(np.float32)

        # s* = F(target) in lifting space — recomputed per episode after rotation
        self.s_star_shared = lifting_map_eval(
            float(self.reference_min_xy[0]),
            float(self.reference_min_xy[1]),
            self.K_max,
        ).astype(np.float32)
        self.s_star = np.tile(self.s_star_shared, (self.n_env, 1)).astype(np.float32)

        # Estimate max objective for normalization
        self.max_objective = self._estimate_max_objective()
        self.max_objective_env = np.full(self.n_env, self.max_objective, dtype=np.float32)

        self.current_xy = np.zeros((self.n_env, self.visible_dim), dtype=np.float32)
        self.initial_xy = np.zeros((self.n_env, self.visible_dim), dtype=np.float32)
        self.steps = np.zeros(self.n_env, dtype=np.int32)
        self.horizons = np.ones(self.n_env, dtype=np.int32)
        self.initial_dist = np.ones(self.n_env, dtype=np.int32)
        self.initial_objective = np.ones(self.n_env, dtype=np.float32)
        self.completed_episodes = 0

        # Needed for compatibility with snapshot/synchronize functions in train.py
        self.f_type = "FOURIER"

        for index in range(self.n_env):
            self._reset_env(index)

    # ------------------------------------------------------------------
    # Lifting map and gradient helpers
    # ------------------------------------------------------------------

    def _hidden_from_z(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        return lifting_map_eval(float(z[0]), float(z[1]), self.K_max)

    def _jacobian(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        return lifting_map_jacobian(float(z[0]), float(z[1]), self.K_max)

    def _gradient_hidden(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        idx = int(env_index)
        s = self._hidden_from_z(z, env_index=idx)
        return (s - self.s_star[idx]).astype(np.float32)

    def _gradient_xy(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        """True energy gradient in visible (phi, psi) space (accounts for torus rotation)."""
        offset = self.torus_offset[int(env_index)]
        phi_raw = float(z[0] - offset[0]) % TWO_PI
        psi_raw = float(z[1] - offset[1]) % TWO_PI
        dphi, dpsi = alanine_gradient(phi_raw, psi_raw)
        return np.array([dphi, dpsi], dtype=np.float32)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective_value(self, z: np.ndarray, env_index: int = 0) -> float:
        """Energy at agent position z, accounting for per-env torus rotation."""
        offset = self.torus_offset[int(env_index)]
        phi_raw = float(z[0] - offset[0]) % TWO_PI
        psi_raw = float(z[1] - offset[1]) % TWO_PI
        return float(alanine_energy(phi_raw, psi_raw))

    def _set_max_objective(self, value: float, env_index: int = 0) -> None:
        idx = int(env_index)
        bounded = max(1.0, float(value))
        self.max_objective_env[idx] = np.float32(bounded)
        if idx == 0:
            self.max_objective = float(bounded)

    def _estimate_max_objective(self, samples: int = 2048, env_index: int = 0) -> float:
        sampled = self.objective_scale_rng.uniform(
            low=0.0, high=TWO_PI, size=(samples, self.visible_dim)
        ).astype(np.float32)
        values = np.asarray(
            [self._objective_value(point) for point in sampled], dtype=np.float32
        )
        max_value = float(np.max(values)) if values.size > 0 else 1.0
        return max(1.0, max_value)

    def _refresh_objective_scale(self, env_index: int = 0, samples: int = 512) -> float:
        value = self._estimate_max_objective(samples=samples, env_index=env_index)
        self._set_max_objective(value, env_index=env_index)
        return float(value)

    def _normalized_objective_from_raw(self, objective: float, env_index: int = 0) -> float:
        scale = max(float(self.max_objective_env[int(env_index)]), 1e-8)
        return float(np.clip(float(objective) / scale, 0.0, 1.0))

    def _normalized_objective_value(self, z: np.ndarray, env_index: int = 0) -> float:
        raw = self._objective_value(z, env_index=env_index)
        return self._normalized_objective_from_raw(raw, env_index=env_index)

    def _is_success(self, z: np.ndarray, env_index: int = 0) -> bool:
        normalized = self._normalized_objective_value(z, env_index=env_index)
        return bool(normalized <= self.success_threshold)

    def _reference_distance(self, z: np.ndarray, env_index: int = 0) -> float:
        """Torus distance to the global minimum."""
        idx = int(env_index)
        diff = z.astype(np.float64) - self.reference_min_xy_env[idx].astype(np.float64)
        # Wrap to [-pi, pi)
        diff = (diff + np.pi) % TWO_PI - np.pi
        return float(np.linalg.norm(diff))

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _token_for_state(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        if self.oracle.mode == "visible_gradient":
            grad_xy = self._gradient_xy(z, env_index=env_index).astype(np.float32)
            return self.oracle.encode_visible_gradient(grad_xy, self.rng)
        grad_h = self._gradient_hidden(z, env_index=env_index)
        return self.oracle.encode_gradient(grad_h, self.rng)

    def _obs_token_features(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        token = self._token_for_state(z, env_index=env_index)
        # Normalize position: [0, 2pi) -> [-1, 1)
        z_norm = (z / np.pi - 1.0).astype(np.float32)
        return np.concatenate([token, z_norm], axis=0).astype(np.float32)

    def _get_token_features(self) -> np.ndarray:
        token_features = np.zeros((self.n_env, self.token_feature_dim), dtype=np.float32)
        for env_index in range(self.n_env):
            token_features[env_index] = self._obs_token_features(
                self.current_xy[env_index], env_index=env_index
            )
        return token_features

    def get_obs(self) -> dict[str, np.ndarray]:
        token_features = self._get_token_features()
        objective = np.asarray(
            [
                self._objective_value(self.current_xy[idx], env_index=idx)
                for idx in range(self.n_env)
            ],
            dtype=np.float32,
        )
        z_feature = np.asarray(
            [
                self._normalized_objective_from_raw(value, env_index=idx)
                for idx, value in enumerate(objective)
            ],
            dtype=np.float32,
        )
        step_fraction = (self.steps / np.maximum(self.horizons, 1)).astype(np.float32)
        return {
            "token_features": token_features,
            "dist": z_feature,
            "step_frac": step_fraction,
        }

    # ------------------------------------------------------------------
    # Action application (torus wrapping instead of clipping)
    # ------------------------------------------------------------------

    def _wrap_to_torus(self, z: np.ndarray) -> np.ndarray:
        """Wrap coordinates to [0, 2pi)."""
        return (z % TWO_PI).astype(np.float32)

    def _apply_action(
        self,
        z: np.ndarray,
        action: np.ndarray,
        step_scale_override: float | None = None,
    ) -> np.ndarray:
        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_vector.shape[0] not in {self.visible_dim, self.visible_dim + 1}:
            raise ValueError(
                "Expected action vector with shape "
                f"({self.visible_dim},) or ({self.visible_dim + 1},), got {action_vector.shape}"
            )

        direction = action_vector[: self.visible_dim]
        if step_scale_override is not None:
            step_scale = float(np.clip(step_scale_override, 0.0, 1.0))
        elif action_vector.shape[0] == self.visible_dim + 1:
            raw_step = float(np.clip(action_vector[self.visible_dim], -20.0, 20.0))
            step_scale = float(self.ppo_step_scale * (1.0 / (1.0 + np.exp(-raw_step))))
        else:
            step_scale = 1.0

        direction64 = direction.astype(np.float64)
        max_abs = float(np.max(np.abs(direction64)))
        if not np.isfinite(max_abs) or max_abs <= 1e-12:
            delta = np.zeros(self.visible_dim, dtype=np.float32)
        else:
            scaled = direction64 / max_abs
            scaled_norm = float(np.linalg.norm(scaled))
            norm = max_abs * scaled_norm
            if norm > 1e-8 and np.isfinite(norm):
                step_magnitude = float(self.step_size) * step_scale
                delta64 = (direction64 / norm) * step_magnitude
                delta = delta64.astype(np.float32)
            else:
                delta = np.zeros(self.visible_dim, dtype=np.float32)

        updated = z + delta
        return self._wrap_to_torus(updated)

    def _apply_baseline_optimizer_step(self, z: np.ndarray, update: np.ndarray) -> np.ndarray:
        next_state = z.astype(np.float32) + update.astype(np.float32)
        return self._wrap_to_torus(next_state)

    # ------------------------------------------------------------------
    # Cosine-annealed learning rate for baselines
    # ------------------------------------------------------------------

    def _cosine_annealed_step_scale(self, step_index: int, horizon: int) -> float:
        h = max(1, int(horizon))
        if h <= 1:
            return 1.0
        t = float(np.clip(float(step_index) / float(h - 1), 0.0, 1.0))
        return float(0.5 * (1.0 + np.cos(np.pi * t)))

    def _cosine_annealed_baseline_lr(
        self,
        step_index: int,
        horizon: int,
        base_lr: float | None = None,
    ) -> float:
        lr0 = float(self.step_size) if base_lr is None else float(base_lr)
        return lr0 * self._cosine_annealed_step_scale(step_index, horizon)

    def set_baseline_learning_rates(
        self,
        *,
        gd: float | None = None,
        adam: float | None = None,
    ) -> None:
        if gd is not None:
            gd_val = float(gd)
            if gd_val <= 0.0:
                raise ValueError("gd baseline lr must be > 0")
            self.baseline_lr_gd = gd_val
        if adam is not None:
            adam_val = float(adam)
            if adam_val <= 0.0:
                raise ValueError("adam baseline lr must be > 0")
            self.baseline_lr_adam = adam_val

    def get_baseline_learning_rates(self) -> dict[str, float]:
        return {
            "gd": float(self.baseline_lr_gd),
            "adam": float(self.baseline_lr_adam),
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_xy(self) -> np.ndarray:
        return self.rng.uniform(low=0.0, high=TWO_PI, size=self.visible_dim).astype(np.float32)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def sample_episode_spec(self, env_index: int = 0, refresh_map: bool = False) -> AlanineEpisodeSpec:
        idx = int(env_index)

        # Sample a random torus rotation for this episode
        offset = self.rng.uniform(low=0.0, high=TWO_PI, size=2).astype(np.float32)
        self.torus_offset[idx] = offset

        # Target in agent's frame = (true_min + offset) % 2π
        target_min = ((ALANINE_GLOBAL_MIN + offset) % TWO_PI).astype(np.float32)
        source = self._sample_xy()

        self.reference_min_xy_env[idx] = target_min
        if idx == 0:
            self.reference_min_xy = target_min.copy()

        # Recompute s* = F(target) in lifting space for the shifted target
        self.s_star[idx] = lifting_map_eval(
            float(target_min[0]), float(target_min[1]), self.K_max
        ).astype(np.float32)

        initial_objective = self._objective_value(source, env_index=idx)
        # Proxy distance in step units
        diff = source.astype(np.float64) - target_min.astype(np.float64)
        diff = (diff + np.pi) % TWO_PI - np.pi
        shortest_dist = int(np.ceil(np.linalg.norm(diff, ord=1) / self.step_size))
        shortest_dist = max(1, shortest_dist)

        return AlanineEpisodeSpec(
            target_min_xy=target_min,
            source=source,
            shortest_dist=shortest_dist,
            horizon=self.max_horizon,
            initial_objective=float(initial_objective),
        )

    def _reset_env(self, env_index: int) -> None:
        spec = self.sample_episode_spec(env_index=env_index)
        self.current_xy[env_index] = spec.source
        self.initial_xy[env_index] = spec.source
        self.steps[env_index] = 0
        self.horizons[env_index] = spec.horizon
        self.initial_dist[env_index] = spec.shortest_dist
        self.initial_objective[env_index] = float(spec.initial_objective)

    # ------------------------------------------------------------------
    # Baseline rollouts
    # ------------------------------------------------------------------

    def rollout_baseline_final_stats(
        self,
        start_xy: np.ndarray,
        horizon: int,
        env_index: int = 0,
        base_lr: float | None = None,
    ) -> tuple[float, float]:
        return self._rollout_descent_baseline_final_stats(
            start_xy=start_xy,
            horizon=horizon,
            env_index=env_index,
            base_lr=(self.baseline_lr_gd if base_lr is None else float(base_lr)),
        )

    def rollout_adam_baseline_final_stats(
        self,
        start_xy: np.ndarray,
        horizon: int,
        env_index: int = 0,
        base_lr: float | None = None,
    ) -> tuple[float, float]:
        state = start_xy.astype(np.float32).copy()
        m = np.zeros(self.visible_dim, dtype=np.float64)
        v = np.zeros(self.visible_dim, dtype=np.float64)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        for step in range(int(horizon)):
            grad_xy = self._gradient_xy(state, env_index=env_index).astype(np.float64)
            m = beta1 * m + (1.0 - beta1) * grad_xy
            v = beta2 * v + (1.0 - beta2) * (grad_xy * grad_xy)
            t = step + 1
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            lr0 = self.baseline_lr_adam if base_lr is None else float(base_lr)
            lr_t = self._cosine_annealed_baseline_lr(step, int(horizon), base_lr=lr0)
            adam_update = (-lr_t * (m_hat / (np.sqrt(v_hat) + eps))).astype(np.float32)
            state = self._apply_baseline_optimizer_step(state, adam_update)
            if self._is_success(state, env_index=env_index):
                break
        final_objective = float(self._objective_value(state, env_index=env_index))
        final_ref_distance = self._reference_distance(state, env_index=env_index)
        return final_objective, final_ref_distance

    def _rollout_descent_baseline_final_stats(
        self,
        start_xy: np.ndarray,
        horizon: int,
        env_index: int = 0,
        base_lr: float | None = None,
    ) -> tuple[float, float]:
        state = start_xy.astype(np.float32).copy()
        for step in range(int(horizon)):
            grad_xy = self._gradient_xy(state, env_index=env_index)
            lr_t = self._cosine_annealed_baseline_lr(step, int(horizon), base_lr=base_lr)
            update = (-lr_t * grad_xy).astype(np.float32)
            state = self._apply_baseline_optimizer_step(state, update)
            if self._is_success(state, env_index=env_index):
                break
        final_objective = float(self._objective_value(state, env_index=env_index))
        final_ref_distance = self._reference_distance(state, env_index=env_index)
        return final_objective, final_ref_distance

    def rollout_baseline_final_objective(
        self,
        start_xy: np.ndarray,
        horizon: int,
        env_index: int = 0,
    ) -> float:
        final_objective, _ = self.rollout_baseline_final_stats(
            start_xy=start_xy, horizon=horizon, env_index=env_index
        )
        return final_objective

    # ------------------------------------------------------------------
    # Energy landscape grid (for plotting)
    # ------------------------------------------------------------------

    def energy_landscape_grid(
        self, resolution: int = 140, env_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return energy grid in the agent's (rotated) coordinate frame."""
        from .alanine_energy import alanine_energy_batch

        idx = int(env_index)
        offset = self.torus_offset[idx]
        res = max(20, int(resolution))
        phi_1d = np.linspace(0, TWO_PI, res, endpoint=False, dtype=np.float64)
        psi_1d = np.linspace(0, TWO_PI, res, endpoint=False, dtype=np.float64)
        phi_grid, psi_grid = np.meshgrid(phi_1d, psi_1d, indexing="xy")
        # Map agent-frame coordinates back to the original surface
        phi_raw = (phi_grid - float(offset[0])) % TWO_PI
        psi_raw = (psi_grid - float(offset[1])) % TWO_PI
        energy_grid = alanine_energy_batch(phi_raw.ravel(), psi_raw.ravel()).reshape(phi_grid.shape)
        return (
            phi_grid.astype(np.float32),
            psi_grid.astype(np.float32),
            energy_grid.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        rewards = np.zeros(self.n_env, dtype=np.float32)
        dones = np.zeros(self.n_env, dtype=np.bool_)
        infos: list[dict] = [{} for _ in range(self.n_env)]

        action_vectors = np.asarray(actions, dtype=np.float32)
        if action_vectors.ndim == 1:
            action_vectors = action_vectors.reshape(1, -1)
        if action_vectors.shape != (self.n_env, self.action_dim):
            raise ValueError(
                f"Expected actions with shape ({self.n_env}, {self.action_dim}), "
                f"got {action_vectors.shape}"
            )

        for env_index in range(self.n_env):
            z_old = self.current_xy[env_index]
            e_old_raw = self._objective_value(z_old, env_index=env_index)

            action = action_vectors[env_index]
            z_new = self._apply_action(z_old, action)
            self.current_xy[env_index] = z_new
            self.steps[env_index] += 1

            e_new_raw = self._objective_value(z_new, env_index=env_index)
            e_new_normalized = self._normalized_objective_from_raw(e_new_raw, env_index=env_index)
            success = bool(e_new_normalized <= self.success_threshold)
            timeout = int(self.steps[env_index]) >= int(self.horizons[env_index])
            episode_done = bool(success or timeout)

            if self.sensing == "S0":
                progress = float(e_old_raw - e_new_raw)
                reward = progress
                if success:
                    reward += 100.0
            else:
                reward = 100.0 if success else self.s1_step_penalty

            if self.reward_noise_std > 0.0:
                reward += float(self.rng.normal(loc=0.0, scale=self.reward_noise_std))
            rewards[env_index] = reward

            if episode_done:
                dones[env_index] = True
                self.completed_episodes += 1
                episode_len = int(self.steps[env_index])
                shortest_dist = int(self.initial_dist[env_index])
                baseline_final_objective = float("nan")
                baseline_final_ref_distance = float("nan")
                adam_baseline_final_objective = float("nan")
                adam_baseline_final_ref_distance = float("nan")
                if self.compute_episode_baselines:
                    baseline_final_objective, baseline_final_ref_distance = (
                        self.rollout_baseline_final_stats(
                            start_xy=self.initial_xy[env_index],
                            horizon=int(self.horizons[env_index]),
                            env_index=env_index,
                        )
                    )
                    adam_baseline_final_objective, adam_baseline_final_ref_distance = (
                        self.rollout_adam_baseline_final_stats(
                            start_xy=self.initial_xy[env_index],
                            horizon=int(self.horizons[env_index]),
                            env_index=env_index,
                        )
                    )
                info = {
                    "episode_done": True,
                    "success": success,
                    "episode_len": episode_len,
                    "shortest_dist": shortest_dist,
                    "initial_objective": self._normalized_objective_from_raw(
                        float(self.initial_objective[env_index]), env_index=env_index
                    ),
                    "initial_objective_raw": float(self.initial_objective[env_index]),
                    "final_objective": self._normalized_objective_from_raw(
                        float(e_new_raw), env_index=env_index
                    ),
                    "final_objective_raw": float(e_new_raw),
                    "final_ref_distance": self._reference_distance(z_new, env_index=env_index),
                    "baseline_final_objective": self._normalized_objective_from_raw(
                        float(baseline_final_objective), env_index=env_index
                    ),
                    "baseline_final_objective_raw": float(baseline_final_objective),
                    "baseline_final_ref_distance": float(baseline_final_ref_distance),
                    "adam_baseline_final_objective": self._normalized_objective_from_raw(
                        float(adam_baseline_final_objective), env_index=env_index
                    ),
                    "adam_baseline_final_objective_raw": float(adam_baseline_final_objective),
                    "adam_baseline_final_ref_distance": float(adam_baseline_final_ref_distance),
                }
                if success and shortest_dist > 0:
                    info["stretch"] = float(episode_len / shortest_dist)
                infos[env_index] = info
                self._reset_env(env_index)

        next_obs = self.get_obs()
        return next_obs, rewards, dones, infos
