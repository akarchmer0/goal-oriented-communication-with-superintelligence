from dataclasses import dataclass

import numpy as np

from .oracle import SpatialOracle


@dataclass
class SpatialEpisodeSpec:
    target_min_xy: np.ndarray
    source: np.ndarray
    shortest_dist: int
    horizon: int
    initial_objective: float


class VectorizedSpatialEnv:
    def __init__(
        self,
        hidden_dim: int,
        visible_dim: int,
        coord_limit: int,
        oracle: SpatialOracle,
        n_env: int,
        sensing: str,
        max_horizon: int,
        seed: int,
        s1_step_penalty: float = -0.01,
        reward_noise_std: float = 0.0,
        step_size: float = 1.0,
        ppo_step_scale: float = 1.0,
        baseline_lr_gd: float | None = None,
        baseline_lr_adam: float | None = None,
        success_threshold: float = 0.01,
        basis_complexity: int = 3,
        freq_sparsity: int = 0,
        f_type: str = "FOURIER",
        refresh_map_each_episode: bool = False,
        compute_episode_baselines: bool = True,
        fixed_start_target: bool = False,
        lattice_rl: bool = False,
        lattice_granularity: int = 20,
    ):
        if sensing not in {"S0", "S1"}:
            raise ValueError("sensing must be S0 or S1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if visible_dim < 2:
            raise ValueError("visible_dim must be >= 2")
        if coord_limit <= 0:
            raise ValueError("coord_limit must be > 0")
        if reward_noise_std < 0.0:
            raise ValueError("reward_noise_std must be >= 0")
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0")
        if not np.isfinite(float(ppo_step_scale)) or float(ppo_step_scale) <= 0.0:
            raise ValueError("ppo_step_scale must be finite and > 0")
        if success_threshold <= 0.0:
            raise ValueError("success_threshold must be > 0")
        if basis_complexity < 1:
            raise ValueError("basis_complexity must be >= 1")
        map_family = str(f_type).upper()
        if map_family not in {"FOURIER", "MLP"}:
            raise ValueError("f_type must be one of {'FOURIER', 'MLP'}")

        self.hidden_dim = int(hidden_dim)
        self.visible_dim = int(visible_dim)
        self.coord_limit = float(coord_limit)
        self.oracle = oracle
        self.n_env = int(n_env)
        self.sensing = sensing
        self.max_horizon = int(max_horizon)
        self.s1_step_penalty = float(s1_step_penalty)
        self.reward_noise_std = float(reward_noise_std)
        self.base_step_size = float(step_size)
        self.control_budget_scale = float(np.sqrt(float(self.visible_dim) / 2.0))
        self.step_size = float(self.base_step_size * self.control_budget_scale)
        self.ppo_step_scale = float(ppo_step_scale)
        gd_lr = float(self.step_size) if baseline_lr_gd is None else float(baseline_lr_gd)
        adam_lr = float(self.step_size) if baseline_lr_adam is None else float(baseline_lr_adam)
        if gd_lr <= 0.0 or adam_lr <= 0.0:
            raise ValueError("baseline learning rates must be > 0")
        self.baseline_lr_gd = gd_lr
        self.baseline_lr_adam = adam_lr
        self.success_threshold = float(success_threshold)
        self.basis_complexity = int(basis_complexity)
        # freq_sparsity r: max nonzero components per frequency vector.
        # 0 means dense (all d components nonzero, original behavior).
        r = int(freq_sparsity)
        if r < 0:
            raise ValueError("freq_sparsity must be >= 0")
        if r > int(visible_dim):
            raise ValueError("freq_sparsity must be <= visible_dim")
        self.freq_sparsity = r if r > 0 else int(visible_dim)
        self.f_type = map_family
        self.refresh_map_each_episode = bool(refresh_map_each_episode)
        self.compute_episode_baselines = bool(compute_episode_baselines)
        self.fixed_start_target = bool(fixed_start_target)
        self.rng = np.random.default_rng(seed)
        self.baseline_rng = np.random.default_rng(int(seed) + 101_003)
        self.objective_scale_rng = np.random.default_rng(int(seed) + 202_007)
        self._fixed_target_min_xy: np.ndarray | None = None
        self._fixed_source_xy: np.ndarray | None = None

        # Lattice RL mode: discrete actions over adjacent lattice nodes.
        self.lattice_rl = bool(lattice_rl)
        self.lattice_granularity = int(lattice_granularity)
        if self.lattice_rl:
            if self.lattice_granularity < 2:
                raise ValueError("lattice_granularity must be >= 2")
            # Action space: 2 * visible_dim discrete actions (±1 along each axis).
            self.action_dim = 2 * self.visible_dim
            # Precompute lattice spacing.
            self.lattice_spacing = float(2.0 * self.coord_limit / (self.lattice_granularity - 1))
            # Lattice node coordinates along each dimension.
            self.lattice_ticks = np.linspace(
                -self.coord_limit, self.coord_limit,
                num=self.lattice_granularity, dtype=np.float32,
            )
        else:
            # The policy emits a continuous (direction, step_raw) vector.
            self.action_dim = self.visible_dim + 1
        self.oracle_token_dim = int(self.oracle.token_dim)
        # Include z coordinates so policy can learn (g_t, z_t) -> action.
        self.token_feature_dim = self.oracle_token_dim + self.visible_dim

        self.reference_min_xy = np.zeros(self.visible_dim, dtype=np.float32)
        self.reference_min_xy_env = np.zeros((self.n_env, self.visible_dim), dtype=np.float32)
        self.max_objective_env = np.ones(self.n_env, dtype=np.float32)
        self.max_objective = 1.0
        self._init_map_storage()
        self._initialize_maps()

        self.current_xy = np.zeros((self.n_env, self.visible_dim), dtype=np.float32)
        self.initial_xy = np.zeros((self.n_env, self.visible_dim), dtype=np.float32)
        self.steps = np.zeros(self.n_env, dtype=np.int32)
        self.horizons = np.ones(self.n_env, dtype=np.int32)
        self.initial_dist = np.ones(self.n_env, dtype=np.int32)
        self.initial_objective = np.ones(self.n_env, dtype=np.float32)
        self.completed_episodes = 0

        for index in range(self.n_env):
            self._reset_env(index)

    def _init_map_storage(self) -> None:
        self.linear_w = np.zeros((self.n_env, self.hidden_dim, self.visible_dim), dtype=np.float32)
        self.sin_w = np.zeros((self.n_env, self.hidden_dim, self.visible_dim), dtype=np.float32)
        self.cos_w = np.zeros((self.n_env, self.hidden_dim, self.visible_dim), dtype=np.float32)
        self.sin_phase = np.zeros((self.n_env, self.hidden_dim), dtype=np.float32)
        self.cos_phase = np.zeros((self.n_env, self.hidden_dim), dtype=np.float32)
        self.sin_amp = np.zeros((self.n_env, self.hidden_dim), dtype=np.float32)
        self.cos_amp = np.zeros((self.n_env, self.hidden_dim), dtype=np.float32)
        mlp_width = max(8, self.basis_complexity * 8)
        self.mlp_width = int(mlp_width)
        self.mlp_w1 = np.zeros((self.n_env, self.mlp_width, self.visible_dim), dtype=np.float32)
        self.mlp_b1 = np.zeros((self.n_env, self.mlp_width), dtype=np.float32)
        self.mlp_w2 = np.zeros((self.n_env, self.hidden_dim, self.mlp_width), dtype=np.float32)
        self.mlp_b2 = np.zeros((self.n_env, self.hidden_dim), dtype=np.float32)
        self.s_star = np.zeros((self.n_env, self.hidden_dim), dtype=np.float32)

    def _sample_sparse_freq(self, shape: tuple[int, int]) -> np.ndarray:
        """Sample frequency vectors with at most `self.freq_sparsity` nonzero components.

        Each row gets exactly min(freq_sparsity, visible_dim) randomly chosen
        active dimensions; the remaining dimensions are zero.  Active components
        are drawn from {±1, …, ±K} where K = basis_complexity.
        """
        n_rows, d = shape
        r = min(self.freq_sparsity, d)
        freq_low = 1
        freq_high = self.basis_complexity + 1

        w = np.zeros((n_rows, d), dtype=np.float32)
        for i in range(n_rows):
            active = self.rng.choice(d, size=r, replace=False)
            magnitudes = self.rng.integers(
                freq_low, freq_high, size=r, dtype=np.int32
            ).astype(np.float32)
            signs = self.rng.choice(
                np.asarray([-1.0, 1.0], dtype=np.float32), size=r
            )
            w[i, active] = magnitudes * signs
        return w

    def _sample_map_parameters(self) -> dict[str, np.ndarray]:
        if self.f_type == "FOURIER":
            # F(z) = A z + a*sin(Wz + b) + c*cos(Vz + d)
            linear_w = self.rng.normal(
                loc=0.0,
                scale=0.18,
                size=(self.hidden_dim, self.visible_dim),
            ).astype(np.float32)

            freq_shape = (self.hidden_dim, self.visible_dim)
            sin_w = self._sample_sparse_freq(freq_shape)
            cos_w = self._sample_sparse_freq(freq_shape)

            sin_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=self.hidden_dim).astype(np.float32)
            cos_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=self.hidden_dim).astype(np.float32)
            sin_amp = self.rng.uniform(0.8, 1.3, size=self.hidden_dim).astype(np.float32)
            cos_amp = self.rng.uniform(0.6, 1.1, size=self.hidden_dim).astype(np.float32)
            return {
                "linear_w": linear_w,
                "sin_w": sin_w,
                "cos_w": cos_w,
                "sin_phase": sin_phase,
                "cos_phase": cos_phase,
                "sin_amp": sin_amp,
                "cos_amp": cos_amp,
            }

        # F(z) = A z + W2 * tanh(W1 z + b1) + b2
        linear_w = self.rng.normal(
            loc=0.0,
            scale=0.20,
            size=(self.hidden_dim, self.visible_dim),
        ).astype(np.float32)
        mlp_w1 = (
            self.rng.normal(size=(self.mlp_width, self.visible_dim)).astype(np.float32)
            / np.sqrt(float(self.visible_dim))
        )
        mlp_b1 = self.rng.normal(loc=0.0, scale=0.7, size=self.mlp_width).astype(np.float32)
        mlp_w2 = (
            self.rng.normal(size=(self.hidden_dim, self.mlp_width)).astype(np.float32)
            / np.sqrt(float(self.mlp_width))
        )
        mlp_b2 = self.rng.normal(loc=0.0, scale=0.5, size=self.hidden_dim).astype(np.float32)
        return {
            "linear_w": linear_w,
            "mlp_w1": mlp_w1,
            "mlp_b1": mlp_b1,
            "mlp_w2": mlp_w2,
            "mlp_b2": mlp_b2,
        }

    def _set_map_for_env(self, env_index: int, params: dict[str, np.ndarray]) -> None:
        idx = int(env_index)
        self.linear_w[idx] = params["linear_w"]
        if self.f_type == "FOURIER":
            self.sin_w[idx] = params["sin_w"]
            self.cos_w[idx] = params["cos_w"]
            self.sin_phase[idx] = params["sin_phase"]
            self.cos_phase[idx] = params["cos_phase"]
            self.sin_amp[idx] = params["sin_amp"]
            self.cos_amp[idx] = params["cos_amp"]
            self.mlp_w1[idx].fill(0.0)
            self.mlp_b1[idx].fill(0.0)
            self.mlp_w2[idx].fill(0.0)
            self.mlp_b2[idx].fill(0.0)
        else:
            self.mlp_w1[idx] = params["mlp_w1"]
            self.mlp_b1[idx] = params["mlp_b1"]
            self.mlp_w2[idx] = params["mlp_w2"]
            self.mlp_b2[idx] = params["mlp_b2"]
            self.sin_w[idx].fill(0.0)
            self.cos_w[idx].fill(0.0)
            self.sin_phase[idx].fill(0.0)
            self.cos_phase[idx].fill(0.0)
            self.sin_amp[idx].fill(0.0)
            self.cos_amp[idx].fill(0.0)
        self.s_star[idx] = self._hidden_from_z(self.reference_min_xy_env[idx], env_index=idx)

    def _refresh_map_for_env(self, env_index: int) -> None:
        self._set_map_for_env(env_index, self._sample_map_parameters())

    def _initialize_maps(self) -> None:
        if self.refresh_map_each_episode:
            for env_index in range(self.n_env):
                self._refresh_map_for_env(env_index)
            return

        shared = self._sample_map_parameters()
        for env_index in range(self.n_env):
            self._set_map_for_env(env_index, shared)

    def _hidden_from_z(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        idx = int(env_index)
        z = z.astype(np.float32)
        linear = np.matmul(self.linear_w[idx], z)
        if self.f_type == "FOURIER":
            sin_arg = np.matmul(self.sin_w[idx], z) + self.sin_phase[idx]
            cos_arg = np.matmul(self.cos_w[idx], z) + self.cos_phase[idx]
            sin_term = self.sin_amp[idx] * np.sin(sin_arg)
            cos_term = self.cos_amp[idx] * np.cos(cos_arg)
            return (linear + sin_term + cos_term).astype(np.float32)

        hidden_pre = np.matmul(self.mlp_w1[idx], z) + self.mlp_b1[idx]
        hidden = np.tanh(hidden_pre)
        nonlinear = np.matmul(self.mlp_w2[idx], hidden) + self.mlp_b2[idx]
        return (linear + nonlinear).astype(np.float32)

    def _jacobian(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        idx = int(env_index)
        z = z.astype(np.float32)
        if self.f_type == "FOURIER":
            sin_arg = np.matmul(self.sin_w[idx], z) + self.sin_phase[idx]
            cos_arg = np.matmul(self.cos_w[idx], z) + self.cos_phase[idx]
            sin_coeff = (self.sin_amp[idx] * np.cos(sin_arg))[:, None]
            cos_coeff = (-self.cos_amp[idx] * np.sin(cos_arg))[:, None]
            return (
                self.linear_w[idx] + sin_coeff * self.sin_w[idx] + cos_coeff * self.cos_w[idx]
            ).astype(np.float32)

        hidden_pre = np.matmul(self.mlp_w1[idx], z) + self.mlp_b1[idx]
        sech2 = (1.0 - np.tanh(hidden_pre) ** 2).astype(np.float32)
        weighted_w1 = sech2[:, None] * self.mlp_w1[idx]
        mlp_jac = np.matmul(self.mlp_w2[idx], weighted_w1).astype(np.float32)
        return (self.linear_w[idx] + mlp_jac).astype(np.float32)

    def _gradient_hidden(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        idx = int(env_index)
        s = self._hidden_from_z(z, env_index=idx)
        return (s - self.s_star[idx]).astype(np.float32)

    def _gradient_xy(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        grad_h = self._gradient_hidden(z, env_index=env_index)
        jacobian = self._jacobian(z, env_index=env_index)
        return np.matmul(jacobian.T, grad_h).astype(np.float32)

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
        # Baseline optimizers use a standard learning-rate schedule:
        # eta_t = eta_0 * 0.5 * (1 + cos(pi * t / (H - 1))).
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

    def _apply_baseline_optimizer_step(self, z: np.ndarray, update: np.ndarray) -> np.ndarray:
        next_state = z.astype(np.float32) + update.astype(np.float32)
        return np.clip(next_state, -self.coord_limit, self.coord_limit).astype(np.float32)

    def _objective_value(self, z: np.ndarray, env_index: int = 0) -> float:
        grad_h = self._gradient_hidden(z, env_index=env_index)
        return 0.5 * float(np.dot(grad_h, grad_h))

    def _set_max_objective(self, value: float, env_index: int = 0) -> None:
        idx = int(env_index)
        bounded = max(1.0, float(value))
        self.max_objective_env[idx] = np.float32(bounded)
        if idx == 0:
            self.max_objective = float(bounded)

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
        idx = int(env_index)
        return float(np.linalg.norm(z.astype(np.float32) - self.reference_min_xy_env[idx]))

    def _token_for_state(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        if self.oracle.mode == "visible_gradient":
            grad_xy = self._gradient_xy(z, env_index=env_index).astype(np.float32)
            return self.oracle.encode_visible_gradient(grad_xy, self.rng)
        grad_h = self._gradient_hidden(z, env_index=env_index)
        return self.oracle.encode_gradient(grad_h, self.rng)

    def _obs_token_features(self, z: np.ndarray, env_index: int = 0) -> np.ndarray:
        token = self._token_for_state(z, env_index=env_index)
        z_norm = np.clip(z / self.coord_limit, -1.0, 1.0).astype(np.float32)
        return np.concatenate([token, z_norm], axis=0).astype(np.float32)

    def _snap_to_lattice(self, z: np.ndarray) -> np.ndarray:
        """Snap a position to the nearest lattice node."""
        indices = np.round(
            (z + self.coord_limit) / self.lattice_spacing
        ).astype(np.int32)
        indices = np.clip(indices, 0, self.lattice_granularity - 1)
        return self.lattice_ticks[indices].astype(np.float32)

    def _apply_lattice_action(self, z: np.ndarray, action_index: int) -> np.ndarray:
        """Move to an adjacent lattice node given a discrete action index.

        Actions are enumerated as:
            0 => +step along dim 0
            1 => -step along dim 0
            2 => +step along dim 1
            3 => -step along dim 1
            ...
            2*d-2 => +step along dim d-1
            2*d-1 => -step along dim d-1
        """
        action_index = int(action_index)
        dim = action_index // 2
        sign = 1.0 if (action_index % 2 == 0) else -1.0
        delta = np.zeros(self.visible_dim, dtype=np.float32)
        delta[dim] = float(sign * self.lattice_spacing)
        updated = z + delta
        clipped = np.clip(updated, -self.coord_limit, self.coord_limit).astype(np.float32)
        return self._snap_to_lattice(clipped)

    def _apply_action(
        self,
        z: np.ndarray,
        action: np.ndarray | int,
        step_scale_override: float | None = None,
    ) -> np.ndarray:
        if self.lattice_rl:
            return self._apply_lattice_action(z, int(action))
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
                # Always scale policy movement by step_size; step_scale only modulates it.
                step_magnitude = float(self.step_size) * step_scale
                delta64 = (direction64 / norm) * step_magnitude
                delta = delta64.astype(np.float32)
            else:
                delta = np.zeros(self.visible_dim, dtype=np.float32)
        updated = z + delta
        return np.clip(updated, -self.coord_limit, self.coord_limit).astype(np.float32)

    def _sample_xy(self) -> np.ndarray:
        xy = self.rng.uniform(
            low=-self.coord_limit,
            high=self.coord_limit,
            size=self.visible_dim,
        ).astype(np.float32)
        return xy

    def _estimate_max_objective(self, samples: int = 2048, env_index: int = 0) -> float:
        sampled = self.objective_scale_rng.uniform(
            low=-self.coord_limit,
            high=self.coord_limit,
            size=(samples, self.visible_dim),
        ).astype(np.float32)
        values = np.asarray(
            [self._objective_value(point, env_index=env_index) for point in sampled],
            dtype=np.float32,
        )
        max_value = float(np.max(values)) if values.size > 0 else 1.0
        return max(1.0, max_value)

    def energy_landscape_grid(
        self, resolution: int = 140, env_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.visible_dim != 2:
            raise ValueError("energy_landscape_grid is only defined when visible_dim == 2")
        resolution = max(20, int(resolution))
        axis = np.linspace(-self.coord_limit, self.coord_limit, num=resolution, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
        stacked = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        energy = np.asarray(
            [self._objective_value(point, env_index=env_index) for point in stacked],
            dtype=np.float32,
        )
        energy = energy.reshape(grid_x.shape)
        return grid_x, grid_y, energy

    def sample_episode_spec(self, env_index: int = 0, refresh_map: bool = False) -> SpatialEpisodeSpec:
        idx = int(env_index)
        if refresh_map:
            self._refresh_map_for_env(idx)

        if self.fixed_start_target:
            if self._fixed_target_min_xy is None:
                self._fixed_target_min_xy = self._sample_xy().astype(np.float32)
            if self._fixed_source_xy is None:
                source_xy = self._sample_xy().astype(np.float32)
                if self.lattice_rl:
                    source_xy = self._snap_to_lattice(source_xy)
                self._fixed_source_xy = source_xy
            target_min = self._fixed_target_min_xy.copy()
            source = self._fixed_source_xy.copy()
        else:
            target_min = self._sample_xy()
            source = self._sample_xy()
            if self.lattice_rl:
                source = self._snap_to_lattice(source)

        self.reference_min_xy_env[idx] = target_min
        if idx == 0:
            self.reference_min_xy = target_min.copy()
        self.s_star[idx] = self._hidden_from_z(target_min, env_index=idx)
        self._refresh_objective_scale(env_index=idx)
        initial_objective = self._objective_value(source, env_index=idx)
        # Proxy "distance" in step units to support existing metrics pipeline.
        shortest_dist = int(np.ceil(np.linalg.norm(source, ord=1) / self.step_size))
        shortest_dist = max(1, shortest_dist)
        return SpatialEpisodeSpec(
            target_min_xy=target_min,
            source=source,
            shortest_dist=shortest_dist,
            horizon=self.max_horizon,
            initial_objective=float(initial_objective),
        )

    def _reset_env(self, env_index: int) -> None:
        spec = self.sample_episode_spec(
            env_index=env_index,
            refresh_map=self.refresh_map_each_episode,
        )
        self.current_xy[env_index] = spec.source
        self.initial_xy[env_index] = spec.source
        self.steps[env_index] = 0
        self.horizons[env_index] = spec.horizon
        self.initial_dist[env_index] = spec.shortest_dist
        self.initial_objective[env_index] = float(spec.initial_objective)

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
            lr_t = self._cosine_annealed_baseline_lr(
                step,
                int(horizon),
                base_lr=lr0,
            )
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
            start_xy=start_xy,
            horizon=horizon,
            env_index=env_index,
        )
        return final_objective

    def _get_token_features(self) -> np.ndarray:
        token_features = np.zeros((self.n_env, self.token_feature_dim), dtype=np.float32)
        for env_index in range(self.n_env):
            token_features[env_index] = self._obs_token_features(
                self.current_xy[env_index],
                env_index=env_index,
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
            [self._normalized_objective_from_raw(value, env_index=idx) for idx, value in enumerate(objective)],
            dtype=np.float32,
        )
        step_fraction = (self.steps / np.maximum(self.horizons, 1)).astype(np.float32)
        return {
            "token_features": token_features,
            "dist": z_feature,
            "step_frac": step_fraction,
        }

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        rewards = np.zeros(self.n_env, dtype=np.float32)
        dones = np.zeros(self.n_env, dtype=np.bool_)
        infos: list[dict] = [{} for _ in range(self.n_env)]

        if self.lattice_rl:
            # Discrete actions: 1-D array of integer action indices, shape (n_env,).
            action_indices = np.asarray(actions, dtype=np.int64).reshape(-1)
            if action_indices.shape[0] != self.n_env:
                raise ValueError(
                    f"Expected {self.n_env} discrete actions, got {action_indices.shape[0]}"
                )
        else:
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

            if self.lattice_rl:
                action = action_indices[env_index]
            else:
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
                        float(self.initial_objective[env_index]),
                        env_index=env_index,
                    ),
                    "initial_objective_raw": float(self.initial_objective[env_index]),
                    "final_objective": self._normalized_objective_from_raw(
                        float(e_new_raw),
                        env_index=env_index,
                    ),
                    "final_objective_raw": float(e_new_raw),
                    "final_ref_distance": self._reference_distance(z_new, env_index=env_index),
                    "baseline_final_objective": self._normalized_objective_from_raw(
                        float(baseline_final_objective),
                        env_index=env_index,
                    ),
                    "baseline_final_objective_raw": float(baseline_final_objective),
                    "baseline_final_ref_distance": float(baseline_final_ref_distance),
                    "adam_baseline_final_objective": self._normalized_objective_from_raw(
                        float(adam_baseline_final_objective),
                        env_index=env_index,
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
