"""Vectorized training environment with frozen lifting net + quadratic oracle.

Each of the n_env parallel environments has its own random Fourier energy
surface and its own PSD-constrained quadratic Q(s)=s^T A s + b^T s + c
fitted in the lifted space of F.

Oracle token = nabla_s Q = 2As + b  (D-dimensional).
Interface matches VectorizedAlanineEnv so the same PPO training loop works.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tasks.alanine_dipeptide.energy import EnergySurface
from tasks.alanine_dipeptide.lifting_map import enumerate_frequency_indices

from .lifting_net import LiftingNet
from .meta_train import _filter_frequencies

TWO_PI = 2.0 * np.pi


# ──────────────────────────────────────────────────────────────────────
# Fast random surface generation
# ──────────────────────────────────────────────────────────────────────


def fast_random_surface(
    d: int,
    K_energy: int,
    seed: int,
    n_search: int = 100_000,
    amplitude_scale: float = 5.0,
    freq_sparsity: int = 0,
) -> EnergySurface:
    """Generate a random Fourier energy surface on T^d with random coefficients.

    Coefficients decay as amplitude_scale / (1 + ||m||) for smooth multi-basin
    landscapes.  The approximate global minimum is found via random search.

    If freq_sparsity > 0, only frequency vectors with at most that many nonzero
    components are used (axis-aligned-ish modes), yielding simpler surfaces.
    """
    rng = np.random.default_rng(seed)
    freqs = enumerate_frequency_indices(d, K_energy)
    freqs = _filter_frequencies(freqs, freq_sparsity)
    n_freq = freqs.shape[0]
    freq_f64 = freqs.astype(np.float64)

    norms = np.linalg.norm(freq_f64, axis=1)
    decay = amplitude_scale / (1.0 + norms)
    a = rng.normal(0.0, 1.0, n_freq) * decay
    b = rng.normal(0.0, 1.0, n_freq) * decay
    c0 = 0.0

    cos_coeffs: dict[tuple[int, ...], float] = {}
    sin_coeffs: dict[tuple[int, ...], float] = {}
    for i in range(n_freq):
        m_tuple = tuple(int(x) for x in freqs[i])
        if abs(a[i]) > 1e-15:
            cos_coeffs[m_tuple] = float(a[i])
        if abs(b[i]) > 1e-15:
            sin_coeffs[m_tuple] = float(b[i])

    search_pts = rng.uniform(0.0, TWO_PI, size=(n_search, d)).astype(np.float64)
    phases = search_pts @ freq_f64.T
    energies = c0 + np.dot(np.cos(phases), a) + np.dot(np.sin(phases), b)
    min_idx = int(np.argmin(energies))

    return EnergySurface(
        d=d,
        K_energy=K_energy,
        c0=c0,
        cos_coeffs=cos_coeffs,
        sin_coeffs=sin_coeffs,
        global_min=search_pts[min_idx],
        global_min_energy=float(energies[min_idx]),
    )


# ──────────────────────────────────────────────────────────────────────
# Quadratic fitting (numpy, for frozen F during PPO)
# ──────────────────────────────────────────────────────────────────────


def _fit_lstsq_numpy(
    Phi: np.ndarray, y: np.ndarray, reg_lambda: float
) -> np.ndarray:
    """Regularised least squares with column normalization for stability.

    Column-normalizes Phi so regularization acts uniformly across features.
    """
    p = Phi.shape[1]
    col_norms = np.linalg.norm(Phi, axis=0, keepdims=True).clip(min=1e-8)  # (1, p)
    Phi_n = Phi / col_norms
    PhiTPhi = Phi_n.T @ Phi_n + reg_lambda * np.eye(p, dtype=np.float32)
    PhiTy = Phi_n.T @ y
    theta_n = np.linalg.solve(PhiTPhi, PhiTy)
    return theta_n / col_norms.ravel()


def fit_quadratic_for_surface(
    lifting_net: LiftingNet,
    surface: EnergySurface,
    n_fit: int = 1024,
    reg_lambda: float = 1e-4,
    psd_eps: float = 1e-6,
    seed: int = 0,
    diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit quadratic in F-space for one surface.

    If diagonal=False: returns (A, b, c) where A is (D, D) PSD matrix.
    If diagonal=True:  returns (a, b, c) where a is (D,) non-negative vector.
    """
    rng = np.random.default_rng(seed)
    d = surface.d
    D = lifting_net.lifting_dim

    z_fit = rng.uniform(0.0, TWO_PI, size=(n_fit, d)).astype(np.float32)
    y_fit = surface.energy_batch(z_fit).astype(np.float32)

    # Min-max normalize energies to [0, 1]
    y_min = float(y_fit.min())
    y_max = float(y_fit.max())
    y_range = max(y_max - y_min, 1e-8)
    y_fit = (y_fit - y_min) / y_range

    with torch.no_grad():
        s_fit = lifting_net(torch.as_tensor(z_fit)).numpy()  # (n_fit, D)

    if diagonal:
        # Features: [s², s, 1]
        ones = np.ones((n_fit, 1), dtype=np.float32)
        Phi = np.concatenate([s_fit * s_fit, s_fit, ones], axis=-1)
        theta = _fit_lstsq_numpy(Phi, y_fit, reg_lambda)
        a_vec = theta[:D]
        b_vec = theta[D : 2 * D]
        c_val = float(theta[-1])
        a_vec = np.maximum(a_vec, psd_eps).astype(np.float32)
        return a_vec, b_vec.astype(np.float32), c_val
    else:
        # Full quadratic features
        idx_i, idx_j = np.triu_indices(D)
        n_quad = len(idx_i)
        quad = s_fit[:, idx_i] * s_fit[:, idx_j]
        is_offdiag = (idx_i != idx_j).astype(np.float32)
        quad *= 1.0 + is_offdiag
        ones = np.ones((n_fit, 1), dtype=np.float32)
        Phi = np.concatenate([quad, s_fit, ones], axis=-1)
        theta = _fit_lstsq_numpy(Phi, y_fit, reg_lambda)
        A_upper = theta[:n_quad]
        b_vec = theta[n_quad : n_quad + D]
        c_val = float(theta[-1])
        A = np.zeros((D, D), dtype=np.float32)
        A[idx_i, idx_j] = A_upper
        A[idx_j, idx_i] = A_upper
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvalues_clamped = np.maximum(eigenvalues, psd_eps)
        A_psd = (eigenvectors * eigenvalues_clamped) @ eigenvectors.T
        return A_psd.astype(np.float32), b_vec.astype(np.float32), c_val


# ──────────────────────────────────────────────────────────────────────
# Episode spec
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TransferTrainEpisodeSpec:
    target_min_xy: np.ndarray
    source: np.ndarray
    shortest_dist: int
    horizon: int
    initial_objective: float


# ──────────────────────────────────────────────────────────────────────
# Vectorized training environment
# ──────────────────────────────────────────────────────────────────────


class VectorizedTransferTrainEnv:
    """PPO training env with frozen lifting net + fitted quadratic oracle.

    Interface matches VectorizedAlanineEnv so the same PPO loop works.
    """

    def __init__(
        self,
        lifting_net: LiftingNet,
        n_env: int,
        K_energy: int,
        max_horizon: int,
        seed: int,
        sensing: str = "S0",
        s1_step_penalty: float = -0.01,
        reward_noise_std: float = 0.0,
        step_size: float = 0.3,
        ppo_step_scale: float = 1.0,
        success_threshold: float = 0.01,
        amplitude_scale: float = 5.0,
        refresh_surface_each_episode: bool = False,
        n_fit_points: int = 1024,
        reg_lambda: float = 1e-4,
        psd_eps: float = 1e-6,
        diagonal: bool = False,
        freq_sparsity: int = 0,
        lattice_rl: bool = False,
        lattice_granularity: int = 20,
    ):
        if sensing not in {"S0", "S1"}:
            raise ValueError("sensing must be S0 or S1")

        self.lifting_net = lifting_net
        self.visible_dim = lifting_net.visible_dim
        self.lifting_dim = lifting_net.lifting_dim
        self.n_env = int(n_env)
        self.K_energy = int(K_energy)
        self.sensing = sensing
        self.max_horizon = int(max_horizon)
        self.s1_step_penalty = float(s1_step_penalty)
        self.reward_noise_std = float(reward_noise_std)
        self.base_step_size = float(step_size)
        self.step_size = float(step_size)
        self.ppo_step_scale = float(ppo_step_scale)
        self.success_threshold = float(success_threshold)
        self.amplitude_scale = float(amplitude_scale)
        self.refresh_surface_each_episode = bool(refresh_surface_each_episode)
        self.n_fit_points = int(n_fit_points)
        self.reg_lambda = float(reg_lambda)
        self.psd_eps = float(psd_eps)
        self.diagonal = bool(diagonal)
        self.freq_sparsity = int(freq_sparsity)

        self.rng = np.random.default_rng(seed)
        self.surface_rng_base_seed = int(seed) + 50_000

        D = self.lifting_dim
        d = self.visible_dim

        # Lattice RL mode
        self.lattice_rl = bool(lattice_rl)
        self.lattice_granularity = int(lattice_granularity)
        if self.lattice_rl:
            if self.lattice_granularity < 2:
                raise ValueError("lattice_granularity must be >= 2")
            self.action_dim = 2 * d
            self.lattice_spacing = float(TWO_PI / self.lattice_granularity)
            self.lattice_ticks = np.linspace(
                0.0, TWO_PI, num=self.lattice_granularity,
                endpoint=False, dtype=np.float32,
            )
        else:
            self.action_dim = d + 1
        self.oracle_token_dim = D          # gradient 2As+b / 2a⊙s+b is D-dimensional
        self.token_feature_dim = D + D + d  # [oracle_grad, s_star, z_norm]

        self.f_type = "FOURIER"
        self.baseline_lr_gd = float(step_size)
        self.baseline_lr_adam = float(step_size)

        # Per-env quadratic coefficients
        self.surfaces: list[EnergySurface] = []
        if self.diagonal:
            self.a_env = np.zeros((n_env, D), dtype=np.float32)
        else:
            self.A_env = np.zeros((n_env, D, D), dtype=np.float32)
        self.b_env = np.zeros((n_env, D), dtype=np.float32)
        self.c_env = np.zeros(n_env, dtype=np.float32)
        self.s_star_env = np.zeros((n_env, D), dtype=np.float32)
        self.q_min_env = np.zeros(n_env, dtype=np.float32)

        self.reference_min_xy_env = np.zeros((n_env, d), dtype=np.float32)
        self.global_min_energy_env = np.zeros(n_env, dtype=np.float64)
        self.max_objective_env = np.ones(n_env, dtype=np.float32)
        self.max_surrogate_env = np.ones(n_env, dtype=np.float32)

        self.current_xy = np.zeros((n_env, d), dtype=np.float32)
        self.initial_xy = np.zeros((n_env, d), dtype=np.float32)
        self.steps = np.zeros(n_env, dtype=np.int32)
        self.horizons = np.ones(n_env, dtype=np.int32)
        self.initial_dist = np.ones(n_env, dtype=np.int32)
        self.initial_objective = np.ones(n_env, dtype=np.float32)
        self.completed_episodes = 0
        self._surface_counter = 0

        print(
            f"  Generating {n_env} random surfaces + fitting quadratics "
            f"(K={K_energy}, D={D})..."
        )
        for i in range(n_env):
            self._generate_surface_for_env(i)
        print(f"  All {n_env} surfaces initialised.")

        for i in range(n_env):
            self._reset_env(i)

    # ------------------------------------------------------------------
    # Surface generation
    # ------------------------------------------------------------------

    def _generate_surface_for_env(self, env_index: int) -> None:
        idx = int(env_index)
        surface_seed = self.surface_rng_base_seed + self._surface_counter
        self._surface_counter += 1

        surface = fast_random_surface(
            d=self.visible_dim,
            K_energy=self.K_energy,
            seed=surface_seed,
            amplitude_scale=self.amplitude_scale,
            freq_sparsity=self.freq_sparsity,
        )

        coeff_first, b, c = fit_quadratic_for_surface(
            self.lifting_net,
            surface,
            n_fit=self.n_fit_points,
            reg_lambda=self.reg_lambda,
            psd_eps=self.psd_eps,
            seed=surface_seed + 1_000_000,
            diagonal=self.diagonal,
        )

        if idx < len(self.surfaces):
            self.surfaces[idx] = surface
        else:
            self.surfaces.append(surface)

        if self.diagonal:
            self.a_env[idx] = coeff_first  # (D,) diagonal vector
        else:
            self.A_env[idx] = coeff_first  # (D, D) matrix
        self.b_env[idx] = b
        self.c_env[idx] = float(c)

        # Surrogate minimum
        if self.diagonal:
            # s*_i = -b_i / (2 a_i)
            a = coeff_first
            s_star = -b / (2.0 * np.maximum(a, 1e-12))
            self.q_min_env[idx] = float(np.dot(a, s_star ** 2) + np.dot(b, s_star) + c)
        else:
            A = coeff_first
            try:
                s_star = -0.5 * np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                s_star = np.zeros(self.lifting_dim, dtype=np.float32)
            self.q_min_env[idx] = float(s_star @ A @ s_star + b @ s_star + c)
        self.s_star_env[idx] = s_star.astype(np.float32)

        self.reference_min_xy_env[idx] = np.asarray(
            surface.global_min, dtype=np.float32
        ).ravel()
        self.global_min_energy_env[idx] = float(surface.global_min_energy)
        self.max_objective_env[idx] = max(
            1.0, self._estimate_max_energy(env_index=idx)
        )
        self.max_surrogate_env[idx] = max(
            1.0, self._estimate_max_surrogate(env_index=idx)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lift_batch(self, z_batch: np.ndarray) -> np.ndarray:
        """Lift (n, d) points through frozen F.  Returns (n, D)."""
        with torch.no_grad():
            return self.lifting_net(torch.as_tensor(z_batch, dtype=torch.float32)).numpy()

    def _energy_value(self, z: np.ndarray, env_index: int = 0) -> float:
        return float(self.surfaces[int(env_index)].energy(z))

    def _estimate_max_energy(self, env_index: int = 0, samples: int = 2048) -> float:
        idx = int(env_index)
        pts = self.rng.uniform(0.0, TWO_PI, size=(samples, self.visible_dim)).astype(np.float32)
        e = self.surfaces[idx].energy_batch(pts)
        return max(1.0, float(np.max(e) - self.global_min_energy_env[idx]))

    def _estimate_max_surrogate(self, env_index: int = 0, samples: int = 2048) -> float:
        idx = int(env_index)
        pts = self.rng.uniform(0.0, TWO_PI, size=(samples, self.visible_dim)).astype(np.float32)
        s = self._lift_batch(pts)
        if self.diagonal:
            q = np.sum(self.a_env[idx] * s * s, axis=-1) + s @ self.b_env[idx] + self.c_env[idx]
        else:
            As = np.einsum("ij,nj->ni", self.A_env[idx], s)
            q = np.einsum("ni,ni->n", s, As) + s @ self.b_env[idx] + self.c_env[idx]
        return max(1.0, float(np.max(q) - self.q_min_env[idx]))

    def _normalized_objective_from_raw(
        self, energy: float, env_index: int = 0
    ) -> float:
        idx = int(env_index)
        scale = max(float(self.max_objective_env[idx]), 1e-8)
        return float(
            np.clip(
                (float(energy) - self.global_min_energy_env[idx]) / scale, 0.0, 1.0
            )
        )

    def _is_success(self, z: np.ndarray, env_index: int = 0) -> bool:
        e = self._energy_value(z, env_index)
        return bool(self._normalized_objective_from_raw(e, env_index) <= self.success_threshold)

    def _reference_distance(self, z: np.ndarray, env_index: int = 0) -> float:
        diff = z.astype(np.float64) - self.reference_min_xy_env[int(env_index)].astype(np.float64)
        diff = (diff + np.pi) % TWO_PI - np.pi
        return float(np.linalg.norm(diff))

    # ------------------------------------------------------------------
    # Observation  (batched for efficiency)
    # ------------------------------------------------------------------

    def get_obs(self) -> dict[str, np.ndarray]:
        s_all = self._lift_batch(self.current_xy)  # (n_env, D)

        if self.diagonal:
            # Oracle gradient:  2 * a_k ⊙ s_k + b_k
            oracle_grads = (2.0 * self.a_env * s_all + self.b_env).astype(np.float32)
            # Surrogate value: sum(a_k * s_k²) + b_k · s_k + c_k
            quad_term = np.sum(self.a_env * s_all * s_all, axis=-1)
            lin_term = np.sum(self.b_env * s_all, axis=-1)
        else:
            # Oracle gradient:  2 * A_k @ s_k + b_k
            As = np.einsum("nij,nj->ni", self.A_env, s_all)  # (n_env, D)
            oracle_grads = (2.0 * As + self.b_env).astype(np.float32)
            quad_term = np.einsum("ni,ni->n", s_all, As)
            lin_term = np.einsum("ni,ni->n", s_all, self.b_env)

        surr_vals = quad_term + lin_term + self.c_env  # (n_env,)

        z_norm = (self.current_xy / np.pi - 1.0).astype(np.float32)
        token_features = np.concatenate([oracle_grads, self.s_star_env, z_norm], axis=-1)

        # Normalised surrogate as dist feature
        z_feature = np.clip(
            (surr_vals - self.q_min_env) / np.maximum(self.max_surrogate_env, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)

        step_fraction = (self.steps / np.maximum(self.horizons, 1)).astype(np.float32)

        return {
            "token_features": token_features,
            "dist": z_feature,
            "step_frac": step_fraction,
        }

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _wrap_to_torus(self, z: np.ndarray) -> np.ndarray:
        return (z % TWO_PI).astype(np.float32)

    def _snap_to_lattice(self, z: np.ndarray) -> np.ndarray:
        """Snap a position to the nearest lattice node on [0, 2pi)."""
        z_wrapped = z % TWO_PI
        indices = np.round(z_wrapped / self.lattice_spacing).astype(np.int32)
        indices = indices % self.lattice_granularity
        return self.lattice_ticks[indices].astype(np.float32)

    def _apply_lattice_action(self, z: np.ndarray, action_index: int) -> np.ndarray:
        """Move to an adjacent lattice node given a discrete action index.

        Actions: 0=>+step dim0, 1=>-step dim0, 2=>+step dim1, 3=>-step dim1, ...
        """
        action_index = int(action_index)
        dim = action_index // 2
        sign = 1.0 if (action_index % 2 == 0) else -1.0
        delta = np.zeros(self.visible_dim, dtype=np.float32)
        delta[dim] = float(sign * self.lattice_spacing)
        updated = z + delta
        return self._snap_to_lattice(self._wrap_to_torus(updated))

    def _apply_action(
        self,
        z: np.ndarray,
        action: np.ndarray | int,
        step_scale_override: float | None = None,
    ) -> np.ndarray:
        if self.lattice_rl:
            return self._apply_lattice_action(z, int(action))

        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        direction = action_vector[: self.visible_dim]
        if step_scale_override is not None:
            step_scale = float(np.clip(step_scale_override, 0.0, 1.0))
        elif action_vector.shape[0] == self.visible_dim + 1:
            raw_step = float(np.clip(action_vector[self.visible_dim], -20.0, 20.0))
            step_scale = float(
                self.ppo_step_scale * (1.0 / (1.0 + np.exp(-raw_step)))
            )
        else:
            step_scale = 1.0

        direction64 = direction.astype(np.float64)
        max_abs = float(np.max(np.abs(direction64)))
        if not np.isfinite(max_abs) or max_abs <= 1e-12:
            return z.astype(np.float32)
        norm = float(np.linalg.norm(direction64))
        if norm > 1e-8 and np.isfinite(norm):
            delta = (direction64 / norm * self.step_size * step_scale).astype(
                np.float32
            )
        else:
            return z.astype(np.float32)
        return self._wrap_to_torus(z + delta)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def _sample_xy(self) -> np.ndarray:
        return self.rng.uniform(0.0, TWO_PI, size=self.visible_dim).astype(np.float32)

    def sample_episode_spec(
        self, env_index: int = 0, **kwargs
    ) -> TransferTrainEpisodeSpec:
        idx = int(env_index)
        target_min = self.reference_min_xy_env[idx].copy()
        source = self._sample_xy()
        diff = source.astype(np.float64) - target_min.astype(np.float64)
        diff = (diff + np.pi) % TWO_PI - np.pi
        shortest_dist = max(
            1, int(np.ceil(np.linalg.norm(diff, ord=1) / self.step_size))
        )
        initial_objective = self._energy_value(source, env_index=idx)
        return TransferTrainEpisodeSpec(
            target_min_xy=target_min,
            source=source,
            shortest_dist=shortest_dist,
            horizon=self.max_horizon,
            initial_objective=float(initial_objective),
        )

    def _reset_env(self, env_index: int) -> None:
        idx = int(env_index)
        if self.refresh_surface_each_episode and self.completed_episodes > 0:
            self._generate_surface_for_env(idx)
        spec = self.sample_episode_spec(env_index=idx)
        source = spec.source
        if self.lattice_rl:
            source = self._snap_to_lattice(source)
        self.current_xy[idx] = source
        self.initial_xy[idx] = source
        self.steps[idx] = 0
        self.horizons[idx] = spec.horizon
        self.initial_dist[idx] = spec.shortest_dist
        self.initial_objective[idx] = float(spec.initial_objective)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        rewards = np.zeros(self.n_env, dtype=np.float32)
        dones = np.zeros(self.n_env, dtype=np.bool_)
        infos: list[dict] = [{} for _ in range(self.n_env)]

        if self.lattice_rl:
            action_indices = np.asarray(actions, dtype=np.int64).reshape(-1)
        else:
            action_vectors = np.asarray(actions, dtype=np.float32)
            if action_vectors.ndim == 1:
                action_vectors = action_vectors.reshape(1, -1)

        for env_index in range(self.n_env):
            z_old = self.current_xy[env_index]
            energy_old = self._energy_value(z_old, env_index=env_index)

            if self.lattice_rl:
                action = action_indices[env_index]
            else:
                action = action_vectors[env_index]
            z_new = self._apply_action(z_old, action)
            self.current_xy[env_index] = z_new
            self.steps[env_index] += 1

            energy_new = self._energy_value(z_new, env_index=env_index)
            success = self._is_success(z_new, env_index=env_index)
            timeout = int(self.steps[env_index]) >= int(self.horizons[env_index])
            episode_done = bool(success or timeout)

            if self.sensing == "S0":
                reward = float(energy_old - energy_new)
            else:
                reward = self.s1_step_penalty

            if self.reward_noise_std > 0.0:
                reward += float(self.rng.normal(0.0, self.reward_noise_std))
            rewards[env_index] = reward

            if episode_done:
                dones[env_index] = True
                self.completed_episodes += 1
                infos[env_index] = {
                    "episode_done": True,
                    "success": success,
                    "episode_len": int(self.steps[env_index]),
                    "shortest_dist": int(self.initial_dist[env_index]),
                    "final_objective": self._normalized_objective_from_raw(
                        float(energy_new), env_index=env_index
                    ),
                    "final_objective_raw": float(energy_new),
                    "final_ref_distance": self._reference_distance(
                        z_new, env_index=env_index
                    ),
                }
                self._reset_env(env_index)

        next_obs = self.get_obs()
        return next_obs, rewards, dones, infos
