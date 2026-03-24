"""Transfer evaluation environment: learned lifting net + quadratic oracle + alanine.

Uses the meta-learned lifting net F and a PSD-constrained quadratic Q fitted
on the alanine surface.  Oracle gradient nabla_s Q = 2As + b (D-dimensional).

Rewards and success are based on the actual energy E(z), not the surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .lifting_net import LiftingNet

TWO_PI = 2.0 * np.pi


@dataclass
class TransferEpisodeSpec:
    target_min_xy: np.ndarray
    source: np.ndarray
    shortest_dist: int
    horizon: int
    initial_objective: float


class TransferEvalEnv:
    """Evaluates a trained policy on the alanine dipeptide surface."""

    def __init__(
        self,
        lifting_net: LiftingNet,
        A: np.ndarray,
        b: np.ndarray,
        c: float,
        energy_surface,
        step_size: float = 0.3,
        ppo_step_scale: float = 1.0,
        max_horizon: int = 120,
        success_threshold: float = 0.05,
        seed: int = 999,
        sensing: str = "S0",
        s1_step_penalty: float = -0.01,
        baseline_lr_gd: float = 0.3,
        baseline_lr_adam: float = 0.3,
        diagonal: bool = False,
        lattice_rl: bool = False,
        lattice_granularity: int = 20,
    ):
        self.lifting_net = lifting_net
        self.energy_surface = energy_surface
        self.visible_dim = lifting_net.visible_dim
        self.lifting_dim = lifting_net.lifting_dim
        self.step_size = float(step_size)
        self.ppo_step_scale = float(ppo_step_scale)
        self.max_horizon = int(max_horizon)
        self.success_threshold = float(success_threshold)
        self.sensing = sensing
        self.s1_step_penalty = float(s1_step_penalty)
        self.rng = np.random.default_rng(seed)
        self.baseline_lr_gd = float(baseline_lr_gd)
        self.baseline_lr_adam = float(baseline_lr_adam)
        self.diagonal = bool(diagonal)

        # Lattice RL mode
        self.lattice_rl = bool(lattice_rl)
        self.lattice_granularity = int(lattice_granularity)
        if self.lattice_rl:
            if self.lattice_granularity < 2:
                raise ValueError("lattice_granularity must be >= 2")
            self.action_dim = 2 * self.visible_dim
            self.lattice_spacing = float(TWO_PI / self.lattice_granularity)
            self.lattice_ticks = np.linspace(
                0.0, TWO_PI, num=self.lattice_granularity,
                endpoint=False, dtype=np.float32,
            )
        else:
            self.action_dim = self.visible_dim + 1
        self.oracle_token_dim = self.lifting_dim
        self.token_feature_dim = self.oracle_token_dim + self.lifting_dim + self.visible_dim
        self.n_env = 1

        # Quadratic coefficients — A is (D,) for diagonal, (D,D) for full
        self.b_vec = np.asarray(b, dtype=np.float32).ravel()
        self.c_val = float(c)

        if self.diagonal:
            self.a_vec = np.asarray(A, dtype=np.float32).ravel()
            # s*_i = -b_i / (2 a_i)
            self.s_star = -self.b_vec / (2.0 * np.maximum(self.a_vec, 1e-12))
            self.q_min = float(
                np.dot(self.a_vec, self.s_star ** 2)
                + np.dot(self.b_vec, self.s_star)
                + self.c_val
            )
        else:
            self.A = np.asarray(A, dtype=np.float32)
            try:
                self.s_star = -0.5 * np.linalg.solve(self.A, self.b_vec)
            except np.linalg.LinAlgError:
                self.s_star = np.zeros(self.lifting_dim, dtype=np.float32)
            self.q_min = float(
                self.s_star @ self.A @ self.s_star
                + self.b_vec @ self.s_star
                + self.c_val
            )

        # Reference minimum
        self.reference_min_xy = np.asarray(
            energy_surface.global_min, dtype=np.float32
        ).ravel()
        self.global_min_energy = float(energy_surface.global_min_energy)
        self.max_objective = self._estimate_max_objective()
        self.max_surrogate = self._estimate_max_surrogate()
        self.completed_episodes = 0

    # ------------------------------------------------------------------
    # Lifting / oracle
    # ------------------------------------------------------------------

    def _lift(self, z: np.ndarray) -> np.ndarray:
        z_t = torch.as_tensor(z.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            s_t = self.lifting_net(z_t)
        return s_t.squeeze(0).numpy()

    def _oracle_gradient(self, z: np.ndarray) -> np.ndarray:
        s = self._lift(z)
        if self.diagonal:
            return (2.0 * self.a_vec * s + self.b_vec).astype(np.float32)
        return (2.0 * (self.A @ s) + self.b_vec).astype(np.float32)

    def _surrogate_value(self, z: np.ndarray) -> float:
        s = self._lift(z)
        if self.diagonal:
            return float(np.dot(self.a_vec, s * s) + np.dot(self.b_vec, s) + self.c_val)
        return float(s @ self.A @ s + self.b_vec @ s + self.c_val)

    def _energy_value(self, z: np.ndarray) -> float:
        return float(self.energy_surface.energy(z))

    def _gradient_xy(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(self.energy_surface.gradient(z), dtype=np.float32)

    # ------------------------------------------------------------------
    # Normalization / success
    # ------------------------------------------------------------------

    def _estimate_max_objective(self, samples: int = 4096) -> float:
        pts = self.rng.uniform(0.0, TWO_PI, size=(samples, self.visible_dim)).astype(
            np.float32
        )
        e = self.energy_surface.energy_batch(pts)
        return max(1.0, float(np.max(e) - self.global_min_energy))

    def _estimate_max_surrogate(self, samples: int = 4096) -> float:
        pts = self.rng.uniform(0.0, TWO_PI, size=(samples, self.visible_dim)).astype(
            np.float32
        )
        vals = np.array([self._surrogate_value(pt) for pt in pts], dtype=np.float32)
        return max(1.0, float(np.max(vals) - self.q_min))

    def _normalized_objective_value(
        self, z: np.ndarray, env_index: int = 0
    ) -> float:
        """Normalised surrogate for the dist observation."""
        raw = self._surrogate_value(z)
        return float(
            np.clip((raw - self.q_min) / max(self.max_surrogate, 1e-8), 0.0, 1.0)
        )

    def _is_success(self, z: np.ndarray, env_index: int = 0) -> bool:
        e = self._energy_value(z)
        normalised = (e - self.global_min_energy) / max(self.max_objective, 1e-8)
        return bool(normalised <= self.success_threshold)

    def _reference_distance(self, z: np.ndarray) -> float:
        diff = z.astype(np.float64) - self.reference_min_xy.astype(np.float64)
        diff = (diff + np.pi) % TWO_PI - np.pi
        return float(np.linalg.norm(diff))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _obs_token_features(
        self, z: np.ndarray, env_index: int = 0
    ) -> np.ndarray:
        token = self._oracle_gradient(z)
        z_norm = (z / np.pi - 1.0).astype(np.float32)
        return np.concatenate([token, self.s_star, z_norm], axis=0).astype(np.float32)

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

    def _apply_baseline_optimizer_step(
        self, z: np.ndarray, update: np.ndarray
    ) -> np.ndarray:
        return self._wrap_to_torus(z.astype(np.float32) + update.astype(np.float32))

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def sample_episode_spec(
        self, env_index: int = 0, **kwargs
    ) -> TransferEpisodeSpec:
        target_min = self.reference_min_xy.copy()
        source = self.rng.uniform(0.0, TWO_PI, size=self.visible_dim).astype(
            np.float32
        )
        diff = source.astype(np.float64) - target_min.astype(np.float64)
        diff = (diff + np.pi) % TWO_PI - np.pi
        shortest_dist = max(
            1, int(np.ceil(np.linalg.norm(diff, ord=1) / self.step_size))
        )
        initial_objective = self._energy_value(source)
        return TransferEpisodeSpec(
            target_min_xy=target_min,
            source=source,
            shortest_dist=shortest_dist,
            horizon=self.max_horizon,
            initial_objective=float(initial_objective),
        )

    # ------------------------------------------------------------------
    # Baseline rollouts (on actual energy surface)
    # ------------------------------------------------------------------

    def _cosine_annealed_lr(
        self, step: int, horizon: int, base_lr: float
    ) -> float:
        h = max(1, int(horizon))
        if h <= 1:
            return base_lr
        t = float(np.clip(float(step) / float(h - 1), 0.0, 1.0))
        return base_lr * float(0.5 * (1.0 + np.cos(np.pi * t)))

    def rollout_gd_baseline(
        self, start_xy, horizon, lr=None
    ) -> tuple[float, float, bool]:
        base_lr = self.baseline_lr_gd if lr is None else float(lr)
        state = start_xy.astype(np.float32).copy()
        for step in range(int(horizon)):
            grad_xy = self._gradient_xy(state)
            lr_t = self._cosine_annealed_lr(step, horizon, base_lr)
            state = self._apply_baseline_optimizer_step(
                state, (-lr_t * grad_xy).astype(np.float32)
            )
            if self._is_success(state):
                return self._energy_value(state), self._reference_distance(state), True
        return self._energy_value(state), self._reference_distance(state), False

    def rollout_adam_baseline(
        self, start_xy, horizon, lr=None
    ) -> tuple[float, float, bool]:
        base_lr = self.baseline_lr_adam if lr is None else float(lr)
        state = start_xy.astype(np.float32).copy()
        d = self.visible_dim
        m = np.zeros(d, dtype=np.float64)
        v = np.zeros(d, dtype=np.float64)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for step in range(int(horizon)):
            grad_xy = self._gradient_xy(state).astype(np.float64)
            m = beta1 * m + (1.0 - beta1) * grad_xy
            v = beta2 * v + (1.0 - beta2) * (grad_xy * grad_xy)
            t = step + 1
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            lr_t = self._cosine_annealed_lr(step, int(horizon), base_lr)
            update = (-lr_t * (m_hat / (np.sqrt(v_hat) + eps))).astype(np.float32)
            state = self._apply_baseline_optimizer_step(state, update)
            if self._is_success(state):
                return self._energy_value(state), self._reference_distance(state), True
        return self._energy_value(state), self._reference_distance(state), False

    def rollout_surrogate_gd_baseline(
        self, start_xy, horizon, lr=None
    ) -> tuple[float, float, bool]:
        """GD using dQ/dz = (dF/dz)^T (2A F(z) + b)."""
        base_lr = self.baseline_lr_gd if lr is None else float(lr)
        state = start_xy.astype(np.float32).copy()
        if self.diagonal:
            a_t = torch.as_tensor(self.a_vec, dtype=torch.float32)
        else:
            A_t = torch.as_tensor(self.A, dtype=torch.float32)
        b_t = torch.as_tensor(self.b_vec, dtype=torch.float32)
        for step in range(int(horizon)):
            z_t = torch.as_tensor(
                state.reshape(1, -1), dtype=torch.float32
            ).requires_grad_(True)
            s_t = self.lifting_net(z_t)  # (1, D)
            if self.diagonal:
                Q = (
                    (a_t * s_t * s_t).sum()
                    + (b_t * s_t).sum()
                    + self.c_val
                )
            else:
                Q = (
                    (s_t @ A_t @ s_t.T).squeeze()
                    + (b_t @ s_t.T).squeeze()
                    + self.c_val
                )
            Q.backward()
            grad_z = z_t.grad.squeeze().detach().numpy().astype(np.float32)
            lr_t = self._cosine_annealed_lr(step, horizon, base_lr)
            state = self._apply_baseline_optimizer_step(
                state, (-lr_t * grad_z).astype(np.float32)
            )
            if self._is_success(state):
                return self._energy_value(state), self._reference_distance(state), True
        return self._energy_value(state), self._reference_distance(state), False

    # ------------------------------------------------------------------
    # Quadratic fit quality diagnostic
    # ------------------------------------------------------------------

    def fit_quadratic_quality(
        self, n_samples: int = 50_000, seed: int = 42
    ) -> dict:
        """Evaluate how well the fitted quadratic approximates the energy.

        Since the quadratic is fitted on min-max normalized energies,
        we normalize the true energies the same way before computing R².
        """
        rng = np.random.default_rng(seed)
        z = rng.uniform(0.0, TWO_PI, size=(n_samples, self.visible_dim)).astype(
            np.float32
        )
        energies = self.energy_surface.energy_batch(z).astype(np.float64)
        preds = np.array(
            [self._surrogate_value(zi) for zi in z], dtype=np.float64
        )
        # Normalize energies to same scale as surrogate (min-max to [0,1])
        e_min, e_max = float(energies.min()), float(energies.max())
        e_range = max(e_max - e_min, 1e-8)
        energies_norm = (energies - e_min) / e_range
        residual = energies_norm - preds
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((energies_norm - np.mean(energies_norm)) ** 2))
        return {
            "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "mae": float(np.mean(np.abs(residual))),
            "mse": float(np.mean(residual ** 2)),
            "max_error": float(np.max(np.abs(residual))),
        }
