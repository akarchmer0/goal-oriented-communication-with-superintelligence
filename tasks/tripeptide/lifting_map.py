"""Fourier lifting map F: T^d -> R^D for the d-dimensional torus.

F(z) = [cos(m1.z), cos(m2.z), ..., sin(m1.z), sin(m2.z), ...]
for all integer multi-indices m with 0 < ||m||_inf <= K_map.

D = 2 * N_freq where N_freq = (2*K_map+1)^d - 1.
"""

from __future__ import annotations

import numpy as np


def enumerate_frequency_indices(d: int, K: int) -> np.ndarray:
    """Enumerate all integer multi-indices m with |m_i| <= K, excluding m=0.

    Returns array of shape (N_freq, d) sorted lexicographically.
    """
    side = 2 * K + 1
    total = side ** d
    # Build all combos via np.indices
    indices = np.indices([side] * d).reshape(d, -1).T - K  # (total, d)
    # Remove the zero vector
    nonzero_mask = np.any(indices != 0, axis=1)
    return indices[nonzero_mask].astype(np.int64)


class LiftingMap:
    """Fourier lifting map F: T^d -> R^D.

    The basis is [cos(m1.z), ..., cos(mN.z), sin(m1.z), ..., sin(mN.z)]
    giving D = 2 * N_freq components.
    """

    def __init__(self, d: int, K_map: int):
        self.d = int(d)
        self.K_map = int(K_map)
        self.frequency_matrix = enumerate_frequency_indices(self.d, self.K_map)  # (N_freq, d)
        self.N_freq = self.frequency_matrix.shape[0]
        self.D = 2 * self.N_freq
        # Precompute float64 frequency matrix for fast eval
        self._freq_f64 = self.frequency_matrix.astype(np.float64)

    def eval(self, z: np.ndarray) -> np.ndarray:
        """Evaluate F(z) -> R^D. z shape (d,)."""
        z64 = np.asarray(z, dtype=np.float64).ravel()
        phases = self._freq_f64 @ z64  # (N_freq,)
        return np.concatenate([np.cos(phases), np.sin(phases)]).astype(np.float32)

    def eval_batch(self, z_batch: np.ndarray) -> np.ndarray:
        """Evaluate F(z) for multiple points. z_batch shape (B, d) -> (B, D)."""
        z64 = np.asarray(z_batch, dtype=np.float64)
        if z64.ndim == 1:
            z64 = z64.reshape(1, -1)
        phases = z64 @ self._freq_f64.T  # (B, N_freq)
        return np.concatenate([np.cos(phases), np.sin(phases)], axis=1).astype(np.float32)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Jacobian dF/dz, shape (D, d)."""
        z64 = np.asarray(z, dtype=np.float64).ravel()
        phases = self._freq_f64 @ z64  # (N_freq,)
        sin_phases = np.sin(phases)  # (N_freq,)
        cos_phases = np.cos(phases)  # (N_freq,)
        # d/dz_j cos(m.z) = -m_j * sin(m.z)
        # d/dz_j sin(m.z) =  m_j * cos(m.z)
        jac_cos = -sin_phases[:, None] * self._freq_f64  # (N_freq, d)
        jac_sin = cos_phases[:, None] * self._freq_f64   # (N_freq, d)
        return np.concatenate([jac_cos, jac_sin], axis=0).astype(np.float32)

    def energy_as_linear(self, energy_surface) -> np.ndarray:
        """Return D-dimensional vector c such that E(z) ≈ c^T F(z) + c0.

        Maps the energy's Fourier coefficients to this lifting map's ordering.
        If K_map >= K_energy, the representation is exact.
        """
        c = np.zeros(self.D, dtype=np.float64)
        for i, m in enumerate(self.frequency_matrix):
            m_tuple = tuple(int(x) for x in m)
            # cos part: index i
            c[i] = energy_surface.cos_coeffs.get(m_tuple, 0.0)
            # sin part: index N_freq + i
            c[self.N_freq + i] = energy_surface.sin_coeffs.get(m_tuple, 0.0)
        return c.astype(np.float64)
