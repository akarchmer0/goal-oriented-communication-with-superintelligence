"""Tripeptide energy surface from Fourier coefficients.

Loads E(z) = c0 + sum_m [a_m cos(m.z) + b_m sin(m.z)] from a JSON file,
where z = (phi1, psi1, phi2, psi2) is on the 4D torus [0, 2pi)^4.

Also provides a synthetic fallback for testing when the JSON is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TWO_PI = 2.0 * np.pi


class EnergySurface:
    """Fourier-series energy surface on the d-dimensional torus."""

    def __init__(
        self,
        d: int,
        K_energy: int,
        c0: float,
        cos_coeffs: dict[tuple[int, ...], float],
        sin_coeffs: dict[tuple[int, ...], float],
        global_min: np.ndarray,
        global_min_energy: float,
        minima: list[dict] | None = None,
        angle_names: list[str] | None = None,
    ):
        self.d = int(d)
        self.K_energy = int(K_energy)
        self.c0 = float(c0)
        self.cos_coeffs = dict(cos_coeffs)
        self.sin_coeffs = dict(sin_coeffs)
        self.global_min = np.asarray(global_min, dtype=np.float64).ravel()
        self.global_min_energy = float(global_min_energy)
        self.minima = minima or []
        self.angle_names = angle_names or [f"theta_{i}" for i in range(self.d)]

        # Build vectorized coefficient arrays for fast evaluation
        all_freqs: set[tuple[int, ...]] = set(self.cos_coeffs.keys()) | set(self.sin_coeffs.keys())
        self._n_terms = len(all_freqs)
        if self._n_terms == 0:
            self._freq_matrix = np.zeros((0, self.d), dtype=np.float64)
            self._a_vec = np.zeros(0, dtype=np.float64)
            self._b_vec = np.zeros(0, dtype=np.float64)
        else:
            freq_list = sorted(all_freqs)
            self._freq_matrix = np.array(freq_list, dtype=np.float64)  # (N, d)
            self._a_vec = np.array(
                [self.cos_coeffs.get(m, 0.0) for m in freq_list], dtype=np.float64
            )
            self._b_vec = np.array(
                [self.sin_coeffs.get(m, 0.0) for m in freq_list], dtype=np.float64
            )

    def energy(self, z: np.ndarray) -> float:
        """Evaluate E(z) at a single point on the torus."""
        z64 = np.asarray(z, dtype=np.float64).ravel()
        if self._n_terms == 0:
            return self.c0
        phases = self._freq_matrix @ z64  # (N,)
        return float(self.c0 + np.dot(self._a_vec, np.cos(phases)) + np.dot(self._b_vec, np.sin(phases)))

    def energy_batch(self, z_batch: np.ndarray) -> np.ndarray:
        """Evaluate E(z) at multiple points. z_batch shape (B, d)."""
        z64 = np.asarray(z_batch, dtype=np.float64)
        if z64.ndim == 1:
            z64 = z64.reshape(1, -1)
        if self._n_terms == 0:
            return np.full(z64.shape[0], self.c0, dtype=np.float64)
        phases = z64 @ self._freq_matrix.T  # (B, N)
        return self.c0 + np.cos(phases) @ self._a_vec + np.sin(phases) @ self._b_vec

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Analytical gradient dE/dz, shape (d,)."""
        z64 = np.asarray(z, dtype=np.float64).ravel()
        if self._n_terms == 0:
            return np.zeros(self.d, dtype=np.float64)
        phases = self._freq_matrix @ z64  # (N,)
        # dE/dz_j = sum_m [-a_m * m_j * sin(m.z) + b_m * m_j * cos(m.z)]
        sin_phases = np.sin(phases)  # (N,)
        cos_phases = np.cos(phases)  # (N,)
        # Weighted: -a * sin + b * cos, shape (N,)
        weights = -self._a_vec * sin_phases + self._b_vec * cos_phases
        # Gradient: freq_matrix.T @ weights, shape (d,)
        return (self._freq_matrix.T @ weights).astype(np.float64)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "EnergySurface":
        """Load from the standard JSON format."""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        d = int(data["d"])
        K_energy = int(data["K_energy"])
        c0 = float(data["c0"])
        angle_names = data.get("angle_names", None)

        cos_coeffs: dict[tuple[int, ...], float] = {}
        sin_coeffs: dict[tuple[int, ...], float] = {}
        for key, val in data["coefficients"]["cos"].items():
            m = tuple(int(x) for x in key.split(","))
            cos_coeffs[m] = float(val)
        for key, val in data["coefficients"]["sin"].items():
            m = tuple(int(x) for x in key.split(","))
            sin_coeffs[m] = float(val)

        minima = data.get("minima", [])
        global_min_idx = int(data.get("global_min_index", 0))
        global_min_entry = minima[global_min_idx]
        global_min = np.array(global_min_entry["angles_rad"], dtype=np.float64)
        global_min_energy = float(global_min_entry["energy_kjmol"])

        return cls(
            d=d,
            K_energy=K_energy,
            c0=c0,
            cos_coeffs=cos_coeffs,
            sin_coeffs=sin_coeffs,
            global_min=global_min,
            global_min_energy=global_min_energy,
            minima=minima,
            angle_names=angle_names,
        )


def generate_synthetic_surface(
    d: int = 4,
    K: int = 3,
    n_minima: int = 8,
    seed: int = 42,
) -> EnergySurface:
    """Generate a synthetic multi-basin Fourier energy surface on the d-torus.

    Places Gaussian wells at random locations, evaluates on a grid,
    fits Fourier coefficients, and identifies minima.
    """
    rng = np.random.default_rng(seed)

    # Place wells
    well_centers = rng.uniform(0.0, TWO_PI, size=(n_minima, d))
    well_depths = rng.uniform(5.0, 25.0, size=n_minima)  # kJ/mol
    well_widths = rng.uniform(0.4, 1.2, size=n_minima)

    # Evaluate on a grid
    res = 24  # per dimension — kept small for d=4 (24^4 = 331776)
    axes = [np.linspace(0, TWO_PI, res, endpoint=False) for _ in range(d)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)  # (res,...,res, d)
    flat_grid = grid.reshape(-1, d)  # (N_grid, d)

    # Energy from Gaussian wells (periodic via cosine distance)
    energy_flat = np.zeros(flat_grid.shape[0], dtype=np.float64)
    for i in range(n_minima):
        diff = flat_grid - well_centers[i]
        # Periodic distance on torus
        cos_dist_sq = np.sum(1.0 - np.cos(diff), axis=-1)  # in [0, 2d]
        energy_flat -= well_depths[i] * np.exp(-cos_dist_sq / (2.0 * well_widths[i] ** 2))

    # Add a mild barrier term
    energy_flat += 5.0  # shift so most values are positive

    # Fit Fourier coefficients via DFT-like projection
    # Enumerate frequencies: all m with |m_i| <= K, excluding m=0
    freq_list: list[tuple[int, ...]] = []
    ranges = [range(-K, K + 1) for _ in range(d)]
    for combo in np.ndindex(*[2 * K + 1 for _ in range(d)]):
        m = tuple(c - K for c in combo)
        if all(x == 0 for x in m):
            continue
        freq_list.append(m)

    freq_matrix = np.array(freq_list, dtype=np.float64)  # (N_freq, d)
    phases = flat_grid @ freq_matrix.T  # (N_grid, N_freq)
    N_grid = flat_grid.shape[0]

    c0 = float(np.mean(energy_flat))
    centered = energy_flat - c0
    a_vec = (2.0 / N_grid) * (np.cos(phases).T @ centered)
    b_vec = (2.0 / N_grid) * (np.sin(phases).T @ centered)

    cos_coeffs = {m: float(a_vec[i]) for i, m in enumerate(freq_list) if abs(a_vec[i]) > 1e-12}
    sin_coeffs = {m: float(b_vec[i]) for i, m in enumerate(freq_list) if abs(b_vec[i]) > 1e-12}

    # Find global minimum by evaluating on a finer grid (sample-based)
    n_search = 100_000
    search_pts = rng.uniform(0.0, TWO_PI, size=(n_search, d))
    search_phases = search_pts @ freq_matrix.T
    search_energy = c0 + search_phases @ np.array([cos_coeffs.get(m, 0.0) for m in freq_list]) * np.cos(search_pts @ freq_matrix.T).sum(axis=1)

    # Re-evaluate properly
    search_energy = np.full(n_search, c0, dtype=np.float64)
    for i, m in enumerate(freq_list):
        a = cos_coeffs.get(m, 0.0)
        b = sin_coeffs.get(m, 0.0)
        ph = search_pts @ np.array(m, dtype=np.float64)
        search_energy += a * np.cos(ph) + b * np.sin(ph)

    min_idx = int(np.argmin(search_energy))
    global_min = search_pts[min_idx]
    global_min_energy = float(search_energy[min_idx])

    # Find local minima (top-8 lowest energy distinct points)
    sorted_indices = np.argsort(search_energy)
    minima: list[dict] = []
    for idx in sorted_indices:
        pt = search_pts[idx]
        en = float(search_energy[idx])
        # Check if sufficiently far from existing minima
        is_new = True
        for existing in minima:
            diff = pt - np.array(existing["angles_rad"])
            dist = np.sqrt(np.sum((1.0 - np.cos(diff))))
            if dist < 0.5:
                is_new = False
                break
        if is_new:
            minima.append({
                "name": f"min_{len(minima)}",
                "angles_rad": [float(x) for x in pt],
                "energy_kjmol": en,
            })
        if len(minima) >= n_minima:
            break

    return EnergySurface(
        d=d,
        K_energy=K,
        c0=c0,
        cos_coeffs=cos_coeffs,
        sin_coeffs=sin_coeffs,
        global_min=global_min,
        global_min_energy=global_min_energy,
        minima=minima,
        angle_names=[f"phi{i // 2 + 1}" if i % 2 == 0 else f"psi{i // 2 + 1}" for i in range(d)],
    )


def load_energy_surface(json_path: str | Path | None, use_synthetic: bool = False, **synthetic_kwargs) -> EnergySurface:
    """Load energy surface from JSON, or fall back to synthetic."""
    if use_synthetic or json_path is None:
        return generate_synthetic_surface(**synthetic_kwargs)
    path = Path(json_path)
    if not path.exists():
        print(f"  Warning: {path} not found, using synthetic fallback")
        return generate_synthetic_surface(**synthetic_kwargs)
    return EnergySurface.from_json(path)
