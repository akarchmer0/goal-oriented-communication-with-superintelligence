"""Alanine dipeptide torsional energy surface via bicubic spline interpolation.

The energy surface is the AMBER ff14SB potential scanned on a (phi, psi) grid
by generate_energy_surface.py and cached in openmm_raw_grid.npz.  Energy and
gradient evaluation use periodic bicubic spline interpolation on this grid —
no Fourier truncation, so basin locations are faithful to the force field.

The Fourier lifting map (F: R^2 -> R^D) is a *separate* feature representation
used by the RL agent; it does not affect the energy surface.

All angles are in radians; the domain is the 2D torus [0, 2pi) x [0, 2pi).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Load the raw OpenMM grid and build a periodic bicubic spline.
# The grid is on [-pi, pi); we rearrange to [0, 2pi) and add one extra
# wraparound row/column so the spline is periodic.
# ---------------------------------------------------------------------------

_GRID_NPZ = Path(__file__).parent / "openmm_raw_grid.npz"


def _load_spline() -> tuple[RectBivariateSpline, np.ndarray, float, float]:
    """Load raw grid, shift to [0, 2pi), shift energy so min=0, build spline.

    Returns (spline, global_min_xy, energy_offset, cap_value).
    """
    data = np.load(_GRID_NPZ)
    phi_vals = data["phi_vals"].astype(np.float64)   # [-pi, pi)
    psi_vals = data["psi_vals"].astype(np.float64)
    raw_energy = data["energy_grid"].astype(np.float64)  # (N_phi, N_psi)

    # Shift angles to [0, 2pi)
    phi_02pi = phi_vals % TWO_PI
    psi_02pi = psi_vals % TWO_PI

    # Rearrange grid rows/columns so angles are sorted in [0, 2pi)
    phi_order = np.argsort(phi_02pi)
    psi_order = np.argsort(psi_02pi)
    phi_sorted = phi_02pi[phi_order]
    psi_sorted = psi_02pi[psi_order]
    energy_sorted = raw_energy[np.ix_(phi_order, psi_order)]

    # Cap extreme energies to keep the spline well-behaved in barrier regions.
    # Piecewise: identity below cap, tanh-squashed overflow above (C1-smooth).
    e_min = float(np.min(energy_sorted))
    cap_kj = 80.0
    shifted = energy_sorted - e_min
    overflow = np.maximum(0.0, shifted - cap_kj)
    energy_capped = e_min + np.minimum(shifted, cap_kj) + cap_kj * np.tanh(overflow / cap_kj)

    # Shift so minimum is 0
    e_min_capped = float(np.min(energy_capped))
    energy_final = energy_capped - e_min_capped

    # Global minimum location
    min_idx = np.unravel_index(np.argmin(energy_final), energy_final.shape)
    global_min_phi = float(phi_sorted[min_idx[0]])
    global_min_psi = float(psi_sorted[min_idx[1]])

    # Add wraparound for periodicity: append first row/col at the end
    # with angle shifted by +2pi
    dphi = phi_sorted[1] - phi_sorted[0]
    dpsi = psi_sorted[1] - psi_sorted[0]
    phi_ext = np.append(phi_sorted, phi_sorted[-1] + dphi)
    psi_ext = np.append(psi_sorted, psi_sorted[-1] + dpsi)
    energy_ext = np.empty((energy_final.shape[0] + 1, energy_final.shape[1] + 1),
                          dtype=np.float64)
    energy_ext[:-1, :-1] = energy_final
    energy_ext[-1, :-1] = energy_final[0, :]   # wrap phi
    energy_ext[:-1, -1] = energy_final[:, 0]   # wrap psi
    energy_ext[-1, -1] = energy_final[0, 0]    # wrap both

    spline = RectBivariateSpline(phi_ext, psi_ext, energy_ext, kx=3, ky=3)

    global_min_xy = np.array([global_min_phi, global_min_psi], dtype=np.float64)
    return spline, global_min_xy, e_min_capped, cap_kj


_SPLINE, ALANINE_GLOBAL_MIN, _E_OFFSET, _CAP_KJ = _load_spline()


# ---------------------------------------------------------------------------
# Energy evaluation (bicubic spline on the raw OpenMM grid)
# ---------------------------------------------------------------------------

def alanine_energy(phi: float, psi: float) -> float:
    """Evaluate the alanine dipeptide torsional energy at (phi, psi).

    Returns energy in kJ/mol, shifted so the global minimum is 0.
    """
    p = float(phi) % TWO_PI
    q = float(psi) % TWO_PI
    return float(_SPLINE(p, q, grid=False))


def alanine_energy_batch(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Vectorized energy evaluation over arrays of angles."""
    p = np.asarray(phi, dtype=np.float64).ravel() % TWO_PI
    q = np.asarray(psi, dtype=np.float64).ravel() % TWO_PI
    return _SPLINE(p, q, grid=False)


def alanine_gradient(phi: float, psi: float) -> tuple[float, float]:
    """Gradient (dE/dphi, dE/dpsi) via the spline partial derivatives."""
    p = float(phi) % TWO_PI
    q = float(psi) % TWO_PI
    dphi = float(_SPLINE(p, q, dx=1, dy=0, grid=False))
    dpsi = float(_SPLINE(p, q, dx=0, dy=1, grid=False))
    return dphi, dpsi


def alanine_energy_grid(resolution: int = 140) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (phi_grid, psi_grid, energy_grid) over [0, 2pi)^2."""
    res = max(20, int(resolution))
    phi_1d = np.linspace(0, TWO_PI, res, endpoint=False, dtype=np.float64)
    psi_1d = np.linspace(0, TWO_PI, res, endpoint=False, dtype=np.float64)
    phi_grid, psi_grid = np.meshgrid(phi_1d, psi_1d, indexing="xy")
    energy_grid = _SPLINE(phi_1d, psi_1d, grid=True)
    return (
        phi_grid.astype(np.float32),
        psi_grid.astype(np.float32),
        energy_grid.T.astype(np.float32),  # transpose: RectBivariateSpline grid=True is (phi, psi)
    )


# ---------------------------------------------------------------------------
# Fourier lifting map  F: R^2 -> R^D
#
# This is a *feature representation* for the RL agent, completely independent
# of the energy surface above.
# ---------------------------------------------------------------------------

def lifting_map_dim(K_max: int) -> int:
    """Dimension D of the Fourier lifting map for given K_max."""
    return 4 * K_max * (K_max + 1)


def lifting_map_eval(phi: float, psi: float, K_max: int) -> np.ndarray:
    """Evaluate the Fourier lifting map F(phi, psi) -> R^D.

    The basis functions are tensor products of 1D Fourier bases:
      For m in 1..K_max, n in 1..K_max:
        cos(m*phi), sin(m*phi)  (pure phi terms, 2K)
        cos(n*psi), sin(n*psi)  (pure psi terms, 2K)
        cos(m*phi)*cos(n*psi), cos(m*phi)*sin(n*psi),
        sin(m*phi)*cos(n*psi), sin(m*phi)*sin(n*psi)  (cross terms, 4K^2)
      Total: 4K + 4K^2 = 4K(K+1) = D
    """
    K = int(K_max)
    D = 4 * K * (K + 1)
    result = np.empty(D, dtype=np.float32)
    phi_f = float(phi)
    psi_f = float(psi)

    idx = 0
    # Pure phi terms
    for m in range(1, K + 1):
        result[idx] = np.cos(m * phi_f)
        idx += 1
        result[idx] = np.sin(m * phi_f)
        idx += 1
    # Pure psi terms
    for n in range(1, K + 1):
        result[idx] = np.cos(n * psi_f)
        idx += 1
        result[idx] = np.sin(n * psi_f)
        idx += 1
    # Cross terms
    for m in range(1, K + 1):
        cm = np.cos(m * phi_f)
        sm = np.sin(m * phi_f)
        for n in range(1, K + 1):
            cn = np.cos(n * psi_f)
            sn = np.sin(n * psi_f)
            result[idx] = cm * cn
            idx += 1
            result[idx] = cm * sn
            idx += 1
            result[idx] = sm * cn
            idx += 1
            result[idx] = sm * sn
            idx += 1
    assert idx == D
    return result


def lifting_map_jacobian(phi: float, psi: float, K_max: int) -> np.ndarray:
    """Jacobian of F(phi, psi): D x 2 matrix [dF/dphi, dF/dpsi]."""
    K = int(K_max)
    D = 4 * K * (K + 1)
    jac = np.zeros((D, 2), dtype=np.float32)
    phi_f = float(phi)
    psi_f = float(psi)

    idx = 0
    # Pure phi terms: d/dphi cos(m*phi) = -m*sin(m*phi), d/dpsi = 0
    for m in range(1, K + 1):
        jac[idx, 0] = -m * np.sin(m * phi_f)
        idx += 1
        jac[idx, 0] = m * np.cos(m * phi_f)
        idx += 1
    # Pure psi terms: d/dphi = 0, d/dpsi cos(n*psi) = -n*sin(n*psi)
    for n in range(1, K + 1):
        jac[idx, 1] = -n * np.sin(n * psi_f)
        idx += 1
        jac[idx, 1] = n * np.cos(n * psi_f)
        idx += 1
    # Cross terms
    for m in range(1, K + 1):
        cm = np.cos(m * phi_f)
        sm = np.sin(m * phi_f)
        for n in range(1, K + 1):
            cn = np.cos(n * psi_f)
            sn = np.sin(n * psi_f)
            # cos(m*phi)*cos(n*psi)
            jac[idx, 0] = -m * sm * cn
            jac[idx, 1] = -n * cm * sn
            idx += 1
            # cos(m*phi)*sin(n*psi)
            jac[idx, 0] = -m * sm * sn
            jac[idx, 1] = n * cm * cn
            idx += 1
            # sin(m*phi)*cos(n*psi)
            jac[idx, 0] = m * cm * cn
            jac[idx, 1] = -n * sm * sn
            idx += 1
            # sin(m*phi)*sin(n*psi)
            jac[idx, 0] = m * cm * sn
            jac[idx, 1] = n * sm * cn
            idx += 1
    assert idx == D
    return jac


def build_alanine_lifting_map(K_max: int) -> dict:
    """Return metadata about the lifting map for given K_max."""
    return {
        "K_max": int(K_max),
        "D": lifting_map_dim(K_max),
    }
