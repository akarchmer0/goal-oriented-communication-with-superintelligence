#!/usr/bin/env python3
"""One-time script: compute the alanine dipeptide Ramachandran E(phi,psi) surface
using OpenMM with the AMBER ff14SB force field, then fit a 2-D Fourier series
and print the coefficients to be hardcoded into alanine_energy.py.

Usage:
    python -m tasks.alanine_dipeptide.generate_energy_surface [--resolution 72] [--cap 80]

This only needs to be run once.  The raw grid is cached to openmm_raw_grid.npz so
re-running skips the expensive OpenMM scan.

Requirements:
    pip install openmm numpy tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Build alanine dipeptide in OpenMM
# ---------------------------------------------------------------------------

_PDB_LINES = (
    "ATOM      1 HH31 ACE A   1       2.000   1.000   0.000  1.00  0.00           H\n"
    "ATOM      2  CH3 ACE A   1       2.000   2.090   0.000  1.00  0.00           C\n"
    "ATOM      3 HH32 ACE A   1       1.486   2.454   0.890  1.00  0.00           H\n"
    "ATOM      4 HH33 ACE A   1       1.486   2.454  -0.890  1.00  0.00           H\n"
    "ATOM      5  C   ACE A   1       3.427   2.641   0.000  1.00  0.00           C\n"
    "ATOM      6  O   ACE A   1       4.391   1.877   0.000  1.00  0.00           O\n"
    "ATOM      7  N   ALA A   2       3.555   3.970   0.000  1.00  0.00           N\n"
    "ATOM      8  H   ALA A   2       2.733   4.556   0.000  1.00  0.00           H\n"
    "ATOM      9  CA  ALA A   2       4.853   4.614   0.000  1.00  0.00           C\n"
    "ATOM     10  HA  ALA A   2       5.408   4.316   0.890  1.00  0.00           H\n"
    "ATOM     11  CB  ALA A   2       5.661   4.221  -1.232  1.00  0.00           C\n"
    "ATOM     12  HB1 ALA A   2       5.123   4.521  -2.131  1.00  0.00           H\n"
    "ATOM     13  HB2 ALA A   2       6.630   4.719  -1.206  1.00  0.00           H\n"
    "ATOM     14  HB3 ALA A   2       5.809   3.141  -1.241  1.00  0.00           H\n"
    "ATOM     15  C   ALA A   2       4.713   6.129   0.000  1.00  0.00           C\n"
    "ATOM     16  O   ALA A   2       3.601   6.653   0.000  1.00  0.00           O\n"
    "ATOM     17  N   NME A   3       5.846   6.835   0.000  1.00  0.00           N\n"
    "ATOM     18  H   NME A   3       6.737   6.359   0.000  1.00  0.00           H\n"
    "ATOM     19  CH3 NME A   3       5.846   8.284   0.000  1.00  0.00           C\n"
    "ATOM     20 HH31 NME A   3       4.819   8.648   0.000  1.00  0.00           H\n"
    "ATOM     21 HH32 NME A   3       6.360   8.648   0.890  1.00  0.00           H\n"
    "ATOM     22 HH33 NME A   3       6.360   8.648  -0.890  1.00  0.00           H\n"
    "TER\n"
    "END\n"
)


def _build_system():
    """Build alanine dipeptide system with AMBER ff14SB."""
    import openmm
    import openmm.app as app
    from openmm import unit

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(_PDB_LINES)
        pdb_path = f.name
    try:
        pdb = app.PDBFile(pdb_path)
    finally:
        os.unlink(pdb_path)

    ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    return pdb.topology, pdb.positions, system


def _get_dihedral_atom_indices(topology):
    """Return (phi_indices, psi_indices) — each a list of 4 atom indices."""
    atoms = list(topology.atoms())
    atom_map = {}
    for a in atoms:
        atom_map[(a.residue.name, a.residue.index, a.name)] = a.index
    residues = list(topology.residues())
    ace_idx = next(r.index for r in residues if r.name == "ACE")
    ala_idx = next(r.index for r in residues if r.name == "ALA")
    nme_idx = next(r.index for r in residues if r.name == "NME")
    phi = [
        atom_map[("ACE", ace_idx, "C")],
        atom_map[("ALA", ala_idx, "N")],
        atom_map[("ALA", ala_idx, "CA")],
        atom_map[("ALA", ala_idx, "C")],
    ]
    psi = [
        atom_map[("ALA", ala_idx, "N")],
        atom_map[("ALA", ala_idx, "CA")],
        atom_map[("ALA", ala_idx, "C")],
        atom_map[("NME", nme_idx, "N")],
    ]
    return phi, psi


# ---------------------------------------------------------------------------
# Geometry manipulation: set dihedral angles by rotating atoms
# ---------------------------------------------------------------------------


def _get_positions_array(positions) -> np.ndarray:
    """Convert OpenMM Quantity positions to numpy array in nm."""
    from openmm import unit

    return np.array(positions.value_in_unit(unit.nanometers), dtype=np.float64)


def _positions_from_array(arr: np.ndarray):
    """Convert numpy array (nm) back to OpenMM Quantity."""
    from openmm import unit

    return arr.tolist() * unit.nanometers


def _dihedral_angle(coords: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Compute dihedral angle (radians) for atoms i-j-k-l."""
    b1 = coords[j] - coords[i]
    b2 = coords[k] - coords[j]
    b3 = coords[l] - coords[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-12 or n2_norm < 1e-12:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for rotating around `axis` by `angle` radians."""
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _atoms_downstream_of_bond(topology, bond_atom_j: int, bond_atom_k: int) -> set[int]:
    """Find all atoms on the k-side of the j-k bond (BFS from k, not crossing j)."""
    adj: dict[int, set[int]] = {}
    for bond in topology.bonds():
        a, b = bond[0].index, bond[1].index
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    visited = {bond_atom_j}
    queue = [bond_atom_k]
    result = set()
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        result.add(node)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)
    return result


def _set_dihedral(
    coords: np.ndarray,
    topology,
    dihedral_indices: list[int],
    target_angle: float,
) -> np.ndarray:
    """Rotate atoms to set a dihedral angle to target_angle (radians).

    Rotates all atoms downstream of the j-k bond (on the k side).
    """
    i, j, k, l = dihedral_indices
    current = _dihedral_angle(coords, i, j, k, l)
    delta = target_angle - current
    delta = (delta + np.pi) % (2 * np.pi) - np.pi

    if abs(delta) < 1e-10:
        return coords.copy()

    axis = coords[k] - coords[j]
    pivot = coords[j].copy()

    downstream = _atoms_downstream_of_bond(topology, j, k)
    R = _rotation_matrix(axis, delta)

    new_coords = coords.copy()
    for atom_idx in downstream:
        new_coords[atom_idx] = pivot + R @ (coords[atom_idx] - pivot)
    return new_coords


# ---------------------------------------------------------------------------
# Energy scan
# ---------------------------------------------------------------------------


def _scan_surface(resolution: int = 72) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan E(phi, psi) by explicitly setting dihedrals, minimizing other DOFs,
    and evaluating the unrestrained energy. Grid is on [-pi, pi)."""
    import openmm
    from openmm import unit

    topology, base_positions, base_system = _build_system()
    phi_idx, psi_idx = _get_dihedral_atom_indices(topology)
    print(f"  phi atoms: {phi_idx}, psi atoms: {psi_idx}")

    # Thorough unconstrained minimization to get good reference geometry
    integrator0 = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx0 = openmm.Context(base_system, integrator0)
    ctx0.setPositions(base_positions)
    openmm.LocalEnergyMinimizer.minimize(ctx0, tolerance=0.1, maxIterations=5000)
    ref_positions = ctx0.getState(getPositions=True).getPositions()
    ref_coords = _get_positions_array(ref_positions)
    del ctx0

    ref_phi = _dihedral_angle(ref_coords, *phi_idx)
    ref_psi = _dihedral_angle(ref_coords, *psi_idx)
    print(f"  Minimized reference: phi={np.degrees(ref_phi):.1f} deg, psi={np.degrees(ref_psi):.1f} deg")

    # Build restrained system for geometry relaxation at each (phi, psi)
    phi_restraint = openmm.CustomTorsionForce(
        "0.5*k_phi*min(dtheta, 2*3.141592653589793-dtheta)^2; "
        "dtheta = abs(theta-theta0_phi)"
    )
    phi_restraint.addGlobalParameter("theta0_phi", 0.0)
    phi_restraint.addGlobalParameter("k_phi", 10000.0)
    phi_restraint.addTorsion(phi_idx[0], phi_idx[1], phi_idx[2], phi_idx[3])

    psi_restraint = openmm.CustomTorsionForce(
        "0.5*k_psi*min(dtheta, 2*3.141592653589793-dtheta)^2; "
        "dtheta = abs(theta-theta0_psi)"
    )
    psi_restraint.addGlobalParameter("theta0_psi", 0.0)
    psi_restraint.addGlobalParameter("k_psi", 10000.0)
    psi_restraint.addTorsion(psi_idx[0], psi_idx[1], psi_idx[2], psi_idx[3])

    restrained_system = openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(base_system)
    )
    restrained_system.addForce(phi_restraint)
    restrained_system.addForce(psi_restraint)

    ctx_min = openmm.Context(
        restrained_system, openmm.VerletIntegrator(0.001 * unit.picoseconds)
    )
    ctx_eval = openmm.Context(
        base_system, openmm.VerletIntegrator(0.001 * unit.picoseconds)
    )

    phi_vals = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
    psi_vals = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
    energy_grid = np.zeros((resolution, resolution), dtype=np.float64)

    total = resolution * resolution
    pbar = tqdm(total=total, desc="Scanning (phi,psi)", unit="pt", ncols=90)

    for i, target_phi in enumerate(phi_vals):
        for j, target_psi in enumerate(psi_vals):
            # Step 1: set dihedrals explicitly by rotating atoms
            coords = _set_dihedral(ref_coords, topology, phi_idx, float(target_phi))
            coords = _set_dihedral(coords, topology, psi_idx, float(target_psi))

            # Step 2: restrained minimization to relax other DOFs
            ctx_min.setPositions(_positions_from_array(coords))
            ctx_min.setParameter("theta0_phi", float(target_phi))
            ctx_min.setParameter("theta0_psi", float(target_psi))
            ctx_min.setParameter("k_phi", 10000.0)
            ctx_min.setParameter("k_psi", 10000.0)
            openmm.LocalEnergyMinimizer.minimize(ctx_min, tolerance=1.0, maxIterations=200)

            # Step 3: verify dihedrals didn't drift too far
            relaxed_pos = ctx_min.getState(getPositions=True).getPositions()
            relaxed_coords = _get_positions_array(relaxed_pos)
            actual_phi = _dihedral_angle(relaxed_coords, *phi_idx)
            actual_psi = _dihedral_angle(relaxed_coords, *psi_idx)
            phi_err = abs((actual_phi - target_phi + np.pi) % (2 * np.pi) - np.pi)
            psi_err = abs((actual_psi - target_psi + np.pi) % (2 * np.pi) - np.pi)
            if phi_err > 0.1 or psi_err > 0.1:  # > ~6 degrees
                tqdm.write(
                    f"  WARNING: dihedral drift at ({np.degrees(target_phi):.0f}, "
                    f"{np.degrees(target_psi):.0f}): "
                    f"phi_err={np.degrees(phi_err):.1f} deg, "
                    f"psi_err={np.degrees(psi_err):.1f} deg"
                )

            # Step 4: evaluate unrestrained energy
            ctx_eval.setPositions(relaxed_pos)
            state = ctx_eval.getState(getEnergy=True)
            energy_grid[i, j] = float(
                state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            )

            pbar.update(1)

    pbar.close()
    del ctx_min, ctx_eval
    return phi_vals, psi_vals, energy_grid


# ---------------------------------------------------------------------------
# Fourier fitting
# ---------------------------------------------------------------------------


def _fit_fourier_2d(energy_grid: np.ndarray, K_max: int) -> dict:
    """Fit 2D Fourier coefficients via FFT.

    The input grid is assumed to cover one full period (either [-pi,pi) or [0,2pi)).
    We shift the energy so the minimum is 0.
    """
    N = energy_grid.shape[0]
    assert energy_grid.shape == (N, N)

    surface = energy_grid - np.min(energy_grid)
    spectrum = np.fft.fft2(surface) / (N * N)

    K = int(K_max)
    size = 2 * K + 1
    a_mn = np.zeros((size, size), dtype=np.float64)
    b_mn = np.zeros((size, size), dtype=np.float64)

    for im, m in enumerate(range(-K, K + 1)):
        for jn, n in enumerate(range(-K, K + 1)):
            mi = m % N
            ni = n % N
            coeff = spectrum[mi, ni]
            a_mn[im, jn] = float(np.real(coeff))
            b_mn[im, jn] = -float(np.imag(coeff))

    c0 = float(a_mn[K, K])
    a_mn[K, K] = 0.0
    b_mn[K, K] = 0.0
    return {"a_mn": a_mn, "b_mn": b_mn, "c0": c0, "K_max": K}


def _eval_fourier(phi: np.ndarray, psi: np.ndarray, coeffs: dict) -> np.ndarray:
    """Evaluate Fourier series at arrays of (phi, psi)."""
    a_mn = coeffs["a_mn"]
    b_mn = coeffs["b_mn"]
    c0 = coeffs["c0"]
    K = coeffs["K_max"]
    result = np.full_like(phi, c0, dtype=np.float64)
    for im, m in enumerate(range(-K, K + 1)):
        for jn, n in enumerate(range(-K, K + 1)):
            if m == 0 and n == 0:
                continue
            angle = m * phi + n * psi
            result += a_mn[im, jn] * np.cos(angle) + b_mn[im, jn] * np.sin(angle)
    return result


def _format_array(arr: np.ndarray, name: str, indent: str = "    ") -> str:
    """Format a 2D numpy array as a Python literal."""
    lines = [f"{indent}{name} = np.array(["]
    for i in range(arr.shape[0]):
        row_vals = ", ".join(f"{arr[i, j]: .14e}" for j in range(arr.shape[1]))
        lines.append(f"{indent}    [{row_vals}],")
    lines.append(f"{indent}], dtype=np.float64)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Known minima of alanine dipeptide in vacuum (approximate, in degrees)
_KNOWN_MINIMA = {
    "C7eq":    (-83.0,  75.0),
    "C7ax":    ( 70.0, -70.0),
    "alpha_R": (-80.0, -40.0),
    "alpha_L": ( 60.0,  40.0),
    "C5":      (-155.0, 160.0),
}


def _validate_surface(
    energy_grid: np.ndarray,
    phi_vals: np.ndarray,
    psi_vals: np.ndarray,
) -> None:
    """Print basic sanity checks on the raw energy surface."""
    e_min = float(np.min(energy_grid))
    min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
    min_phi_deg = float(np.degrees(phi_vals[min_idx[0]]))
    min_psi_deg = float(np.degrees(psi_vals[min_idx[1]]))

    print("\n--- Surface validation ---")
    print(f"  Global minimum: E={e_min:.2f} kJ/mol at phi={min_phi_deg:.1f} deg, psi={min_psi_deg:.1f} deg")

    # Check energy at known minima locations
    print("  Energy at known minima (interpolated from nearest grid point):")
    for name, (phi_deg, psi_deg) in _KNOWN_MINIMA.items():
        phi_rad = np.radians(phi_deg)
        psi_rad = np.radians(psi_deg)
        i_nearest = int(np.argmin(np.abs(phi_vals - phi_rad)))
        j_nearest = int(np.argmin(np.abs(psi_vals - psi_rad)))
        e_val = energy_grid[i_nearest, j_nearest] - e_min
        print(f"    {name:8s}: ({phi_deg:+6.0f}, {psi_deg:+6.0f}) deg  ->  E = {e_val:7.2f} kJ/mol above min")

    # Check that the global min is near C7eq
    c7eq_phi, c7eq_psi = _KNOWN_MINIMA["C7eq"]
    dphi = abs(min_phi_deg - c7eq_phi)
    dpsi = abs(min_psi_deg - c7eq_psi)
    if dphi > 180:
        dphi = 360 - dphi
    if dpsi > 180:
        dpsi = 360 - dpsi
    if dphi > 20 or dpsi > 20:
        print(f"  WARNING: global min ({min_phi_deg:.0f}, {min_psi_deg:.0f}) is far from "
              f"expected C7eq ({c7eq_phi:.0f}, {c7eq_psi:.0f})")
    else:
        print(f"  OK: global min is near expected C7eq location")


def _validate_fourier_fit(
    coeffs: dict,
    energy_for_fft: np.ndarray,
    resolution: int,
) -> None:
    """Validate that the Fourier reconstruction preserves secondary minima."""
    K = coeffs["K_max"]
    original = energy_for_fft - np.min(energy_for_fft)

    print(f"\n--- Fourier fit validation (K={K}) ---")
    print("  Reconstructed energy at known minima:")

    fine = 360
    phi_fine = np.linspace(0, 2 * np.pi, fine, endpoint=False)
    psi_fine = np.linspace(0, 2 * np.pi, fine, endpoint=False)
    pfg, qfg = np.meshgrid(phi_fine, psi_fine, indexing="ij")
    rec_fine = _eval_fourier(pfg.ravel(), qfg.ravel(), coeffs).reshape(fine, fine)
    rec_min = float(np.min(rec_fine))

    for name, (phi_deg, psi_deg) in _KNOWN_MINIMA.items():
        # Convert to [0, 2pi) for evaluation
        phi_rad = np.radians(phi_deg) % (2 * np.pi)
        psi_rad = np.radians(psi_deg) % (2 * np.pi)
        e_rec = float(_eval_fourier(
            np.array([phi_rad]), np.array([psi_rad]), coeffs
        )[0])
        print(f"    {name:8s}: E_reconstructed = {e_rec - rec_min:7.2f} kJ/mol above recon min")

    # Check that all expected minima are actually local minima in the reconstruction
    # (i.e., the reconstruction hasn't flattened them out)
    print("  Local minima check (is there a basin near each known minimum?):")
    for name, (phi_deg, psi_deg) in _KNOWN_MINIMA.items():
        phi_rad = np.radians(phi_deg) % (2 * np.pi)
        psi_rad = np.radians(psi_deg) % (2 * np.pi)
        i_center = int(np.argmin(np.abs(phi_fine - phi_rad)))
        j_center = int(np.argmin(np.abs(psi_fine - psi_rad)))
        # Check 15-degree neighborhood
        window = max(1, fine // 24)  # ~15 degrees
        i_lo = max(0, i_center - window)
        i_hi = min(fine, i_center + window + 1)
        j_lo = max(0, j_center - window)
        j_hi = min(fine, j_center + window + 1)
        patch = rec_fine[i_lo:i_hi, j_lo:j_hi]
        local_min_idx = np.unravel_index(np.argmin(patch), patch.shape)
        local_min_phi = phi_fine[i_lo + local_min_idx[0]]
        local_min_psi = psi_fine[j_lo + local_min_idx[1]]
        local_min_phi_deg = np.degrees(local_min_phi)
        if local_min_phi_deg > 180:
            local_min_phi_deg -= 360
        local_min_psi_deg = np.degrees(local_min_psi)
        if local_min_psi_deg > 180:
            local_min_psi_deg -= 360
        local_min_e = float(patch[local_min_idx]) - rec_min
        center_e = float(rec_fine[i_center, j_center]) - rec_min
        is_basin = float(np.min(patch)) < center_e + 1.0  # basin exists if min is near center
        status = "OK" if is_basin else "FLAT"
        print(f"    {name:8s}: local min at ({local_min_phi_deg:+6.0f}, {local_min_psi_deg:+6.0f}) "
              f"E={local_min_e:6.2f} kJ/mol  [{status}]")


# ---------------------------------------------------------------------------
# Fit and report
# ---------------------------------------------------------------------------


def _fit_and_report(energy_for_fft: np.ndarray, K_max: int, resolution: int) -> dict:
    """Fit at given K_max and report quality."""
    coeffs = _fit_fourier_2d(energy_for_fft, K_max=K_max)

    phi_check = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    psi_check = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    pg, qg = np.meshgrid(phi_check, psi_check, indexing="ij")
    reconstructed = _eval_fourier(pg.ravel(), qg.ravel(), coeffs).reshape(resolution, resolution)
    original = energy_for_fft - np.min(energy_for_fft)
    rmse = float(np.sqrt(np.mean((reconstructed - original) ** 2)))
    max_err = float(np.max(np.abs(reconstructed - original)))
    span = float(np.max(original))
    D = (2 * K_max + 1) ** 2 - 1

    # Fine-grid search for reconstructed minimum
    fine = 360
    phi_fine = np.linspace(0, 2 * np.pi, fine, endpoint=False)
    psi_fine = np.linspace(0, 2 * np.pi, fine, endpoint=False)
    pfg, qfg = np.meshgrid(phi_fine, psi_fine, indexing="ij")
    rec_fine = _eval_fourier(pfg.ravel(), qfg.ravel(), coeffs).reshape(fine, fine)
    min_idx = np.unravel_index(np.argmin(rec_fine), rec_fine.shape)
    rec_min_phi = float(phi_fine[min_idx[0]])
    rec_min_psi = float(psi_fine[min_idx[1]])
    rec_min_phi_deg = np.degrees(rec_min_phi) if rec_min_phi <= np.pi else np.degrees(rec_min_phi) - 360
    rec_min_psi_deg = np.degrees(rec_min_psi) if rec_min_psi <= np.pi else np.degrees(rec_min_psi) - 360

    print(
        f"  K={K_max:2d}  D={D:5d}  RMSE={rmse:7.3f}  max_err={max_err:7.3f}  "
        f"rel_RMSE={rmse / max(span, 1e-8) * 100:5.2f}%  "
        f"recon_min=({rec_min_phi_deg:+.0f}, {rec_min_psi_deg:+.0f}) deg"
    )

    return {
        "coeffs": coeffs,
        "rmse": rmse,
        "max_err": max_err,
        "rel_rmse": rmse / max(span, 1e-8),
        "rec_min_phi": rec_min_phi,
        "rec_min_psi": rec_min_psi,
        "D": D,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate alanine dipeptide Ramachandran surface")
    parser.add_argument("--resolution", type=int, default=72, help="Grid resolution (default: 72)")
    parser.add_argument("--cap", type=float, default=80.0, help="Energy cap in kJ/mol above minimum (default: 80)")
    parser.add_argument("--outdir", type=str, default="tasks/alanine_dipeptide", help="Output directory")
    args = parser.parse_args()

    RESOLUTION = args.resolution
    cap_kj = args.cap
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    GRID_NPY = os.path.join(outdir, "openmm_raw_grid.npz")

    # Check if we have a cached raw grid
    if os.path.exists(GRID_NPY):
        print(f"Loading cached grid from {GRID_NPY}")
        data = np.load(GRID_NPY)
        phi_vals = data["phi_vals"]
        psi_vals = data["psi_vals"]
        energy_grid = data["energy_grid"]
        print(f"  Grid shape: {energy_grid.shape}")
    else:
        print("=" * 70)
        print("Generating alanine dipeptide Ramachandran E(phi,psi) surface")
        print(f"  Resolution: {RESOLUTION}x{RESOLUTION} ({RESOLUTION**2} points)")
        print(f"  Force field: AMBER ff14SB (amber14-all.xml)")
        print("=" * 70)

        phi_vals, psi_vals, energy_grid = _scan_surface(resolution=RESOLUTION)
        os.makedirs(os.path.dirname(GRID_NPY) or ".", exist_ok=True)
        np.savez(GRID_NPY, phi_vals=phi_vals, psi_vals=psi_vals, energy_grid=energy_grid)
        print(f"\nSaved raw grid to {GRID_NPY}")

    e_min = float(np.min(energy_grid))
    e_max = float(np.max(energy_grid))
    print(f"\nRaw energy range: [{e_min:.4f}, {e_max:.4f}] kJ/mol  (span={e_max - e_min:.1f})")

    min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
    min_phi_deg = np.degrees(phi_vals[min_idx[0]])
    min_psi_deg = np.degrees(psi_vals[min_idx[1]])
    print(f"Raw grid minimum at phi={min_phi_deg:+.1f} deg, psi={min_psi_deg:+.1f} deg")

    # Validate the raw surface
    _validate_surface(energy_grid, phi_vals, psi_vals)

    # Global minimum in [0, 2pi)
    raw_min_phi_02pi = float(phi_vals[min_idx[0]]) % (2 * np.pi)
    raw_min_psi_02pi = float(psi_vals[min_idx[1]]) % (2 * np.pi)

    # Piecewise soft cap: identity below cap, tanh-squashed overflow above.
    # This preserves the low-energy basins exactly while smoothly compressing
    # the high-energy barriers, avoiding the sharp edge that causes Gibbs ringing.
    # The junction at cap is C1-continuous (both value and derivative match).
    shifted = energy_grid - e_min  # minimum at 0
    overflow = np.maximum(0.0, shifted - cap_kj)
    energy_capped = e_min + np.minimum(shifted, cap_kj) + cap_kj * np.tanh(overflow / cap_kj)
    n_affected = int(np.sum(shifted > cap_kj))
    print(f"\nPiecewise soft-cap at {cap_kj} kJ/mol (identity below, tanh above)  "
          f"({n_affected} points ({100 * n_affected / energy_grid.size:.1f}%) above cap)")

    # Rearrange [-pi,pi) grid to [0,2pi) for FFT
    energy_for_fft = np.fft.ifftshift(energy_capped)

    # Fit multiple K values
    K_values = [4, 6, 8, 10, 12, 15, 20, 25]
    K_values = [K for K in K_values if 2 * K + 1 <= RESOLUTION]

    print(f"\nFourier fit quality (capped at {cap_kj} kJ/mol above min):")
    print(f"  {'K':>3s}  {'D':>5s}  {'RMSE':>7s}  {'max_err':>7s}  {'rel_RMSE':>8s}  {'recon_min':>14s}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*14}")

    results = {}
    for K in K_values:
        results[K] = _fit_and_report(energy_for_fft, K, RESOLUTION)

    # Select K: smallest K with minimum within ~20 degrees and relative RMSE < 5%
    K_MAX = K_values[0]  # fallback to smallest
    for K in sorted(results.keys()):
        r = results[K]
        rec_phi_deg = (
            np.degrees(r["rec_min_phi"])
            if r["rec_min_phi"] <= np.pi
            else np.degrees(r["rec_min_phi"]) - 360
        )
        rec_psi_deg = (
            np.degrees(r["rec_min_psi"])
            if r["rec_min_psi"] <= np.pi
            else np.degrees(r["rec_min_psi"]) - 360
        )
        dphi = abs(rec_phi_deg - min_phi_deg)
        dpsi = abs(rec_psi_deg - min_psi_deg)
        if dphi > 180:
            dphi = 360 - dphi
        if dpsi > 180:
            dpsi = 360 - dpsi
        if dphi < 20 and dpsi < 20 and r["rel_rmse"] < 0.05:
            K_MAX = K
            break

    print(f"\nSelected K_max = {K_MAX}  (D = {results[K_MAX]['D']})")
    chosen = results[K_MAX]
    coeffs = chosen["coeffs"]

    # Validate the chosen fit
    _validate_fourier_fit(coeffs, energy_for_fft, RESOLUTION)

    # Use raw grid minimum as ground truth
    global_min_phi = raw_min_phi_02pi
    global_min_psi = raw_min_psi_02pi
    print(f"\nGlobal minimum (from raw OpenMM grid):")
    print(f"  phi = {min_phi_deg:+.1f} deg = {global_min_phi:.6f} rad [0,2pi)")
    print(f"  psi = {min_psi_deg:+.1f} deg = {global_min_psi:.6f} rad [0,2pi)")

    # Save all K fits to JSON
    all_fits = {}
    for K, r in results.items():
        c = r["coeffs"]
        all_fits[str(K)] = {
            "K_max": K,
            "D": r["D"],
            "c0": c["c0"],
            "a_mn": c["a_mn"].tolist(),
            "b_mn": c["b_mn"].tolist(),
            "rmse": r["rmse"],
            "max_err": r["max_err"],
            "rel_rmse": r["rel_rmse"],
        }

    output_data = {
        "selected_K_max": K_MAX,
        "resolution": RESOLUTION,
        "force_field": "AMBER ff14SB (amber14-all.xml)",
        "cap_kj_mol": cap_kj,
        "global_min_phi_rad": global_min_phi,
        "global_min_psi_rad": global_min_psi,
        "raw_grid_min_phi_deg": float(min_phi_deg),
        "raw_grid_min_psi_deg": float(min_psi_deg),
        "raw_energy_range_kjmol": [e_min, e_max],
        "fits": all_fits,
    }
    json_path = os.path.join(outdir, "openmm_fourier_coefficients.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved all fits to {json_path}")

    # Print Python code for hardcoding the selected fit
    print("\n" + "=" * 70)
    print(f"PYTHON CODE FOR alanine_energy.py  (K_max={K_MAX}, D={chosen['D']})")
    print("=" * 70 + "\n")
    print(f"_DEFAULT_K_MAX = {K_MAX}")
    print(f"_OPENMM_C0 = {coeffs['c0']:.15e}")
    print(f"_OPENMM_GLOBAL_MIN_PHI = {global_min_phi:.15e}")
    print(f"_OPENMM_GLOBAL_MIN_PSI = {global_min_psi:.15e}")
    print()
    print(_format_array(coeffs["a_mn"], "_OPENMM_A_MN"))
    print()
    print(_format_array(coeffs["b_mn"], "_OPENMM_B_MN"))


if __name__ == "__main__":
    main()
