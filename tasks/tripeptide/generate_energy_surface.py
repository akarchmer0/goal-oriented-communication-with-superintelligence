#!/usr/bin/env python3
"""One-time script: compute the Ace-Ala-Ala-Nme tripeptide energy surface
E(phi1, psi1, phi2, psi2) using OpenMM with AMBER ff14SB, fit a 4D Fourier
series, and save coefficients to fourier_coefficients.json.

Usage:
    python -m tasks.tripeptide.generate_energy_surface [--resolution 12] [--K_energy 3]

This only needs to be run once. The raw grid is cached to openmm_raw_grid_4d.npz.

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

TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Build Ace-Ala-Ala-Nme in OpenMM
# ---------------------------------------------------------------------------

# PDB for Ace-Ala-Ala-Nme (extended conformation)
_PDB_LINES = """\
ATOM      1 HH31 ACE A   1       2.000   1.000   0.000  1.00  0.00           H
ATOM      2  CH3 ACE A   1       2.000   2.090   0.000  1.00  0.00           C
ATOM      3 HH32 ACE A   1       1.486   2.454   0.890  1.00  0.00           H
ATOM      4 HH33 ACE A   1       1.486   2.454  -0.890  1.00  0.00           H
ATOM      5  C   ACE A   1       3.427   2.641   0.000  1.00  0.00           C
ATOM      6  O   ACE A   1       4.391   1.877   0.000  1.00  0.00           O
ATOM      7  N   ALA A   2       3.555   3.970   0.000  1.00  0.00           N
ATOM      8  H   ALA A   2       2.733   4.556   0.000  1.00  0.00           H
ATOM      9  CA  ALA A   2       4.853   4.614   0.000  1.00  0.00           C
ATOM     10  HA  ALA A   2       5.408   4.316   0.890  1.00  0.00           H
ATOM     11  CB  ALA A   2       5.661   4.221  -1.232  1.00  0.00           C
ATOM     12  HB1 ALA A   2       5.123   4.521  -2.131  1.00  0.00           H
ATOM     13  HB2 ALA A   2       6.630   4.719  -1.206  1.00  0.00           H
ATOM     14  HB3 ALA A   2       5.809   3.141  -1.241  1.00  0.00           H
ATOM     15  C   ALA A   2       4.713   6.129   0.000  1.00  0.00           C
ATOM     16  O   ALA A   2       3.601   6.653   0.000  1.00  0.00           O
ATOM     17  N   ALA A   3       5.846   6.835   0.000  1.00  0.00           N
ATOM     18  H   ALA A   3       6.737   6.359   0.000  1.00  0.00           H
ATOM     19  CA  ALA A   3       5.846   8.284   0.000  1.00  0.00           C
ATOM     20  HA  ALA A   3       6.401   8.582   0.890  1.00  0.00           H
ATOM     21  CB  ALA A   3       6.654   8.677  -1.232  1.00  0.00           C
ATOM     22  HB1 ALA A   3       6.116   8.977  -2.131  1.00  0.00           H
ATOM     23  HB2 ALA A   3       7.623   9.175  -1.206  1.00  0.00           H
ATOM     24  HB3 ALA A   3       6.802   7.597  -1.241  1.00  0.00           H
ATOM     25  C   ALA A   3       5.706   9.799   0.000  1.00  0.00           C
ATOM     26  O   ALA A   3       4.594  10.323   0.000  1.00  0.00           O
ATOM     27  N   NME A   4       6.839  10.505   0.000  1.00  0.00           N
ATOM     28  H   NME A   4       7.730  10.029   0.000  1.00  0.00           H
ATOM     29  CH3 NME A   4       6.839  11.954   0.000  1.00  0.00           C
ATOM     30 HH31 NME A   4       5.812  12.318   0.000  1.00  0.00           H
ATOM     31 HH32 NME A   4       7.353  12.318   0.890  1.00  0.00           H
ATOM     32 HH33 NME A   4       7.353  12.318  -0.890  1.00  0.00           H
TER
END
"""


def _build_system():
    """Build Ace-Ala-Ala-Nme system with AMBER ff14SB."""
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
    """Return (phi1, psi1, phi2, psi2) — each a list of 4 atom indices.

    phi1: ACE:C  - ALA2:N  - ALA2:CA - ALA2:C
    psi1: ALA2:N - ALA2:CA - ALA2:C  - ALA3:N
    phi2: ALA2:C - ALA3:N  - ALA3:CA - ALA3:C
    psi2: ALA3:N - ALA3:CA - ALA3:C  - NME:N
    """
    atoms = list(topology.atoms())
    atom_map = {}
    for a in atoms:
        atom_map[(a.residue.name, a.residue.index, a.name)] = a.index

    residues = list(topology.residues())
    ace_idx = next(r.index for r in residues if r.name == "ACE")
    ala_indices = [r.index for r in residues if r.name == "ALA"]
    nme_idx = next(r.index for r in residues if r.name == "NME")

    if len(ala_indices) != 2:
        raise RuntimeError(f"Expected 2 ALA residues, found {len(ala_indices)}")
    ala1_idx, ala2_idx = ala_indices

    phi1 = [
        atom_map[("ACE", ace_idx, "C")],
        atom_map[("ALA", ala1_idx, "N")],
        atom_map[("ALA", ala1_idx, "CA")],
        atom_map[("ALA", ala1_idx, "C")],
    ]
    psi1 = [
        atom_map[("ALA", ala1_idx, "N")],
        atom_map[("ALA", ala1_idx, "CA")],
        atom_map[("ALA", ala1_idx, "C")],
        atom_map[("ALA", ala2_idx, "N")],
    ]
    phi2 = [
        atom_map[("ALA", ala1_idx, "C")],
        atom_map[("ALA", ala2_idx, "N")],
        atom_map[("ALA", ala2_idx, "CA")],
        atom_map[("ALA", ala2_idx, "C")],
    ]
    psi2 = [
        atom_map[("ALA", ala2_idx, "N")],
        atom_map[("ALA", ala2_idx, "CA")],
        atom_map[("ALA", ala2_idx, "C")],
        atom_map[("NME", nme_idx, "N")],
    ]
    return phi1, psi1, phi2, psi2


# ---------------------------------------------------------------------------
# Geometry manipulation
# ---------------------------------------------------------------------------

def _get_positions_array(positions) -> np.ndarray:
    from openmm import unit
    return np.array(positions.value_in_unit(unit.nanometers), dtype=np.float64)


def _positions_from_array(arr: np.ndarray):
    from openmm import unit
    return arr.tolist() * unit.nanometers


def _dihedral_angle(coords: np.ndarray, i: int, j: int, k: int, l: int) -> float:
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
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _atoms_downstream_of_bond(topology, bond_atom_j: int, bond_atom_k: int) -> set[int]:
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
    i, j, k, l = dihedral_indices
    current = _dihedral_angle(coords, i, j, k, l)
    delta = target_angle - current
    delta = (delta + np.pi) % TWO_PI - np.pi
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
# Energy scan on 4D torus
# ---------------------------------------------------------------------------

def _scan_surface_4d(
    resolution: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan E(phi1, psi1, phi2, psi2) on a regular grid.

    Grid is on [-pi, pi). Returns angle arrays and energy_grid of shape
    (resolution, resolution, resolution, resolution).
    """
    import openmm
    from openmm import unit

    topology, base_positions, base_system = _build_system()
    phi1_idx, psi1_idx, phi2_idx, psi2_idx = _get_dihedral_atom_indices(topology)
    dihedrals = [
        ("phi1", phi1_idx),
        ("psi1", psi1_idx),
        ("phi2", phi2_idx),
        ("psi2", psi2_idx),
    ]
    for name, idx in dihedrals:
        print(f"  {name} atoms: {idx}")

    # Minimize reference geometry
    integrator0 = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx0 = openmm.Context(base_system, integrator0)
    ctx0.setPositions(base_positions)
    openmm.LocalEnergyMinimizer.minimize(ctx0, tolerance=0.1, maxIterations=5000)
    ref_positions = ctx0.getState(getPositions=True).getPositions()
    ref_coords = _get_positions_array(ref_positions)
    del ctx0

    for name, idx in dihedrals:
        angle = _dihedral_angle(ref_coords, *idx)
        print(f"  Minimized reference {name}: {np.degrees(angle):.1f} deg")

    # Build restrained system
    restrained_system = openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(base_system)
    )
    restraint_params = []
    for name, idx in dihedrals:
        param_name = f"theta0_{name}"
        k_name = f"k_{name}"
        force = openmm.CustomTorsionForce(
            f"0.5*{k_name}*min(dtheta, 2*3.141592653589793-dtheta)^2; "
            f"dtheta = abs(theta-{param_name})"
        )
        force.addGlobalParameter(param_name, 0.0)
        force.addGlobalParameter(k_name, 10000.0)
        force.addTorsion(idx[0], idx[1], idx[2], idx[3])
        restrained_system.addForce(force)
        restraint_params.append((param_name, k_name))

    ctx_min = openmm.Context(
        restrained_system, openmm.VerletIntegrator(0.001 * unit.picoseconds)
    )
    ctx_eval = openmm.Context(
        base_system, openmm.VerletIntegrator(0.001 * unit.picoseconds)
    )

    angles = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
    total = resolution ** 4
    energy_grid = np.zeros((resolution,) * 4, dtype=np.float64)

    pbar = tqdm(total=total, desc="Scanning 4D torus", unit="pt", ncols=100)
    drift_warnings = 0

    for i0, a0 in enumerate(angles):
        for i1, a1 in enumerate(angles):
            for i2, a2 in enumerate(angles):
                for i3, a3 in enumerate(angles):
                    target_angles = [a0, a1, a2, a3]

                    # Set dihedrals by rotating atoms
                    coords = ref_coords.copy()
                    for d_idx, (_, dih_atoms) in enumerate(dihedrals):
                        coords = _set_dihedral(
                            coords, topology, dih_atoms, target_angles[d_idx]
                        )

                    # Restrained minimization
                    ctx_min.setPositions(_positions_from_array(coords))
                    for d_idx, (param_name, k_name) in enumerate(restraint_params):
                        ctx_min.setParameter(param_name, float(target_angles[d_idx]))
                        ctx_min.setParameter(k_name, 10000.0)
                    openmm.LocalEnergyMinimizer.minimize(
                        ctx_min, tolerance=1.0, maxIterations=200
                    )

                    # Verify dihedrals
                    relaxed_pos = ctx_min.getState(getPositions=True).getPositions()
                    relaxed_coords = _get_positions_array(relaxed_pos)
                    max_err = 0.0
                    for d_idx, (name, dih_atoms) in enumerate(dihedrals):
                        actual = _dihedral_angle(relaxed_coords, *dih_atoms)
                        err = abs(
                            (actual - target_angles[d_idx] + np.pi) % TWO_PI - np.pi
                        )
                        max_err = max(max_err, err)
                    if max_err > 0.15:  # ~8.6 degrees
                        drift_warnings += 1

                    # Evaluate unrestrained energy
                    ctx_eval.setPositions(relaxed_pos)
                    state = ctx_eval.getState(getEnergy=True)
                    energy_grid[i0, i1, i2, i3] = float(
                        state.getPotentialEnergy().value_in_unit(
                            unit.kilojoules_per_mole
                        )
                    )
                    pbar.update(1)

    pbar.close()
    if drift_warnings > 0:
        print(
            f"  WARNING: {drift_warnings}/{total} points "
            f"({100*drift_warnings/total:.1f}%) had dihedral drift > 8.6 deg"
        )
    del ctx_min, ctx_eval
    return angles, angles, angles, angles, energy_grid


# ---------------------------------------------------------------------------
# Fourier fitting via least squares on 4D torus
# ---------------------------------------------------------------------------

def _fit_fourier_4d(
    energy_grid: np.ndarray,
    angle_vals: np.ndarray,
    K_max: int,
) -> dict:
    """Fit 4D Fourier coefficients via FFT on regular grid.

    energy_grid: shape (N, N, N, N) on [0, 2pi) grid.
    Returns dict with c0, cos_coeffs, sin_coeffs.
    """
    d = 4
    N = angle_vals.shape[0]
    assert energy_grid.shape == (N,) * d

    # Use multi-dimensional FFT
    spectrum = np.fft.fftn(energy_grid) / (N ** d)

    K = int(K_max)
    freq_range = range(-K, K + 1)
    cos_coeffs: dict[str, float] = {}
    sin_coeffs: dict[str, float] = {}

    c0 = float(np.real(spectrum[(0,) * d]))

    for m0 in freq_range:
        for m1 in freq_range:
            for m2 in freq_range:
                for m3 in freq_range:
                    m = (m0, m1, m2, m3)
                    if all(x == 0 for x in m):
                        continue
                    # FFT index (handle negative frequencies)
                    idx = tuple(mi % N for mi in m)
                    coeff = spectrum[idx]
                    a = float(np.real(coeff))
                    b = -float(np.imag(coeff))
                    key = f"{m0},{m1},{m2},{m3}"
                    if abs(a) > 1e-14:
                        cos_coeffs[key] = a
                    if abs(b) > 1e-14:
                        sin_coeffs[key] = b

    return {"c0": c0, "cos_coeffs": cos_coeffs, "sin_coeffs": sin_coeffs, "K_max": K}


def _eval_fourier_4d(z: np.ndarray, coeffs: dict) -> float:
    """Evaluate 4D Fourier series at a single point z = (z0, z1, z2, z3)."""
    c0 = coeffs["c0"]
    result = c0
    for key, a in coeffs["cos_coeffs"].items():
        m = np.array([int(x) for x in key.split(",")], dtype=np.float64)
        result += a * np.cos(np.dot(m, z))
    for key, b in coeffs["sin_coeffs"].items():
        m = np.array([int(x) for x in key.split(",")], dtype=np.float64)
        result += b * np.sin(np.dot(m, z))
    return float(result)


def _eval_fourier_4d_batch(z_batch: np.ndarray, coeffs: dict) -> np.ndarray:
    """Evaluate 4D Fourier series at multiple points. z_batch shape (B, 4)."""
    B = z_batch.shape[0]
    result = np.full(B, coeffs["c0"], dtype=np.float64)

    # Build arrays for vectorized evaluation
    cos_keys = list(coeffs["cos_coeffs"].keys())
    sin_keys = list(coeffs["sin_coeffs"].keys())

    if cos_keys:
        cos_freqs = np.array(
            [[int(x) for x in k.split(",")] for k in cos_keys], dtype=np.float64
        )
        cos_vals = np.array(
            [coeffs["cos_coeffs"][k] for k in cos_keys], dtype=np.float64
        )
        phases = z_batch @ cos_freqs.T  # (B, N_cos)
        result += np.cos(phases) @ cos_vals

    if sin_keys:
        sin_freqs = np.array(
            [[int(x) for x in k.split(",")] for k in sin_keys], dtype=np.float64
        )
        sin_vals = np.array(
            [coeffs["sin_coeffs"][k] for k in sin_keys], dtype=np.float64
        )
        phases = z_batch @ sin_freqs.T  # (B, N_sin)
        result += np.sin(phases) @ sin_vals

    return result


# ---------------------------------------------------------------------------
# Find minima
# ---------------------------------------------------------------------------

def _find_minima(
    coeffs: dict,
    n_random: int = 200_000,
    n_refine_candidates: int = 50,
    min_torus_distance: float = 0.5,
) -> list[dict]:
    """Find local minima by random search + scipy refinement."""
    from scipy.optimize import minimize as scipy_minimize

    rng = np.random.default_rng(42)
    z_random = rng.uniform(0.0, TWO_PI, size=(n_random, 4))
    energies = _eval_fourier_4d_batch(z_random, coeffs)

    # Take top candidates
    sorted_idx = np.argsort(energies)
    candidates = []
    for idx in sorted_idx[:n_refine_candidates]:
        z0 = z_random[idx]
        # Refine with L-BFGS-B (periodic: we optimize, then wrap)
        def neg_energy(z):
            return _eval_fourier_4d(z % TWO_PI, coeffs)
        result = scipy_minimize(neg_energy, z0, method="L-BFGS-B", options={"maxiter": 200})
        z_opt = result.x % TWO_PI
        e_opt = _eval_fourier_4d(z_opt, coeffs)
        candidates.append((z_opt, e_opt))

    # Deduplicate by torus distance
    candidates.sort(key=lambda x: x[1])
    minima: list[dict] = []
    for z_opt, e_opt in candidates:
        is_new = True
        for existing in minima:
            diff = z_opt - np.array(existing["angles_rad"])
            diff = (diff + np.pi) % TWO_PI - np.pi
            dist = np.linalg.norm(diff)
            if dist < min_torus_distance:
                is_new = False
                break
        if is_new:
            minima.append({
                "name": f"min_{len(minima)}",
                "angles_rad": [float(x) for x in z_opt],
                "energy_kjmol": float(e_opt),
            })
    return minima


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_fit(
    coeffs: dict,
    energy_grid: np.ndarray,
    angle_vals: np.ndarray,
) -> dict:
    """Validate Fourier fit against the raw grid."""
    N = angle_vals.shape[0]
    # Reconstruct on the same grid
    grid_02pi = angle_vals % TWO_PI
    flat_points = np.stack(
        np.meshgrid(grid_02pi, grid_02pi, grid_02pi, grid_02pi, indexing="ij"),
        axis=-1,
    ).reshape(-1, 4)
    recon_flat = _eval_fourier_4d_batch(flat_points, coeffs)
    recon_grid = recon_flat.reshape((N,) * 4)

    diff = recon_grid - energy_grid
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(np.abs(diff)))
    span = float(np.max(energy_grid) - np.min(energy_grid))
    rel_rmse = rmse / max(span, 1e-8)

    print(f"  Fit quality: RMSE={rmse:.4f} kJ/mol, max_err={max_err:.4f} kJ/mol, "
          f"rel_RMSE={100*rel_rmse:.2f}%, span={span:.1f} kJ/mol")
    return {"rmse": rmse, "max_err": max_err, "rel_rmse": rel_rmse, "span": span}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate tripeptide (Ace-Ala-Ala-Nme) 4D energy surface"
    )
    parser.add_argument(
        "--resolution", type=int, default=12,
        help="Grid points per angle (default: 12, giving 12^4=20736 points)",
    )
    parser.add_argument(
        "--K_energy", type=int, default=3,
        help="Max Fourier frequency (default: 3)",
    )
    parser.add_argument(
        "--cap", type=float, default=100.0,
        help="Energy cap in kJ/mol above minimum (default: 100)",
    )
    parser.add_argument(
        "--outdir", type=str, default="tasks/tripeptide",
        help="Output directory",
    )
    args = parser.parse_args()

    resolution = args.resolution
    K_energy = args.K_energy
    cap_kj = args.cap
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    grid_path = os.path.join(outdir, "openmm_raw_grid_4d.npz")
    total_points = resolution ** 4

    # Step 1: Scan or load cached grid
    if os.path.exists(grid_path):
        print(f"Loading cached grid from {grid_path}")
        data = np.load(grid_path)
        angle_vals = data["angle_vals"]
        energy_grid = data["energy_grid"]
        print(f"  Grid shape: {energy_grid.shape}, resolution={angle_vals.shape[0]}")
    else:
        print("=" * 70)
        print("Generating Ace-Ala-Ala-Nme 4D energy surface")
        print(f"  Resolution: {resolution} per angle ({total_points:,} points)")
        print(f"  Force field: AMBER ff14SB (amber14-all.xml)")
        print("=" * 70)

        t0 = time.perf_counter()
        a0, a1, a2, a3, energy_grid = _scan_surface_4d(resolution=resolution)
        angle_vals = a0  # all the same for regular grid
        elapsed = time.perf_counter() - t0
        print(f"\nScan completed in {elapsed:.1f}s ({elapsed/total_points*1000:.1f}ms/point)")

        np.savez(grid_path, angle_vals=angle_vals, energy_grid=energy_grid)
        print(f"Saved raw grid to {grid_path}")

    e_min = float(np.min(energy_grid))
    e_max = float(np.max(energy_grid))
    min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
    print(f"\nRaw energy range: [{e_min:.2f}, {e_max:.2f}] kJ/mol (span={e_max-e_min:.1f})")
    print(f"Grid minimum at indices {min_idx}")
    min_angles_deg = [float(np.degrees(angle_vals[i])) for i in min_idx]
    print(f"  = ({', '.join(f'{a:+.1f}' for a in min_angles_deg)}) deg")

    # Step 2: Soft cap high-energy barriers
    shifted = energy_grid - e_min
    overflow = np.maximum(0.0, shifted - cap_kj)
    energy_capped = e_min + np.minimum(shifted, cap_kj) + cap_kj * np.tanh(overflow / cap_kj)
    n_affected = int(np.sum(shifted > cap_kj))
    print(f"\nSoft cap at {cap_kj} kJ/mol above min: "
          f"{n_affected}/{total_points} points ({100*n_affected/total_points:.1f}%) affected")

    # Step 3: Rearrange grid from [-pi,pi) to [0,2pi) for FFT
    # Apply fftshift inverse (ifftshift) along each axis
    energy_for_fft = energy_capped.copy()
    for axis in range(4):
        energy_for_fft = np.fft.ifftshift(energy_for_fft, axes=axis)

    # Step 4: Fit Fourier coefficients
    print(f"\nFitting Fourier series with K_energy={K_energy}...")
    coeffs = _fit_fourier_4d(energy_for_fft, angle_vals, K_max=K_energy)
    n_cos = len(coeffs["cos_coeffs"])
    n_sin = len(coeffs["sin_coeffs"])
    print(f"  c0={coeffs['c0']:.4f}, {n_cos} cos terms, {n_sin} sin terms")

    # Step 5: Validate fit
    print("\nValidating fit on the original grid...")
    _validate_fit(coeffs, energy_for_fft, angle_vals)

    # Step 6: Find minima
    print("\nSearching for minima...")
    minima = _find_minima(coeffs, n_random=500_000, n_refine_candidates=100)
    print(f"  Found {len(minima)} distinct minima:")
    for m in minima[:20]:
        angles_deg = [np.degrees(a) for a in m["angles_rad"]]
        # Convert to [-180, 180) for display
        angles_deg = [(a + 180) % 360 - 180 for a in angles_deg]
        print(f"    {m['name']}: E={m['energy_kjmol']:.2f} kJ/mol at "
              f"({', '.join(f'{a:+.0f}' for a in angles_deg)}) deg")

    global_min_idx = 0  # already sorted by energy
    global_min = minima[global_min_idx]
    print(f"\n  Global minimum: {global_min['name']} at E={global_min['energy_kjmol']:.4f} kJ/mol")

    # Step 7: Save to JSON
    output = {
        "d": 4,
        "K_energy": K_energy,
        "c0": coeffs["c0"],
        "angle_names": ["phi1", "psi1", "phi2", "psi2"],
        "minima": minima,
        "global_min_index": global_min_idx,
        "coefficients": {
            "cos": coeffs["cos_coeffs"],
            "sin": coeffs["sin_coeffs"],
        },
        "metadata": {
            "force_field": "AMBER ff14SB (amber14-all.xml)",
            "molecule": "Ace-Ala-Ala-Nme",
            "resolution": int(angle_vals.shape[0]),
            "cap_kj_mol": cap_kj,
            "raw_energy_range_kjmol": [e_min, e_max],
        },
    }

    json_path = os.path.join(outdir, "fourier_coefficients.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
