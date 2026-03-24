"""SDP relaxation for computing the oracle target s*_SDP.

Solves: min c_energy^T s  subject to M(s) >= 0 (PSD)
where M(s) is the trigonometric moment matrix parametrised by s.

The solution s*_SDP approximates F(z*) where z* is the true global minimum,
WITHOUT requiring knowledge of z*.

Uses sparse basis matrices for memory-efficient construction of the moment
matrix constraint.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from .lifting_map import LiftingMap, enumerate_frequency_indices


def _canonical_positive(m: tuple[int, ...]) -> bool:
    """Return True if m is 'positive' — first nonzero component > 0."""
    for x in m:
        if x != 0:
            return x > 0
    return False  # zero vector


def solve_sdp_oracle(
    lifting_map: LiftingMap,
    energy_surface,
    K_relax: int | None = None,
) -> tuple[np.ndarray, float, str]:
    """Solve the SDP relaxation to compute s*_SDP.

    Args:
        lifting_map: LiftingMap instance (provides frequency structure).
        energy_surface: EnergySurface instance (provides Fourier coefficients).
        K_relax: Relaxation order for moment matrix. Default: min(K_map, 2).

    Returns:
        s_star_sdp: D-dimensional vector in the lifting map's format.
        sdp_bound: Lower bound on the energy from the SDP.
        status: Solver status string.
    """
    import cvxpy as cp

    d = energy_surface.d
    if K_relax is None:
        K_relax = min(lifting_map.K_map, 2)

    print(f"  Solving SDP relaxation (K_relax={K_relax}, d={d})...")

    # Index set I: all m with |m_i| <= K_relax
    I_indices_raw = enumerate_frequency_indices(d, K_relax)
    # Add the zero vector
    zero = np.zeros((1, d), dtype=np.int64)
    I_indices = np.concatenate([zero, I_indices_raw], axis=0)
    N_I = I_indices.shape[0]
    print(f"  Moment matrix size: {N_I} x {N_I}")

    I_tuples = [tuple(int(x) for x in row) for row in I_indices]

    # Group matrix entries by their difference n = I[i] - I[j]
    diff_groups: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for i, mi in enumerate(I_tuples):
        for j, mj in enumerate(I_tuples):
            n = tuple(mi[k] - mj[k] for k in range(d))
            diff_groups[n].append((i, j))

    # Identify unique differences and canonical positive set
    all_diffs = sorted(diff_groups.keys())
    zero_tuple = tuple(0 for _ in range(d))

    # Create real decision variables for canonical positive frequencies
    # c_n = alpha_n + i * beta_n for positive n
    # c_{-n} = alpha_n - i * beta_n
    # c_0 = 1 (fixed)
    positive_diffs = [n for n in all_diffs if n != zero_tuple and _canonical_positive(n)]
    N_pos = len(positive_diffs)
    pos_to_idx = {n: k for k, n in enumerate(positive_diffs)}

    alpha = cp.Variable(N_pos, name="alpha")  # Re(c_n)
    beta = cp.Variable(N_pos, name="beta")    # Im(c_n)

    # Build moment matrix using sparse basis matrices.
    #
    # M_real = M_0_real + sum_k alpha[k] * A_real[k]
    # M_imag = M_0_imag + sum_k beta[k] * B_imag[k]
    # (A_imag and B_real are identically zero by construction.)
    #
    # We flatten each N_I x N_I matrix to a vector of length NN = N_I^2,
    # then store the per-variable contributions as sparse matrices
    # A_real_sp (NN x N_pos) and B_imag_sp (NN x N_pos).

    NN = N_I * N_I
    M_0_real = np.zeros((N_I, N_I), dtype=np.float64)
    M_0_imag = np.zeros((N_I, N_I), dtype=np.float64)

    # Collect sparse entries: (flat_index, variable_index, value)
    a_rows: list[int] = []
    a_cols: list[int] = []
    a_vals: list[float] = []
    b_rows: list[int] = []
    b_cols: list[int] = []
    b_vals: list[float] = []

    for n, pairs in diff_groups.items():
        if n == zero_tuple:
            for i, j in pairs:
                M_0_real[i, j] = 1.0
            continue

        if _canonical_positive(n):
            k = pos_to_idx[n]
            for i, j in pairs:
                flat = i * N_I + j
                a_rows.append(flat)
                a_cols.append(k)
                a_vals.append(1.0)
                b_rows.append(flat)
                b_cols.append(k)
                b_vals.append(1.0)
        else:
            neg_n = tuple(-x for x in n)
            if neg_n in pos_to_idx:
                k = pos_to_idx[neg_n]
                for i, j in pairs:
                    flat = i * N_I + j
                    a_rows.append(flat)
                    a_cols.append(k)
                    a_vals.append(1.0)
                    b_rows.append(flat)
                    b_cols.append(k)
                    b_vals.append(-1.0)

    A_real_sp = sp.csc_matrix(
        (a_vals, (a_rows, a_cols)), shape=(NN, N_pos), dtype=np.float64
    )
    B_imag_sp = sp.csc_matrix(
        (b_vals, (b_rows, b_cols)), shape=(NN, N_pos), dtype=np.float64
    )

    nnz = A_real_sp.nnz
    density = nnz / (NN * N_pos) if (NN * N_pos) > 0 else 0.0
    print(
        f"  Basis matrices: {NN}x{N_pos}, nnz={nnz:,}, "
        f"density={density:.4%}, "
        f"sparse={nnz * 16 / 1e6:.1f} MB vs dense={NN * N_pos * 8 / 1e9:.1f} GB"
    )

    # Build CVXPY expressions using sparse matmul
    M_real_flat = M_0_real.ravel() + A_real_sp @ alpha
    M_imag_flat = M_0_imag.ravel() + B_imag_sp @ beta
    M_real_expr = cp.reshape(M_real_flat, (N_I, N_I))
    M_imag_expr = cp.reshape(M_imag_flat, (N_I, N_I))

    # Real PSD formulation: [M_real, -M_imag; M_imag, M_real] >> 0
    top = cp.hstack([M_real_expr, -M_imag_expr])
    bottom = cp.hstack([M_imag_expr, M_real_expr])
    M_block = cp.vstack([top, bottom])

    constraints = [M_block >> 0]

    # Objective: minimize E_μ = c0 + sum_m [a_m * Re(c_m) + b_m * Im(c_m)]
    obj_alpha_coeffs = np.zeros(N_pos, dtype=np.float64)
    obj_beta_coeffs = np.zeros(N_pos, dtype=np.float64)
    obj_constant = energy_surface.c0

    for m_tuple, a_m in energy_surface.cos_coeffs.items():
        if m_tuple in pos_to_idx:
            k = pos_to_idx[m_tuple]
            obj_alpha_coeffs[k] += a_m
        else:
            neg_m = tuple(-x for x in m_tuple)
            if neg_m in pos_to_idx:
                k = pos_to_idx[neg_m]
                obj_alpha_coeffs[k] += a_m

    for m_tuple, b_m in energy_surface.sin_coeffs.items():
        if m_tuple in pos_to_idx:
            k = pos_to_idx[m_tuple]
            obj_beta_coeffs[k] += b_m
        else:
            neg_m = tuple(-x for x in m_tuple)
            if neg_m in pos_to_idx:
                k = pos_to_idx[neg_m]
                obj_beta_coeffs[k] -= b_m

    objective = cp.Minimize(
        obj_constant + obj_alpha_coeffs @ alpha + obj_beta_coeffs @ beta
    )

    problem = cp.Problem(objective, constraints)
    print(f"  Solving SDP ({N_pos} moment variables, block matrix {2*N_I}x{2*N_I})...")
    problem.solve(solver=cp.SCS, verbose=False, max_iters=10000, eps=1e-7)

    status = str(problem.status)
    sdp_bound = float(problem.value) if problem.value is not None else float("nan")
    print(f"  SDP status: {status}, bound: {sdp_bound:.4f}")

    if alpha.value is None or beta.value is None:
        print("  WARNING: SDP solver failed to find a solution")
        s_star = lifting_map.eval(energy_surface.global_min)
        return s_star, sdp_bound, status

    alpha_val = np.asarray(alpha.value, dtype=np.float64).ravel()
    beta_val = np.asarray(beta.value, dtype=np.float64).ravel()

    # Extract s*_SDP in lifting map format
    s_star = np.zeros(lifting_map.D, dtype=np.float64)
    for i, m_row in enumerate(lifting_map.frequency_matrix):
        m_tuple = tuple(int(x) for x in m_row)
        if m_tuple in pos_to_idx:
            k = pos_to_idx[m_tuple]
            s_star[i] = alpha_val[k]
            s_star[lifting_map.N_freq + i] = beta_val[k]
        else:
            neg_m = tuple(-x for x in m_tuple)
            if neg_m in pos_to_idx:
                k = pos_to_idx[neg_m]
                s_star[i] = alpha_val[k]
                s_star[lifting_map.N_freq + i] = -beta_val[k]

    return s_star.astype(np.float32), sdp_bound, status


def validate_sdp_solution(
    s_star_sdp: np.ndarray,
    lifting_map: LiftingMap,
    energy_surface,
) -> dict:
    """Compare s*_SDP against the known global minimum (for validation only).

    Uses energy_surface.global_min which is known from the JSON.
    NOT used during oracle construction.
    """
    s_star_true = lifting_map.eval(energy_surface.global_min)
    distance = float(np.linalg.norm(s_star_sdp - s_star_true))

    c_vec = lifting_map.energy_as_linear(energy_surface)
    sdp_energy = float(np.dot(c_vec, s_star_sdp.astype(np.float64)) + energy_surface.c0)
    true_energy = float(energy_surface.energy(energy_surface.global_min))

    print(f"  ||s*_SDP - F(z*)||: {distance:.6f}")
    print(f"  SDP energy: {sdp_energy:.4f} kJ/mol")
    print(f"  True global min energy: {true_energy:.4f} kJ/mol")
    print(f"  Gap: {sdp_energy - true_energy:.4f} kJ/mol")

    return {
        "distance": distance,
        "sdp_energy": sdp_energy,
        "true_energy": true_energy,
        "gap": sdp_energy - true_energy,
    }
