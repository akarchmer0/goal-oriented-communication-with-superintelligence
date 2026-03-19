"""SDP relaxation for computing the oracle target s*_SDP.

Solves: min c_energy^T s  subject to M(s) >= 0 (PSD)
where M(s) is the trigonometric moment matrix parametrized by s.

The solution s*_SDP approximates F(z*) where z* is the true global minimum,
WITHOUT requiring knowledge of z*.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

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
    I_to_idx = {m: i for i, m in enumerate(I_tuples)}

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

    # Build moment matrix as sum of basis matrices weighted by variables
    # M = M_0 + sum_k alpha_k * A_k + sum_k beta_k * B_k
    # where A_k, B_k are complex basis matrices

    # For efficiency, build the real and imaginary parts separately
    # M_real = Re(M) is symmetric, M_imag = Im(M) is antisymmetric
    # M = M_real + i * M_imag
    # M >> 0 iff [M_real, -M_imag; M_imag, M_real] >> 0

    # Build coefficient arrays for each variable
    # M_0_real[i,j] = Re(c_n) where n = I[i]-I[j] and n=0 => 1
    # For n=0: c_0 = 1, so M_0[i,j] = 1 for all (i,j) in diff_groups[0]
    M_0_real = np.zeros((N_I, N_I), dtype=np.float64)
    M_0_imag = np.zeros((N_I, N_I), dtype=np.float64)

    # For each positive diff k:
    # A_k_real[i,j] = 1 if n=pos_k, or 1 if n=neg(pos_k), else 0
    # A_k_imag[i,j] = 0 if n=pos_k... need to be careful
    #
    # c_n = alpha + i*beta (for positive n)
    # c_{-n} = alpha - i*beta
    # So for entry (i,j) with diff n = positive: contributes alpha*1 + beta*i
    #    Re part: alpha * 1 + beta * 0 = alpha
    #    Im part: alpha * 0 + beta * 1 = beta
    # For entry (i,j) with diff n = negative of positive:
    #    c_{-n} = alpha - i*beta
    #    Re part: alpha
    #    Im part: -beta

    A_real = np.zeros((N_pos, N_I, N_I), dtype=np.float64)
    A_imag = np.zeros((N_pos, N_I, N_I), dtype=np.float64)
    B_real = np.zeros((N_pos, N_I, N_I), dtype=np.float64)
    B_imag = np.zeros((N_pos, N_I, N_I), dtype=np.float64)

    for n, pairs in diff_groups.items():
        if n == zero_tuple:
            for i, j in pairs:
                M_0_real[i, j] = 1.0
            continue

        if _canonical_positive(n):
            k = pos_to_idx[n]
            for i, j in pairs:
                A_real[k, i, j] = 1.0  # alpha contributes to Re
                B_imag[k, i, j] = 1.0  # beta contributes to Im
        else:
            neg_n = tuple(-x for x in n)
            if neg_n in pos_to_idx:
                k = pos_to_idx[neg_n]
                for i, j in pairs:
                    A_real[k, i, j] = 1.0   # alpha contributes to Re (same)
                    B_imag[k, i, j] = -1.0  # -beta contributes to Im

    # Build M_real and M_imag as CVXPY expressions
    # M_real = M_0_real + sum_k alpha[k] * A_real[k] + sum_k beta[k] * B_real[k]
    # M_imag = M_0_imag + sum_k alpha[k] * A_imag[k] + sum_k beta[k] * B_imag[k]
    # (A_imag and B_real are all zero in our construction)

    M_real_expr = M_0_real
    M_imag_expr = M_0_imag
    for k in range(N_pos):
        if np.any(A_real[k] != 0):
            M_real_expr = M_real_expr + alpha[k] * A_real[k]
        if np.any(A_imag[k] != 0):
            M_imag_expr = M_imag_expr + alpha[k] * A_imag[k]
        if np.any(B_real[k] != 0):
            M_real_expr = M_real_expr + beta[k] * B_real[k]
        if np.any(B_imag[k] != 0):
            M_imag_expr = M_imag_expr + beta[k] * B_imag[k]

    # Real PSD formulation: [M_real, -M_imag; M_imag, M_real] >> 0
    top = cp.hstack([M_real_expr, -M_imag_expr])
    bottom = cp.hstack([M_imag_expr, M_real_expr])
    M_block = cp.vstack([top, bottom])

    constraints = [M_block >> 0]

    # Objective: minimize E = c0 + sum_m [a_m * (2*Re(c_m)) + b_m * (-2*Im(c_m))]
    # The moments c_m here are the SDP decision variables.
    # For positive m: Re(c_m) = alpha[pos_idx[m]], Im(c_m) = beta[pos_idx[m]]
    # For negative m: Re(c_m) = alpha[pos_idx[-m]], Im(c_m) = -beta[pos_idx[-m]]
    # So: a_m * 2*Re(c_m) + b_m * (-2)*Im(c_m)
    #   = for positive m: 2*a_m*alpha[k] - 2*b_m*beta[k]
    #   = for negative m (let m' = -m which is positive):
    #     a_m * 2*alpha[pos_idx[m']] + b_m * (-2)*(-beta[pos_idx[m']])
    #     = 2*a_m*alpha[k'] + 2*b_m*beta[k']
    # But also a_{-m} = a_m (cos is even) and b_{-m} = -b_m (sin is odd) in standard Fourier
    # Actually NO: our frequencies m are general multi-indices, and cos(m.z) and cos((-m).z)
    # are the same function. So the energy should not have both m and -m as separate terms.
    # Let me re-derive.
    #
    # E = c0 + sum_{m in energy_freqs} [a_m * cos(m.z) + b_m * sin(m.z)]
    # E_μ = c0 + sum_m [a_m * E_μ[cos(m.z)] + b_m * E_μ[sin(m.z)]]
    # E_μ[cos(m.z)] = Re(c_m + c_{-m}) = 2*Re(c_m) if we define c_m as half-moment
    #
    # Actually, with our definition c_m = E_μ[e^{im.z}]:
    # E_μ[cos(m.z)] = Re(E_μ[e^{im.z}]) + Re(E_μ[e^{-im.z}]) / ... no.
    # cos(m.z) = (e^{im.z} + e^{-im.z})/2
    # E_μ[cos(m.z)] = (c_m + c_{-m})/2 = (c_m + conj(c_m))/2 = Re(c_m)
    # sin(m.z) = (e^{im.z} - e^{-im.z})/(2i)
    # E_μ[sin(m.z)] = (c_m - c_{-m})/(2i) = (c_m - conj(c_m))/(2i) = Im(c_m)/1 ... let me redo
    # sin(m.z) = (e^{im.z} - e^{-im.z})/(2i)
    # E_μ[sin(m.z)] = (c_m - conj(c_m))/(2i) = 2*Im(c_m)/(2) = Im(c_m)
    # Wait: c_m - conj(c_m) = 2i*Im(c_m), so (c_m - conj(c_m))/(2i) = Im(c_m)
    #
    # So: E_μ[cos(m.z)] = Re(c_m), E_μ[sin(m.z)] = Im(c_m)
    # => E_μ = c0 + sum_m [a_m * Re(c_m) + b_m * Im(c_m)]

    obj_alpha_coeffs = np.zeros(N_pos, dtype=np.float64)
    obj_beta_coeffs = np.zeros(N_pos, dtype=np.float64)
    obj_constant = energy_surface.c0

    for m_tuple, a_m in energy_surface.cos_coeffs.items():
        if m_tuple in pos_to_idx:
            k = pos_to_idx[m_tuple]
            obj_alpha_coeffs[k] += a_m  # a_m * Re(c_m) = a_m * alpha[k]
        else:
            neg_m = tuple(-x for x in m_tuple)
            if neg_m in pos_to_idx:
                k = pos_to_idx[neg_m]
                # Re(c_m) = Re(conj(c_{-m})) = Re(c_{-m}) = alpha[k]
                obj_alpha_coeffs[k] += a_m

    for m_tuple, b_m in energy_surface.sin_coeffs.items():
        if m_tuple in pos_to_idx:
            k = pos_to_idx[m_tuple]
            obj_beta_coeffs[k] += b_m  # b_m * Im(c_m) = b_m * beta[k]
        else:
            neg_m = tuple(-x for x in m_tuple)
            if neg_m in pos_to_idx:
                k = pos_to_idx[neg_m]
                # Im(c_m) = Im(conj(c_{-m})) = -Im(c_{-m}) = -beta[k]
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
        # Return lifting_map.eval at the known global min as fallback
        s_star = lifting_map.eval(energy_surface.global_min)
        return s_star, sdp_bound, status

    alpha_val = np.asarray(alpha.value, dtype=np.float64).ravel()
    beta_val = np.asarray(beta.value, dtype=np.float64).ravel()

    # Extract s*_SDP in lifting map format
    # For each frequency m_i in the lifting map:
    #   s_cos[i] = E_μ[cos(m_i.z)] = Re(c_{m_i})
    #   s_sin[i] = E_μ[sin(m_i.z)] = Im(c_{m_i})
    s_star = np.zeros(lifting_map.D, dtype=np.float64)
    for i, m_row in enumerate(lifting_map.frequency_matrix):
        m_tuple = tuple(int(x) for x in m_row)
        if m_tuple in pos_to_idx:
            k = pos_to_idx[m_tuple]
            s_star[i] = alpha_val[k]                           # cos part = Re(c_m)
            s_star[lifting_map.N_freq + i] = beta_val[k]       # sin part = Im(c_m)
        else:
            neg_m = tuple(-x for x in m_tuple)
            if neg_m in pos_to_idx:
                k = pos_to_idx[neg_m]
                s_star[i] = alpha_val[k]                        # Re(c_m) = Re(conj(c_{-m})) = alpha
                s_star[lifting_map.N_freq + i] = -beta_val[k]   # Im(c_m) = -Im(c_{-m}) = -beta
            # else: frequency not in SDP variables, leave as 0

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
