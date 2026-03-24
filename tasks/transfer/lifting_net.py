"""Meta-learned lifting map F: R^d -> R^D with differentiable quadratic fitting.

The lifting map is trained offline so that energy surfaces are well-approximated
by quadratic functions in the lifted space. Two modes are supported:

- **Full PSD**: Q(s) = s^T A s + b^T s + c  with A PSD (D(D+1)/2 + D + 1 params)
- **Diagonal**:  Q(s) = a·(s⊙s) + b·s + c  with a >= 0   (2D + 1 params)

The diagonal mode scales to large D by avoiding the O(D²) matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LiftingNet(nn.Module):
    """Neural network lifting map F: T^d -> R^D.

    Input is periodic (angles on the torus), so we use cos/sin encoding
    to give the network a periodic input representation.
    """

    def __init__(
        self,
        visible_dim: int,
        lifting_dim: int,
        net_hidden_dims: tuple[int, ...] = (128, 128),
        output_activation: str = "tanh",
    ):
        super().__init__()
        self.visible_dim = visible_dim
        self.lifting_dim = lifting_dim

        input_dim = 2 * visible_dim  # cos(z), sin(z)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in net_hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, lifting_dim))
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "layernorm":
            layers.append(nn.LayerNorm(lifting_dim))
        elif output_activation != "none":
            raise ValueError(f"Unknown output_activation: {output_activation!r}")
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Lift points from visible space to D-dimensional space.

        Args:
            z: (..., d) angles in [0, 2pi)
        Returns:
            s: (..., D) lifted representation
        """
        x = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
# Quadratic parameterisation helpers (differentiable, batched)
# ──────────────────────────────────────────────────────────────────────


def n_quadratic_params(D: int) -> int:
    """Number of parameters in Q(s) = s^T A s + b^T s + c."""
    return D * (D + 1) // 2 + D + 1


def build_quadratic_features(s: torch.Tensor) -> torch.Tensor:
    """Build feature matrix Phi such that Q(s_i) = phi(s_i)^T theta.

    Off-diagonal products are doubled so that theta directly encodes the
    symmetric matrix A (upper triangle), b, and c.

    Args:
        s: (..., n, D)
    Returns:
        Phi: (..., n, p)  where p = D*(D+1)/2 + D + 1
    """
    *batch_dims, n, D = s.shape
    device, dtype = s.device, s.dtype

    idx_i, idx_j = torch.triu_indices(D, D, device=device)

    s_flat = s.reshape(-1, n, D)  # (B, n, D)
    quad = s_flat[..., idx_i] * s_flat[..., idx_j]  # (B, n, n_quad)

    # Double off-diagonal terms: s^T A s = sum A_ii s_i^2 + 2 sum_{i<j} A_ij s_i s_j
    is_offdiag = (idx_i != idx_j).to(dtype)  # (n_quad,)
    quad = quad * (1.0 + is_offdiag)

    ones = torch.ones(*s_flat.shape[:-1], 1, device=device, dtype=dtype)
    Phi = torch.cat([quad, s_flat, ones], dim=-1)
    return Phi.reshape(*batch_dims, n, -1)


def fit_quadratic(
    s: torch.Tensor,
    y: torch.Tensor,
    reg_lambda: float = 1e-4,
) -> torch.Tensor:
    """Fit Q via regularised least squares (differentiable).

    theta = (Phi^T Phi + lambda I)^{-1} Phi^T y

    Args:
        s: (..., n, D)  lifted training points
        y: (..., n)     target energies
    Returns:
        theta: (..., p)
    """
    Phi = build_quadratic_features(s)
    *batch_dims, n, p = Phi.shape

    Phi_flat = Phi.reshape(-1, n, p)
    y_flat = y.reshape(-1, n)

    PhiTPhi = torch.bmm(Phi_flat.transpose(-2, -1), Phi_flat)
    PhiTPhi = PhiTPhi + reg_lambda * torch.eye(p, device=s.device, dtype=s.dtype)
    PhiTy = torch.bmm(Phi_flat.transpose(-2, -1), y_flat.unsqueeze(-1)).squeeze(-1)

    theta = torch.linalg.solve(PhiTPhi, PhiTy)
    return theta.reshape(*batch_dims, p)


def extract_quadratic(
    theta: torch.Tensor, D: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (A, b, c) from parameter vector theta.

    Returns:
        A: (..., D, D) symmetric (NOT yet PSD-projected)
        b: (..., D)
        c: (...,)
    """
    n_quad = D * (D + 1) // 2
    A_upper_flat = theta[..., :n_quad]
    b = theta[..., n_quad : n_quad + D]
    c = theta[..., -1]

    idx_i, idx_j = torch.triu_indices(D, D, device=theta.device)
    A = theta.new_zeros(*theta.shape[:-1], D, D)
    A[..., idx_i, idx_j] = A_upper_flat
    A[..., idx_j, idx_i] = A_upper_flat
    return A, b, c


def project_psd(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Project symmetric A to PSD via eigenvalue clamping (differentiable)."""
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues_clamped = eigenvalues.clamp(min=eps)
    return eigenvectors @ torch.diag_embed(eigenvalues_clamped) @ eigenvectors.transpose(-2, -1)


def predict_quadratic(
    s: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Q(s) = s^T A s + b^T s + c.

    Args:
        s: (..., n, D)
        A: (..., D, D)
        b: (..., D)
        c: (...,)
    Returns:
        Q: (..., n)
    """
    As = torch.einsum("...ij,...nj->...ni", A, s)  # (..., n, D)
    quad_term = (s * As).sum(dim=-1)               # (..., n)
    lin_term = torch.einsum("...d,...nd->...n", b, s)
    return quad_term + lin_term + c.unsqueeze(-1)


def quadratic_gradient(
    s: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute nabla_s Q = 2As + b.

    Args:
        s: (..., D)
        A: (..., D, D)
        b: (..., D)
    Returns:
        grad: (..., D)
    """
    return 2.0 * torch.einsum("...ij,...j->...i", A, s) + b


def fit_and_project_quadratic(
    s: torch.Tensor,
    y: torch.Tensor,
    D: int,
    reg_lambda: float = 1e-4,
    psd_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit quadratic and project A to PSD.  Returns (A_psd, b, c)."""
    theta = fit_quadratic(s, y, reg_lambda=reg_lambda)
    A, b, c = extract_quadratic(theta, D)
    A_psd = project_psd(A, eps=psd_eps)
    return A_psd, b, c


# ──────────────────────────────────────────────────────────────────────
# Diagonal quadratic helpers  (2D + 1 parameters, scales to large D)
# ──────────────────────────────────────────────────────────────────────


def n_diagonal_params(D: int) -> int:
    """Number of parameters in diagonal Q(s) = a·(s⊙s) + b·s + c."""
    return 2 * D + 1


def build_diagonal_features(s: torch.Tensor) -> torch.Tensor:
    """Feature matrix for diagonal quadratic: [s², s, 1].

    Args:
        s: (..., n, D)
    Returns:
        Phi: (..., n, 2D+1)
    """
    ones = torch.ones(*s.shape[:-1], 1, device=s.device, dtype=s.dtype)
    return torch.cat([s * s, s, ones], dim=-1)


def fit_diagonal(
    s: torch.Tensor,
    y: torch.Tensor,
    reg_lambda: float = 1e-4,
) -> torch.Tensor:
    """Fit diagonal Q via regularised least squares (differentiable).

    Column-normalizes Phi before solving so that regularization acts uniformly
    regardless of feature scale.  This prevents ill-conditioning when many
    s² columns are correlated (common at initialization with large D).

    Args:
        s: (..., n, D)  lifted training points
        y: (..., n)     target energies
    Returns:
        theta: (..., 2D+1)
    """
    Phi = build_diagonal_features(s)
    *batch_dims, n, p = Phi.shape

    Phi_flat = Phi.reshape(-1, n, p)
    y_flat = y.reshape(-1, n)

    # Normalize columns for stable solve
    col_norms = Phi_flat.norm(dim=-2, keepdim=True).clamp(min=1e-8)  # (B, 1, p)
    Phi_n = Phi_flat / col_norms

    PhiTPhi = torch.bmm(Phi_n.transpose(-2, -1), Phi_n)
    PhiTPhi = PhiTPhi + reg_lambda * torch.eye(p, device=s.device, dtype=s.dtype)
    PhiTy = torch.bmm(Phi_n.transpose(-2, -1), y_flat.unsqueeze(-1)).squeeze(-1)

    theta_n = torch.linalg.solve(PhiTPhi, PhiTy)
    # Un-normalize: theta_i = theta_n_i / col_norm_i
    theta = theta_n / col_norms.squeeze(-2)
    return theta.reshape(*batch_dims, p)


def extract_diagonal(
    theta: torch.Tensor, D: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (a, b, c) from diagonal parameter vector.

    Returns:
        a: (..., D)  diagonal of A
        b: (..., D)
        c: (...,)
    """
    a = theta[..., :D]
    b = theta[..., D : 2 * D]
    c = theta[..., -1]
    return a, b, c


def project_psd_diagonal(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Project diagonal to PSD by clamping a >= eps."""
    return a.clamp(min=eps)


def predict_diagonal(
    s: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Evaluate diagonal Q(s) = a·(s⊙s) + b·s + c.

    Args:
        s: (..., n, D)
        a: (..., D)
        b: (..., D)
        c: (...,)
    Returns:
        Q: (..., n)
    """
    # Unsqueeze a, b to broadcast over the n dimension
    quad_term = (a.unsqueeze(-2) * s * s).sum(dim=-1)
    lin_term = (b.unsqueeze(-2) * s).sum(dim=-1)
    return quad_term + lin_term + c.unsqueeze(-1)


def diagonal_gradient(
    s: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute nabla_s Q = 2*a⊙s + b for diagonal quadratic.

    Args:
        s: (..., D)
        a: (..., D)
        b: (..., D)
    Returns:
        grad: (..., D)
    """
    return 2.0 * a * s + b


def fit_and_project_diagonal(
    s: torch.Tensor,
    y: torch.Tensor,
    D: int,
    reg_lambda: float = 1e-4,
    psd_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit diagonal quadratic and clamp a >= eps.  Returns (a_psd, b, c)."""
    theta = fit_diagonal(s, y, reg_lambda=reg_lambda)
    a, b, c = extract_diagonal(theta, D)
    a_psd = project_psd_diagonal(a, eps=psd_eps)
    return a_psd, b, c
