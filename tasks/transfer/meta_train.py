"""Meta-training of the neural lifting map F across random Fourier surfaces.

For each random surface, fits a quadratic (full PSD or diagonal) in F-space
via differentiable least squares.  The loss is MSE of the fitted quadratic on
held-out points, back-propagated through the lstsq solve into F's weights.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from tasks.alanine_dipeptide.lifting_map import enumerate_frequency_indices

from .lifting_net import (
    LiftingNet,
    fit_and_project_diagonal,
    fit_and_project_quadratic,
    n_diagonal_params,
    n_quadratic_params,
    predict_diagonal,
    predict_quadratic,
)

TWO_PI = 2.0 * np.pi


# ──────────────────────────────────────────────────────────────────────
# Lightweight surface energy evaluation (no EnergySurface object needed)
# ──────────────────────────────────────────────────────────────────────


def _filter_frequencies(freqs: np.ndarray, freq_sparsity: int) -> np.ndarray:
    """Keep only frequency vectors with at most `freq_sparsity` nonzero components."""
    if freq_sparsity <= 0:
        return freqs
    nnz = np.count_nonzero(freqs, axis=1)
    return freqs[nnz <= freq_sparsity]


class _SurfaceSampler:
    """Pre-computes shared frequency data for fast random-surface evaluation."""

    def __init__(
        self,
        d: int,
        K_energy: int,
        amplitude_scale: float = 5.0,
        freq_sparsity: int = 0,
    ):
        freqs = enumerate_frequency_indices(d, K_energy)
        freqs = _filter_frequencies(freqs, freq_sparsity)
        self.freq_f64 = freqs.astype(np.float64)
        norms = np.linalg.norm(self.freq_f64, axis=1)
        self.decay = amplitude_scale / (1.0 + norms)
        self.n_freq = freqs.shape[0]
        self.d = d

    def evaluate_batch(
        self,
        seed: int,
        z_batch: np.ndarray,
    ) -> np.ndarray:
        """Sample random coefficients and evaluate energy at z_batch.

        Args:
            seed: RNG seed for this surface's coefficients.
            z_batch: (n, d) points on the torus.
        Returns:
            energies: (n,) float32
        """
        rng = np.random.default_rng(seed)
        a = rng.normal(0.0, 1.0, self.n_freq) * self.decay
        b = rng.normal(0.0, 1.0, self.n_freq) * self.decay
        phases = z_batch.astype(np.float64) @ self.freq_f64.T  # (n, n_freq)
        energies = np.dot(np.cos(phases), a) + np.dot(np.sin(phases), b)
        return energies.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Meta-training loop
# ──────────────────────────────────────────────────────────────────────


def meta_train_lifting(
    *,
    visible_dim: int = 2,
    lifting_dim: int = 32,
    net_hidden_dims: tuple[int, ...] = (128, 128),
    K_energy: int = 10,
    amplitude_scale: float = 5.0,
    freq_sparsity: int = 0,
    n_surfaces_per_batch: int = 16,
    n_fit: int = 512,
    n_test: int = 256,
    reg_lambda: float = 1e-4,
    psd_eps: float = 1e-6,
    diagonal: bool = False,
    meta_lr: float = 1e-3,
    meta_steps: int = 5000,
    seed: int = 0,
    device: str = "cpu",
    print_interval: int = 200,
) -> tuple[LiftingNet, dict]:
    """Train lifting net F via meta-learning across random Fourier surfaces.

    Returns (lifting_net, stats_dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    p = n_diagonal_params(lifting_dim) if diagonal else n_quadratic_params(lifting_dim)
    mode_str = "diagonal" if diagonal else "full PSD"
    if n_fit < p:
        print(
            f"  WARNING: n_fit={n_fit} < n_params={p} for D={lifting_dim} ({mode_str}). "
            f"Increase --meta_n_fit or decrease --D."
        )

    lifting_net = LiftingNet(
        visible_dim=visible_dim,
        lifting_dim=lifting_dim,
        net_hidden_dims=net_hidden_dims,
    ).to(dev)

    optimizer = torch.optim.Adam(lifting_net.parameters(), lr=meta_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=meta_steps, eta_min=meta_lr * 0.01,
    )

    sampler = _SurfaceSampler(visible_dim, K_energy, amplitude_scale, freq_sparsity)
    print(
        f"  Meta-training: {mode_str}, D={lifting_dim}, p={p}, "
        f"n_freq={sampler.n_freq}, freq_sparsity={freq_sparsity or 'dense'}"
    )
    n_total = n_fit + n_test
    surf_seed_base = seed * 1_000_000

    losses: list[float] = []
    r2_scores: list[float] = []

    pbar = tqdm(range(meta_steps), desc="Meta-training F")
    for step in pbar:
        # Sample points (shared across surfaces for this step)
        z_np = rng.uniform(0.0, TWO_PI, size=(n_surfaces_per_batch, n_total, visible_dim)).astype(
            np.float32
        )

        # Evaluate energies per surface (numpy, no grad)
        y_np = np.empty((n_surfaces_per_batch, n_total), dtype=np.float32)
        for k in range(n_surfaces_per_batch):
            surf_seed = surf_seed_base + step * n_surfaces_per_batch + k
            y_np[k] = sampler.evaluate_batch(surf_seed, z_np[k])

        z_t = torch.as_tensor(z_np, device=dev)
        y_t = torch.as_tensor(y_np, device=dev)

        # Min-max normalize energies per surface to [0, 1]
        y_min = y_t.min(dim=-1, keepdim=True).values
        y_max = y_t.max(dim=-1, keepdim=True).values
        y_range = (y_max - y_min).clamp(min=1e-8)
        y_t = (y_t - y_min) / y_range

        # Forward: lift all points
        s_all = lifting_net(z_t)  # (K, n_total, D)

        # Split
        s_fit, s_test = s_all[:, :n_fit], s_all[:, n_fit:]
        y_fit, y_test = y_t[:, :n_fit], y_t[:, n_fit:]

        # Fit quadratic on fit points (differentiable)
        if diagonal:
            a_psd, b_vec, c_val = fit_and_project_diagonal(
                s_fit, y_fit, D=lifting_dim,
                reg_lambda=reg_lambda, psd_eps=psd_eps,
            )
            y_pred = predict_diagonal(s_test, a_psd, b_vec, c_val)
        else:
            A_psd, b_vec, c_val = fit_and_project_quadratic(
                s_fit, y_fit, D=lifting_dim,
                reg_lambda=reg_lambda, psd_eps=psd_eps,
            )
            y_pred = predict_quadratic(s_test, A_psd, b_vec, c_val)

        # Loss
        loss = ((y_pred - y_test) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lifting_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(float(loss.item()))

        with torch.no_grad():
            ss_res = ((y_pred - y_test) ** 2).sum()
            ss_tot = ((y_test - y_test.mean(dim=-1, keepdim=True)) ** 2).sum()
            r2 = 1.0 - float(ss_res) / max(float(ss_tot), 1e-12)
            r2_scores.append(r2)

        if step % print_interval == 0 or step == meta_steps - 1:
            window = min(print_interval, len(losses))
            avg_loss = float(np.mean(losses[-window:]))
            avg_r2 = float(np.mean(r2_scores[-window:]))
            pbar.set_postfix(loss=f"{avg_loss:.4f}", r2=f"{avg_r2:.4f}")

    pbar.close()
    stats = {
        "final_loss": float(np.mean(losses[-100:])) if losses else 0.0,
        "final_r2": float(np.mean(r2_scores[-100:])) if r2_scores else 0.0,
        "losses": losses,
        "r2_scores": r2_scores,
    }
    return lifting_net, stats
