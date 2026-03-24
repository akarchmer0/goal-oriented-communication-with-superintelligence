"""Learned lifting map: encoder F (MLP) + decoder G (ICNN).

F: T^d -> R^D  (neural encoder, torus-aware via sin/cos input features)
G: R^D -> R    (input-convex neural network, convex in s by construction)

Trained so that G(F(z)) ≈ E_true(z).  Since G is convex, ∇_s G provides a
valid descent oracle in the lifted space — the RL agent receives this gradient
as its token, exactly as in the Fourier-lift setup.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class _ICNNLayer(nn.Module):
    """Single hidden layer of an input-convex neural network.

    h_{l+1} = activation( softplus(W_raw) @ h_l + U @ s + b )

    The hidden-to-hidden weight is kept non-negative via softplus
    reparameterization so that convexity in s is preserved.
    """

    def __init__(self, hidden_in: int, hidden_out: int, input_dim: int):
        super().__init__()
        self.W_raw = nn.Parameter(torch.empty(hidden_out, hidden_in))
        self.U = nn.Linear(input_dim, hidden_out)
        nn.init.kaiming_normal_(self.W_raw)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        W_pos = F.softplus(self.W_raw)
        return F.relu(F.linear(h, W_pos) + self.U(s))


class ICNN(nn.Module):
    """Input-Convex Neural Network (Amos et al. 2017).

    G(s) is convex in s by construction:
      - First layer is an unconstrained linear + activation.
      - Subsequent layers use non-negative hidden-to-hidden weights
        (softplus reparameterization) plus unconstrained passthrough
        from the input s.
      - Output layer follows the same pattern.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.first = nn.Linear(input_dim, hidden_dims[0])
        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.layers.append(_ICNNLayer(hidden_dims[i - 1], hidden_dims[i], input_dim))
        # Output: scalar
        self.out_W_raw = nn.Parameter(torch.empty(1, hidden_dims[-1]))
        self.out_U = nn.Linear(input_dim, 1)
        nn.init.kaiming_normal_(self.out_W_raw)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.first(s))
        for layer in self.layers:
            h = layer(h, s)
        W_pos = F.softplus(self.out_W_raw)
        out = F.linear(h, W_pos) + self.out_U(s)
        return out.squeeze(-1)


class Encoder(nn.Module):
    """Encoder F: T^d -> R^D.

    Uses [cos(z), sin(z)] as input features to respect torus topology.
    """

    def __init__(self, d: int, D: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 2 * d  # [cos(z), sin(z)]
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, D))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        features = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
        return self.net(features)


class LearnedLiftOracle:
    """Bundles a trained encoder F and ICNN decoder G.

    Drop-in replacement for the (LiftingMap, s_star_sdp) pair used by the env.

    Provides:
        eval(z)               -> F(z),  shape (D,)
        objective_from_z(z)   -> G(F(z)) - G(s*),  shifted so minimum ≈ 0
        gradient_s_from_z(z)  -> ∇_s G(F(z)),  oracle gradient in lifted space
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: ICNN,
        d: int,
        D: int,
        s_star: np.ndarray,
        g_star: float,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.d = int(d)
        self.D = int(D)
        self.s_star = np.asarray(s_star, dtype=np.float32).ravel()
        self.g_star = float(g_star)

    # -- numpy interface for the env ------------------------------------------

    def eval(self, z: np.ndarray) -> np.ndarray:
        """F(z) -> shape (D,)."""
        z_t = torch.as_tensor(z, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            s = self.encoder(z_t)
        return s.squeeze(0).numpy().astype(np.float32)

    def eval_batch(self, z_batch: np.ndarray) -> np.ndarray:
        z_t = torch.as_tensor(z_batch, dtype=torch.float32)
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        with torch.no_grad():
            s = self.encoder(z_t)
        return s.numpy().astype(np.float32)

    def objective_from_z(self, z: np.ndarray) -> float:
        """G(F(z)) - G(s*), so the minimum is near 0."""
        z_t = torch.as_tensor(z, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            s = self.encoder(z_t)
            g = self.decoder(s)
        return float(g.item()) - self.g_star

    def gradient_s_from_z(self, z: np.ndarray) -> np.ndarray:
        """∇_s G  evaluated at s = F(z).  Shape (D,)."""
        z_t = torch.as_tensor(z, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            s = self.encoder(z_t)
        s = s.detach().requires_grad_(True)
        with torch.enable_grad():
            g = self.decoder(s)
            g.backward()
        assert s.grad is not None
        return s.grad.squeeze(0).detach().numpy().astype(np.float32)

    # -- training factory -----------------------------------------------------

    @staticmethod
    def train_from_energy_surface(
        energy_surface,
        d: int,
        D: int,
        *,
        encoder_hidden_dims: tuple[int, ...] = (256, 256),
        decoder_hidden_dims: tuple[int, ...] = (256, 256),
        n_train: int = 100_000,
        n_epochs: int = 800,
        batch_size: int = 2048,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> "LearnedLiftOracle":
        """Train encoder + ICNN decoder on energy surface samples."""
        rng = np.random.default_rng(seed)
        two_pi = 2.0 * np.pi

        # ---- sample training data -------------------------------------------
        z_train = rng.uniform(0, two_pi, size=(n_train, d)).astype(np.float32)
        e_train = energy_surface.energy_batch(z_train).astype(np.float32)
        min_energy_train_samples = float(np.min(e_train))
        print(
            f"  [learned_lift] min energy over {n_train} training samples: "
            f"{min_energy_train_samples:.6f}"
        )

        z_t = torch.as_tensor(z_train)
        e_t = torch.as_tensor(e_train)

        n_val = int(min(4096, max(256, n_train // 5)))
        z_val = rng.uniform(0, two_pi, size=(n_val, d)).astype(np.float32)
        e_val = energy_surface.energy_batch(z_val).astype(np.float32)
        z_val_t = torch.as_tensor(z_val)
        e_val_t = torch.as_tensor(e_val)

        encoder = Encoder(d, D, list(encoder_hidden_dims))
        decoder = ICNN(D, list(decoder_hidden_dims))

        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        def _val_metrics() -> tuple[float, float]:
            encoder.eval()
            decoder.eval()
            mse_sum = 0.0
            mae_sum = 0.0
            with torch.no_grad():
                for vstart in range(0, n_val, batch_size):
                    vend = min(vstart + batch_size, n_val)
                    s_b = encoder(z_val_t[vstart:vend])
                    g_b = decoder(s_b)
                    t_b = e_val_t[vstart:vend]
                    mse_sum += F.mse_loss(g_b, t_b, reduction="sum").item()
                    mae_sum += F.l1_loss(g_b, t_b, reduction="sum").item()
            encoder.train()
            decoder.train()
            nv = float(max(1, n_val))
            return mse_sum / nv, mae_sum / nv

        # ---- joint training -------------------------------------------------
        best_train_mse = float("inf")
        best_val_mse = float("inf")
        best_train_mae = float("inf")
        best_val_mae = float("inf")
        pbar = tqdm(
            range(n_epochs),
            desc="learned_lift",
            unit="epoch",
            leave=False,
        )
        for epoch in pbar:
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            train_mae_sum = 0.0
            n_batches = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                s = encoder(z_t[idx])
                g = decoder(s)
                target = e_t[idx]
                loss = F.mse_loss(g, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_mae_sum += F.l1_loss(g, target, reduction="sum").item()
                n_batches += 1
            scheduler.step()
            avg_train_mse = epoch_loss / max(1, n_batches)
            avg_train_mae = train_mae_sum / float(max(1, n_train))
            avg_val_mse, avg_val_mae = _val_metrics()
            if avg_train_mse < best_train_mse:
                best_train_mse = avg_train_mse
            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
            if avg_train_mae < best_train_mae:
                best_train_mae = avg_train_mae
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
            pbar.set_postfix(
                train_mse=f"{avg_train_mse:.4f}",
                train_mae=f"{avg_train_mae:.4f}",
                val_mse=f"{avg_val_mse:.4f}",
                val_mae=f"{avg_val_mae:.4f}",
                refresh=True,
            )

        # ---- find s* = F(z*) constrained to image of encoder ----------------
        # Optimizing s freely in R^D leaves the encoder's manifold — G
        # extrapolates wildly off-manifold.  Instead, optimise over z-space
        # so s* always lies on image(F), then refine with a small grid search.
        z_star_np = np.asarray(energy_surface.global_min, dtype=np.float32)
        z_opt = torch.as_tensor(z_star_np).reshape(1, -1).clone().detach().requires_grad_(True)
        opt_z = torch.optim.Adam([z_opt], lr=1e-3)
        for _ in range(3000):
            s_cand = encoder(z_opt)
            g_cand = decoder(s_cand)
            opt_z.zero_grad(set_to_none=True)
            g_cand.backward()
            opt_z.step()

        with torch.no_grad():
            s_star = encoder(z_opt).squeeze(0).numpy().astype(np.float32)
            g_star = float(decoder(torch.as_tensor(s_star).unsqueeze(0)).item())

        encoder.eval()
        decoder.eval()

        # ---- report on fixed val set (same as training monitor) ------------
        with torch.no_grad():
            s_val = encoder(z_val_t)
            g_val = decoder(s_val).numpy()
        residual = np.abs(g_val - e_val)
        print(
            f"  [learned_lift] done  best_train_mse={best_train_mse:.6f}  best_val_mse={best_val_mse:.6f}  "
            f"best_train_mae={best_train_mae:.6f}  best_val_mae={best_val_mae:.6f}  "
            f"mean|G(F(z))-E(z)|={residual.mean():.6f}  max={residual.max():.6f}  "
            f"G(s*)={g_star:.6f}  E(z*)={float(energy_surface.global_min_energy):.6f}"
        )

        return LearnedLiftOracle(
            encoder=encoder,
            decoder=decoder,
            d=d,
            D=D,
            s_star=s_star,
            g_star=g_star,
        )
