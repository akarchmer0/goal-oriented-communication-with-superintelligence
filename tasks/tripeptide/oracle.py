import numpy as np


ORACLE_MODES = {
    "convex_gradient",
    "visible_gradient",
    "linear_embedding",
    "fresh_linear_embedding",
    "random_message",
    "no_oracle",
}


class SpatialOracle:
    def __init__(
        self,
        hidden_dim: int,
        token_dim: int,
        mode: str,
        seed: int,
        token_noise_std: float = 0.0,
    ):
        if mode not in ORACLE_MODES:
            raise ValueError(f"Unsupported oracle mode: {mode}")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if token_dim < 1:
            raise ValueError("token_dim must be >= 1")
        if token_noise_std < 0.0:
            raise ValueError("token_noise_std must be >= 0")

        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.mode = mode
        self.token_noise_std = float(token_noise_std)
        self.rng = np.random.default_rng(seed)

        if self.mode == "convex_gradient" and self.token_dim != self.hidden_dim:
            raise ValueError(
                "For mode='convex_gradient', token_dim must equal hidden_dim so the oracle "
                "sends the full high-dimensional gradient"
            )

        self.fixed_embedding = self._sample_embedding(self.rng)

    def _sample_embedding(self, rng: np.random.Generator) -> np.ndarray:
        scale = np.sqrt(float(self.hidden_dim))
        return (rng.normal(size=(self.token_dim, self.hidden_dim)).astype(np.float32) / scale)

    def _step_embedding(self, rng: np.random.Generator) -> np.ndarray:
        if self.mode == "fresh_linear_embedding":
            return self._sample_embedding(rng)
        return self.fixed_embedding

    def encode_gradient(
        self,
        gradient: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=np.float32)
        if gradient.shape != (self.hidden_dim,):
            raise ValueError(
                f"gradient must have shape ({self.hidden_dim},), got {gradient.shape}"
            )

        if self.mode == "no_oracle":
            token = np.zeros(self.token_dim, dtype=np.float32)
        elif self.mode == "random_message":
            token = rng.normal(size=self.token_dim).astype(np.float32)
        elif self.mode == "visible_gradient":
            raise ValueError(
                "mode='visible_gradient' requires encode_visible_gradient(...) with visible-space gradient."
            )
        elif self.mode == "convex_gradient":
            token = gradient.copy()
        else:
            embedding = self._step_embedding(rng)
            token = np.matmul(embedding, gradient).astype(np.float32)

        if self.token_noise_std > 0.0:
            noise = rng.normal(loc=0.0, scale=self.token_noise_std, size=self.token_dim).astype(
                np.float32
            )
            token = token + noise
        return token

    def encode_visible_gradient(
        self,
        visible_gradient: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.mode != "visible_gradient":
            raise ValueError(
                f"encode_visible_gradient is only valid for mode='visible_gradient', got {self.mode!r}"
            )

        visible_gradient = np.asarray(visible_gradient, dtype=np.float32)
        if visible_gradient.shape != (self.token_dim,):
            raise ValueError(
                f"visible_gradient must have shape ({self.token_dim},), got {visible_gradient.shape}"
            )

        token = visible_gradient.copy()
        if self.token_noise_std > 0.0:
            noise = rng.normal(loc=0.0, scale=self.token_noise_std, size=self.token_dim).astype(
                np.float32
            )
            token = token + noise
        return token
