import numpy as np


ORACLE_MODES = {
    "fixed_cipher",
    "fresh_cipher",
    "fst_cipher",
    "random_message",
    "no_oracle",
}


class Oracle:
    def __init__(
        self,
        d: int,
        mode: str,
        sigma_size: int,
        seed: int,
        lie_prob: float = 0.0,
        fst_k: int = 4,
    ):
        if mode not in ORACLE_MODES:
            raise ValueError(f"Unsupported oracle_mode: {mode}")
        if sigma_size < 1:
            raise ValueError("sigma_size must be >= 1")
        if lie_prob < 0.0 or lie_prob > 1.0:
            raise ValueError("lie_prob must be in [0, 1]")
        if fst_k < 1:
            raise ValueError("fst_k must be >= 1")

        self.d = d
        self.mode = mode
        self.sigma_size = sigma_size
        self.lie_prob = float(lie_prob)
        self.fst_k = int(fst_k)
        self.rng = np.random.default_rng(seed)
        self.fixed_mapping = self._sample_mapping(self.rng)
        self.fst_lookup = self._sample_fst_lookup(self.rng)

    @property
    def token_vocab_size(self) -> int:
        return self.sigma_size

    @property
    def true_inverse_mapping(self) -> np.ndarray | None:
        if self.mode != "fixed_cipher":
            return None
        if self.sigma_size != self.d:
            return None
        inverse = np.empty(self.sigma_size, dtype=np.int64)
        for action, token in enumerate(self.fixed_mapping):
            inverse[int(token)] = action
        return inverse

    def _sample_mapping(self, rng: np.random.Generator) -> np.ndarray:
        if self.sigma_size == self.d:
            return rng.permutation(self.d).astype(np.int64)

        if self.sigma_size == 1:
            return np.zeros(self.d, dtype=np.int64)

        base = np.arange(self.d, dtype=np.int64) % self.sigma_size
        rng.shuffle(base)
        return base

    def _episode_mapping(self, rng: np.random.Generator) -> np.ndarray:
        if self.mode == "fresh_cipher":
            return self._sample_mapping(rng)
        return self.fixed_mapping

    def _sample_fst_lookup(self, rng: np.random.Generator) -> np.ndarray:
        table = [rng.permutation(self.sigma_size).astype(np.int64) for _ in range(self.fst_k)]
        return np.stack(table, axis=0)

    def _apply_fst(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.size == 0:
            return token_ids
        state_ids = np.arange(token_ids.shape[0], dtype=np.int64) % self.fst_k
        return self.fst_lookup[state_ids, token_ids]

    def encode_actions(
        self,
        action_indices: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        path_len = int(action_indices.shape[0])

        if self.mode == "no_oracle":
            return np.empty(0, dtype=np.int64)

        if self.mode == "random_message":
            return rng.integers(0, self.sigma_size, size=path_len, dtype=np.int64)

        mapping = self._episode_mapping(rng)
        encoded = mapping[action_indices]
        if self.mode == "fixed_cipher" and self.lie_prob > 0.0 and path_len > 0:
            lie_mask = rng.random(path_len) < self.lie_prob
            if np.any(lie_mask):
                encoded = encoded.copy()
                encoded[lie_mask] = rng.integers(
                    0,
                    self.sigma_size,
                    size=int(np.count_nonzero(lie_mask)),
                    dtype=np.int64,
                )
        if self.mode == "fst_cipher":
            return self._apply_fst(encoded)
        return encoded
