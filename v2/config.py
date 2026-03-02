from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    n: int = 10000
    d: int = 8
    t_pool: int = 256
    n_env: int = 32
    algo: str = "ppo"
    train_steps: int = 300_000
    rollout_len: int = 64
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    token_embed_dim: int = 16
    sensing: str = "S0"
    reward_noise_std: float = 0.0
    oracle_mode: str = "fst_cipher"
    sigma_size: int = 32
    fst_k: int = 2
    lie_prob: float = 0.0
    max_horizon: int = 60
    s1_step_penalty: float = -0.01
    running_avg_window: int = 100
    eval_interval_episodes: int = 200
    eval_episodes: int = 200
    minibatches: int = 4
    seed: int = 0
    logdir: str = "v2/runs"
    run_name: str = ""
    device: str = "cpu"
    diagnostic_cipher: bool = True

    def resolve_run_dir(self) -> Path:
        base = Path(self.logdir)
        run_name = self.run_name or (
            f"n{self.n}_{self.oracle_mode}_{self.sensing}_sigma{self.sigma_size}_seed{self.seed}"
        )
        return base / run_name


@dataclass(frozen=True)
class RequestedExperimentsConfig:
    # Shared defaults for v2.run_requested_experiments and manifest/plot fallbacks.
    n: int = 10000
    n_values: tuple[int, ...] = (1000, 10000, 100000)
    sensing: str = "S0"
    train_steps: int = 1_000_000
    n_env: int = 32
    running_avg_window: int = 500
    seed: int = 0
    num_seeds: int = 5
    logdir: str = "v2/runs"
    manifest_path: str = ""
    lie_probs: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    noise_sigmas: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0)
    fst_k: int = TrainConfig().fst_k
    fst_k_values: tuple[int, ...] = (1, 2, 4, 8)
    oracle_mode_exp2: str = TrainConfig().oracle_mode
    # lie_prob currently affects fixed_cipher semantics in Oracle.encode_actions.
    oracle_mode_exp3: str = "fixed_cipher"
    oracle_mode_exp4: str = TrainConfig().oracle_mode
    d: int = 4
    d_values: tuple[int, ...] = (2, 4, 8, 16)
    t_pool: int = 256
    sigma_size: int = 32
    binary_signal_sensing: str = "S1"
    binary_signal_step_penalty: float = 0.0


@dataclass(frozen=True)
class PlotConfig:
    plotdir: str = "plots"
    max_points: int = 2500
