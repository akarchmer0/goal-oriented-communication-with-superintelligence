from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    task: str = "graph"
    n: int = 10000
    d: int = 8
    t_pool: int = 256
    spatial_hidden_dim: int = 10
    spatial_visible_dim: int = 2
    spatial_coord_limit: int = 8
    spatial_token_dim: int = 10
    spatial_token_noise_std: float = 0.0
    ppo_step_scale: float = 1.0
    spatial_step_size: float = 1.0
    spatial_success_threshold: float = 0.01
    spatial_enable_success_curriculum: bool = True
    spatial_success_curriculum_start: float = 0.5
    spatial_success_curriculum_trigger_rate: float = 0.8
    spatial_success_curriculum_decay: float = 0.2
    spatial_success_curriculum_min: float = 0.01
    spatial_basis_complexity: int = 3
    spatial_f_type: str = "FOURIER"
    spatial_policy_arch: str = "mlp"
    spatial_refresh_map_each_episode: bool = False
    spatial_fixed_start_target: bool = False
    spatial_plot_interval_episodes: int = 100
    spatial_enable_baselines: bool = True
    spatial_tune_baseline_lrs: bool = True
    spatial_early_stop_on_all_methods_success: bool = False
    spatial_baseline_lr_candidates: str = "0.001,0.003,0.01,0.03,0.05,0.1,0.2"
    spatial_baseline_lr_tune_tasks: int = 64
    spatial_optimization_curve_tasks: int = 100
    spatial_enable_optimization_curve_eval: bool = True
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
    lr_scheduler: str = "none"
    lr_min_factor: float = 0.1
    lr_warmup_updates: int = 0
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    token_embed_dim: int = 16
    sensing: str = "S0"
    reward_noise_std: float = 0.0
    oracle_mode: str = "fst_cipher"
    sigma_size: int = 32
    fst_k: int = 1
    lie_prob: float = 0.0
    max_horizon: int = 60
    s1_step_penalty: float = -0.01
    running_avg_window: int = 100
    save_metrics_interval_episodes: int = 500
    eval_interval_episodes: int = 200
    eval_episodes: int = 200
    minibatches: int = 4
    seed: int = 0
    logdir: str = "v2/runs"
    run_name: str = ""
    device: str = "cpu"
    diagnostic_cipher: bool = True
    enable_training_plots: bool = True

    def resolve_run_dir(self) -> Path:
        base = Path(self.logdir)
        if self.task == "graph":
            default_run_name = (
                f"graph_n{self.n}_{self.oracle_mode}_{self.sensing}_sigma{self.sigma_size}_"
                f"seed{self.seed}"
            )
        else:
            default_run_name = (
                f"spatial_D{self.spatial_hidden_dim}_vis{self.spatial_visible_dim}_"
                f"{self.oracle_mode}_{self.sensing}_k{self.spatial_token_dim}_"
                f"F{self.spatial_f_type}_seed{self.seed}"
            )
        run_name = self.run_name or default_run_name
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
    num_seeds: int = 30
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
    t_pool: int = 1024
    sigma_size: int = 32
    binary_signal_sensing: str = "S1"
    binary_signal_step_penalty: float = 0.0


@dataclass(frozen=True)
class PlotConfig:
    plotdir: str = "plots"
    max_points: int = 2500
