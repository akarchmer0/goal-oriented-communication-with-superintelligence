import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from v2.config import RequestedExperimentsConfig, TrainConfig
from v2.train import run_training

TRAIN_DEFAULTS = TrainConfig()
REQUESTED_DEFAULTS = RequestedExperimentsConfig()
DEFAULT_ORACLE_MODE_EXP2 = REQUESTED_DEFAULTS.oracle_mode_exp2
DEFAULT_ORACLE_MODE_EXP3 = REQUESTED_DEFAULTS.oracle_mode_exp3
DEFAULT_ORACLE_MODE_EXP4 = REQUESTED_DEFAULTS.oracle_mode_exp4
DEFAULT_FST_K = REQUESTED_DEFAULTS.fst_k


def _random_seeds(base_seed: int, num_seeds: int) -> list[int]:
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    if num_seeds == 1:
        return [int(base_seed)]

    rng = np.random.default_rng(int(base_seed))
    sampled = rng.choice(2_000_000_000, size=int(num_seeds), replace=False)
    return [int(value) for value in sampled]


def _train_condition(
    *,
    n: int,
    d: int = REQUESTED_DEFAULTS.d,
    sensing: str,
    train_steps: int,
    n_env: int,
    running_avg_window: int,
    seed: int,
    logdir: str,
    run_name: str,
    oracle_mode: str,
    sigma_size: int = 4,
    fst_k: int = DEFAULT_FST_K,
    lie_prob: float = 0.0,
    reward_noise_std: float = 0.0,
    s1_step_penalty: float = -0.01,
) -> dict:
    config = TrainConfig(
        n=n,
        d=int(d),
        t_pool=REQUESTED_DEFAULTS.t_pool,
        n_env=n_env,
        train_steps=train_steps,
        sensing=sensing,
        reward_noise_std=reward_noise_std,
        oracle_mode=oracle_mode,
        sigma_size=sigma_size,
        fst_k=fst_k,
        lie_prob=lie_prob,
        s1_step_penalty=s1_step_penalty,
        running_avg_window=running_avg_window,
        seed=seed,
        logdir=logdir,
        run_name=run_name,
    )
    return run_training(config)


def _train_condition_over_seeds(
    *,
    n: int,
    d: int = REQUESTED_DEFAULTS.d,
    sensing: str,
    train_steps: int,
    n_env: int,
    running_avg_window: int,
    seeds: list[int],
    logdir: str,
    run_name_prefix: str,
    oracle_mode: str,
    sigma_size: int = 4,
    fst_k: int = DEFAULT_FST_K,
    lie_prob: float = 0.0,
    reward_noise_std: float = 0.0,
    s1_step_penalty: float = -0.01,
) -> list[dict]:
    seed_runs: list[dict] = []
    for run_seed in seeds:
        run_name = f"{run_name_prefix}_seed{run_seed}"
        output = _train_condition(
            n=n,
            d=d,
            sensing=sensing,
            train_steps=train_steps,
            n_env=n_env,
            running_avg_window=running_avg_window,
            seed=int(run_seed),
            logdir=logdir,
            run_name=run_name,
            oracle_mode=oracle_mode,
            sigma_size=sigma_size,
            fst_k=fst_k,
            lie_prob=lie_prob,
            reward_noise_std=reward_noise_std,
            s1_step_penalty=s1_step_penalty,
        )
        seed_runs.append(
            {
                "seed": int(run_seed),
                "run_name": run_name,
                "run_dir": output["summary"]["run_dir"],
            }
        )
    return seed_runs


def run_experiment_1(
    *,
    n: int,
    sensing: str,
    train_steps: int,
    n_env: int,
    running_avg_window: int,
    seeds: list[int],
    logdir: str,
    fst_k: int = DEFAULT_FST_K,
) -> dict:
    conditions = [
        ("fixed_cipher", "fixed_cipher"),
        ("fresh_cipher", "fresh_cipher"),
        ("fst_cipher", "fst_cipher"),
        ("random_message", "random_message"),
        ("no_oracle", "no_message"),
    ]
    manifest_section: dict[str, list[dict]] = {}
    for mode, label in conditions:
        run_prefix = f"exp1_{sensing}_{mode}_n{n}_w{running_avg_window}"
        if mode == "fst_cipher":
            run_prefix = f"exp1_{sensing}_{mode}_k{fst_k}_n{n}_w{running_avg_window}"
        manifest_section[label] = _train_condition_over_seeds(
            n=n,
            sensing=sensing,
            train_steps=train_steps,
            n_env=n_env,
            running_avg_window=running_avg_window,
            seeds=seeds,
            logdir=logdir,
            run_name_prefix=run_prefix,
            oracle_mode=mode,
            sigma_size=REQUESTED_DEFAULTS.sigma_size,
            fst_k=fst_k,
            lie_prob=0.0,
            reward_noise_std=0.0,
            s1_step_penalty=TRAIN_DEFAULTS.s1_step_penalty,
        )
    return manifest_section


def run_experiment_2(
    *,
    n_values: list[int],
    sensing: str,
    train_steps: int,
    n_env: int,
    running_avg_window: int,
    seeds: list[int],
    logdir: str,
    oracle_mode: str = DEFAULT_ORACLE_MODE_EXP2,
    fst_k: int = DEFAULT_FST_K,
) -> dict:
    manifest_section: dict[str, list[dict]] = {}
    for n in sorted(n_values):
        run_prefix = f"exp2_{sensing}_{oracle_mode}_n{n}_w{running_avg_window}"
        manifest_section[str(n)] = _train_condition_over_seeds(
            n=n,
            sensing=sensing,
            train_steps=train_steps,
            n_env=n_env,
            running_avg_window=running_avg_window,
            seeds=seeds,
            logdir=logdir,
            run_name_prefix=run_prefix,
            oracle_mode=oracle_mode,
            sigma_size=REQUESTED_DEFAULTS.sigma_size,
            fst_k=fst_k,
            lie_prob=0.0,
            reward_noise_std=0.0,
            s1_step_penalty=TRAIN_DEFAULTS.s1_step_penalty,
        )
    return manifest_section


def run_experiment_3(
    *,
    n: int,
    sensing: str,
    train_steps: int,
    n_env: int,
    running_avg_window: int,
    seeds: list[int],
    logdir: str,
    lie_probs: list[float],
    oracle_mode: str = DEFAULT_ORACLE_MODE_EXP3,
    fst_k: int = DEFAULT_FST_K,
) -> dict:
    manifest_section: dict[str, list[dict]] = {}
    for p_lie in lie_probs:
        label = f"P(lie)={p_lie:g}"
        p_tag = str(p_lie).replace(".", "p")
        run_prefix = f"exp3_{sensing}_{oracle_mode}_plie{p_tag}_n{n}_w{running_avg_window}"
        manifest_section[label] = _train_condition_over_seeds(
            n=n,
            sensing=sensing,
            train_steps=train_steps,
            n_env=n_env,
            running_avg_window=running_avg_window,
            seeds=seeds,
            logdir=logdir,
            run_name_prefix=run_prefix,
            oracle_mode=oracle_mode,
            sigma_size=REQUESTED_DEFAULTS.sigma_size,
            fst_k=fst_k,
            lie_prob=float(p_lie),
            reward_noise_std=0.0,
            s1_step_penalty=TRAIN_DEFAULTS.s1_step_penalty,
        )
    return manifest_section


def run_experiment_4(
    *,
    n: int,
    sensing: str,
    train_steps: int,
    n_env: int,
    running_avg_window: int,
    seeds: list[int],
    logdir: str,
    noise_sigmas: list[float],
    binary_sensing: str = REQUESTED_DEFAULTS.binary_signal_sensing,
    oracle_mode: str = DEFAULT_ORACLE_MODE_EXP4,
    fst_k: int = DEFAULT_FST_K,
) -> dict:
    manifest_section: dict[str, list[dict]] = {}

    for sigma in sorted(set(float(value) for value in noise_sigmas)):
        if sigma < 0.0:
            raise ValueError("noise sigmas must be >= 0")
        label = f"σ={sigma:g}"
        sigma_tag = f"{sigma:g}".replace(".", "p")
        run_prefix = (
            f"exp4_{sensing}_{oracle_mode}_noise{sigma_tag}_n{n}_w{running_avg_window}"
        )
        manifest_section[label] = _train_condition_over_seeds(
            n=n,
            sensing=sensing,
            train_steps=train_steps,
            n_env=n_env,
            running_avg_window=running_avg_window,
            seeds=seeds,
            logdir=logdir,
            run_name_prefix=run_prefix,
            oracle_mode=oracle_mode,
            sigma_size=REQUESTED_DEFAULTS.sigma_size,
            fst_k=fst_k,
            lie_prob=0.0,
            reward_noise_std=float(sigma),
            s1_step_penalty=TRAIN_DEFAULTS.s1_step_penalty,
        )

    binary_label = "Binary Signal (no noise)"
    binary_prefix = (
        f"exp4_{binary_sensing}_{oracle_mode}_binary_no_noise_n{n}_w{running_avg_window}"
    )
    manifest_section[binary_label] = _train_condition_over_seeds(
        n=n,
        sensing=binary_sensing,
        train_steps=train_steps,
        n_env=n_env,
        running_avg_window=running_avg_window,
        seeds=seeds,
        logdir=logdir,
        run_name_prefix=binary_prefix,
        oracle_mode=oracle_mode,
        sigma_size=REQUESTED_DEFAULTS.sigma_size,
        fst_k=fst_k,
        lie_prob=0.0,
        reward_noise_std=0.0,
        s1_step_penalty=REQUESTED_DEFAULTS.binary_signal_step_penalty,
    )
    return manifest_section


def run_experiment_5(
    *,
    n: int,
    sensing: str,
    train_steps: int,
    n_env: int,
    running_avg_window: int,
    seeds: list[int],
    logdir: str,
    fst_k_values: list[int],
) -> dict:
    manifest_section: dict[str, list[dict]] = {}
    for k in sorted(set(int(value) for value in fst_k_values)):
        if k < 1:
            raise ValueError("fst_k values must be >= 1")
        label = f"{k} Hidden States"
        run_prefix = f"exp5_{sensing}_fst_cipher_k{k}_n{n}_w{running_avg_window}"
        manifest_section[label] = _train_condition_over_seeds(
            n=n,
            sensing=sensing,
            train_steps=train_steps,
            n_env=n_env,
            running_avg_window=running_avg_window,
            seeds=seeds,
            logdir=logdir,
            run_name_prefix=run_prefix,
            oracle_mode="fst_cipher",
            sigma_size=REQUESTED_DEFAULTS.sigma_size,
            fst_k=int(k),
            lie_prob=0.0,
            reward_noise_std=0.0,
            s1_step_penalty=TRAIN_DEFAULTS.s1_step_penalty,
        )
    return manifest_section


def run_experiment_6(
    *,
    n: int,
    sensing: str,
    train_steps: int,
    n_env: int,
    running_avg_window: int,
    seeds: list[int],
    logdir: str,
    d_values: list[int],
    oracle_mode: str = DEFAULT_ORACLE_MODE_EXP2,
    fst_k: int = DEFAULT_FST_K,
) -> dict:
    manifest_section: dict[str, list[dict]] = {}
    for d in sorted(set(int(value) for value in d_values)):
        if d < 1:
            raise ValueError("d values must be >= 1")
        if d >= n:
            raise ValueError(f"d values must be < n when self loops are disabled (d={d}, n={n})")
        label = f"d={d}"
        run_prefix = f"exp6_{sensing}_{oracle_mode}_d{d}_n{n}_w{running_avg_window}"
        manifest_section[label] = _train_condition_over_seeds(
            n=n,
            d=d,
            sensing=sensing,
            train_steps=train_steps,
            n_env=n_env,
            running_avg_window=running_avg_window,
            seeds=seeds,
            logdir=logdir,
            run_name_prefix=run_prefix,
            oracle_mode=oracle_mode,
            sigma_size=REQUESTED_DEFAULTS.sigma_size,
            fst_k=fst_k,
            lie_prob=0.0,
            reward_noise_std=0.0,
            s1_step_penalty=TRAIN_DEFAULTS.s1_step_penalty,
        )
    return manifest_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run requested experiments and write a manifest. "
            "Use v2.plot_requested_experiments to generate/re-generate plots."
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        default=REQUESTED_DEFAULTS.n,
        help="Graph size used for experiments 1, 3, 4, 5, and 6",
    )
    parser.add_argument(
        "--n_values",
        type=str,
        default=",".join(str(value) for value in REQUESTED_DEFAULTS.n_values),
        help="Comma-separated graph sizes for experiment 2",
    )
    parser.add_argument("--sensing", type=str, choices=["S0", "S1"], default=REQUESTED_DEFAULTS.sensing)
    parser.add_argument("--train_steps", type=int, default=REQUESTED_DEFAULTS.train_steps)
    parser.add_argument("--n_env", type=int, default=REQUESTED_DEFAULTS.n_env)
    parser.add_argument(
        "--running_avg_window", type=int, default=REQUESTED_DEFAULTS.running_avg_window
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=REQUESTED_DEFAULTS.seed,
        help="Base RNG seed used to sample run seeds",
    )
    parser.add_argument("--num_seeds", type=int, default=REQUESTED_DEFAULTS.num_seeds)
    parser.add_argument("--logdir", type=str, default=REQUESTED_DEFAULTS.logdir)
    parser.add_argument("--manifest_path", type=str, default=REQUESTED_DEFAULTS.manifest_path)
    parser.add_argument(
        "--lie_probs",
        type=str,
        default=",".join(str(value) for value in REQUESTED_DEFAULTS.lie_probs),
    )
    parser.add_argument(
        "--noise_sigmas",
        type=str,
        default=",".join(str(value) for value in REQUESTED_DEFAULTS.noise_sigmas),
    )
    parser.add_argument("--fst_k", type=int, default=DEFAULT_FST_K)
    parser.add_argument(
        "--fst_k_values",
        type=str,
        default=",".join(str(value) for value in REQUESTED_DEFAULTS.fst_k_values),
    )
    parser.add_argument(
        "--d_values",
        type=str,
        default=",".join(str(value) for value in REQUESTED_DEFAULTS.d_values),
        help="Comma-separated graph out-degrees for experiment 6 (degree sweep)",
    )
    parser.add_argument(
        "--oracle_mode_exp2",
        type=str,
        choices=["fixed_cipher", "fresh_cipher", "fst_cipher", "random_message", "no_oracle"],
        default=DEFAULT_ORACLE_MODE_EXP2,
        help="Oracle mode for experiment 2.",
    )
    parser.add_argument(
        "--oracle_mode_exp3",
        type=str,
        choices=["fixed_cipher", "fresh_cipher", "fst_cipher", "random_message", "no_oracle"],
        default=DEFAULT_ORACLE_MODE_EXP3,
        help=(
            "Oracle mode for experiment 3. "
            "For p_lie sweeps, fixed_cipher is recommended."
        ),
    )
    parser.add_argument(
        "--oracle_mode_exp4",
        type=str,
        choices=["fixed_cipher", "fresh_cipher", "fst_cipher", "random_message", "no_oracle"],
        default=DEFAULT_ORACLE_MODE_EXP4,
        help="Oracle mode for experiment 4.",
    )
    parser.add_argument(
        "--oracle_mode",
        type=str,
        choices=["fixed_cipher", "fresh_cipher", "fst_cipher", "random_message", "no_oracle"],
        default=None,
        help=(
            "Deprecated alias. If set, overrides oracle mode for experiments 2/3/4 "
            "(and experiment 6 via experiment 2 mode)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = [int(value.strip()) for value in args.n_values.split(",") if value.strip()]
    lie_probs = [float(value.strip()) for value in args.lie_probs.split(",") if value.strip()]
    noise_sigmas = [float(value.strip()) for value in args.noise_sigmas.split(",") if value.strip()]
    fst_k_values = [int(value.strip()) for value in args.fst_k_values.split(",") if value.strip()]
    d_values = [int(value.strip()) for value in args.d_values.split(",") if value.strip()]
    seeds = _random_seeds(base_seed=args.seed, num_seeds=args.num_seeds)
    oracle_mode_exp2 = args.oracle_mode_exp2
    oracle_mode_exp3 = args.oracle_mode_exp3
    oracle_mode_exp4 = args.oracle_mode_exp4
    if args.oracle_mode is not None:
        oracle_mode_exp2 = args.oracle_mode
        oracle_mode_exp3 = args.oracle_mode
        oracle_mode_exp4 = args.oracle_mode
    if oracle_mode_exp3 != "fixed_cipher" and any(value > 0.0 for value in lie_probs):
        warnings.warn(
            "experiment_3 uses lie_prob, but lie_prob currently affects only fixed_cipher; "
            f"oracle_mode_exp3={oracle_mode_exp3!r} may produce overlapping curves."
        )

    exp1 = run_experiment_1(
        n=args.n,
        sensing=args.sensing,
        train_steps=args.train_steps,
        n_env=args.n_env,
        running_avg_window=args.running_avg_window,
        seeds=seeds,
        logdir=args.logdir,
        fst_k=args.fst_k,
    )
    exp2 = run_experiment_2(
        n_values=n_values,
        sensing=args.sensing,
        train_steps=args.train_steps,
        n_env=args.n_env,
        running_avg_window=args.running_avg_window,
        seeds=seeds,
        logdir=args.logdir,
        oracle_mode=oracle_mode_exp2,
        fst_k=args.fst_k,
    )
    exp3 = run_experiment_3(
        n=args.n,
        sensing=args.sensing,
        train_steps=args.train_steps,
        n_env=args.n_env,
        running_avg_window=args.running_avg_window,
        seeds=seeds,
        logdir=args.logdir,
        lie_probs=lie_probs,
        oracle_mode=oracle_mode_exp3,
        fst_k=args.fst_k,
    )
    exp4 = run_experiment_4(
        n=args.n,
        sensing=args.sensing,
        train_steps=args.train_steps,
        n_env=args.n_env,
        running_avg_window=args.running_avg_window,
        seeds=seeds,
        logdir=args.logdir,
        noise_sigmas=noise_sigmas,
        binary_sensing=REQUESTED_DEFAULTS.binary_signal_sensing,
        oracle_mode=oracle_mode_exp4,
        fst_k=args.fst_k,
    )
    exp5 = run_experiment_5(
        n=args.n,
        sensing=args.sensing,
        train_steps=args.train_steps,
        n_env=args.n_env,
        running_avg_window=args.running_avg_window,
        seeds=seeds,
        logdir=args.logdir,
        fst_k_values=fst_k_values,
    )
    exp6 = run_experiment_6(
        n=args.n,
        sensing=args.sensing,
        train_steps=args.train_steps,
        n_env=args.n_env,
        running_avg_window=args.running_avg_window,
        seeds=seeds,
        logdir=args.logdir,
        d_values=d_values,
        oracle_mode=oracle_mode_exp2,
        fst_k=args.fst_k,
    )

    manifest = {
        "version": 2,
        "config": {
            "n": args.n,
            "d": REQUESTED_DEFAULTS.d,
            "d_values": sorted(set(d_values)),
            "t_pool": REQUESTED_DEFAULTS.t_pool,
            "sigma_size": REQUESTED_DEFAULTS.sigma_size,
            "n_values": sorted(n_values),
            "sensing": args.sensing,
            "train_steps": args.train_steps,
            "n_env": args.n_env,
            "running_avg_window": args.running_avg_window,
            "base_seed": args.seed,
            "num_seeds": args.num_seeds,
            "seeds_used": seeds,
            "logdir": args.logdir,
            "lie_probs": lie_probs,
            "noise_sigmas": sorted(set(noise_sigmas)),
            "fst_k": args.fst_k,
            "fst_k_values": sorted(set(fst_k_values)),
            "oracle_mode_exp2": oracle_mode_exp2,
            "oracle_mode_exp3": oracle_mode_exp3,
            "oracle_mode_exp4": oracle_mode_exp4,
            "oracle_mode_exp2_3_4": (
                oracle_mode_exp2
                if oracle_mode_exp2 == oracle_mode_exp3 == oracle_mode_exp4
                else "mixed"
            ),
            "binary_signal_sensing": REQUESTED_DEFAULTS.binary_signal_sensing,
            "binary_signal_step_penalty": REQUESTED_DEFAULTS.binary_signal_step_penalty,
        },
        "experiment_1": exp1,
        "experiment_2": exp2,
        "experiment_3": exp3,
        "experiment_4": exp4,
        "experiment_5": exp5,
        "experiment_6": exp6,
    }

    if args.manifest_path.strip():
        manifest_path = Path(args.manifest_path)
    else:
        manifest_path = Path("plots") / (
            f"requested_experiments_manifest_{args.sensing}_n{args.n}_"
            f"w{args.running_avg_window}_seeds{len(seeds)}.json"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path.resolve()),
                "seeds_used": seeds,
                "plot_cmd": (
                    f"python3 -m v2.plot_requested_experiments --manifest "
                    f"{manifest_path}"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
