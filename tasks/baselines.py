"""Additional baseline optimizers for evaluation.

Provides CMA-ES, multi-start GD/Adam, and Jacobian-based controllers
that follow the same rollout curve interface as existing baselines.
"""

import numpy as np


def rollout_cmaes_curve(
    *,
    obj_fn,
    start_xy: np.ndarray,
    horizon: int,
    bounds_low: np.ndarray | float,
    bounds_high: np.ndarray | float,
    sigma0: float = 0.5,
    popsize: int | None = None,
) -> np.ndarray:
    """Run CMA-ES for `horizon` function evaluations, returning best-so-far curve.

    Args:
        obj_fn: callable(x) -> float, returns normalized objective.
        start_xy: initial point, shape (d,).
        horizon: total function-evaluation budget.
        bounds_low / bounds_high: domain bounds (scalar or array).
        sigma0: initial step-size for CMA-ES.
        popsize: population size (None = CMA default).

    Returns:
        curve of shape (horizon+1,) with best-so-far objective.
    """
    import cma

    d = start_xy.shape[0]
    h = max(1, int(horizon))
    curve = np.empty(h + 1, dtype=np.float32)

    best_obj = float(obj_fn(start_xy.astype(np.float32)))
    curve[0] = best_obj

    # Configure CMA-ES
    opts = {
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
        "verb_filenameprefix": "/dev/null",
        "maxfevals": h,
        "bounds": [
            np.full(d, bounds_low, dtype=np.float64).tolist(),
            np.full(d, bounds_high, dtype=np.float64).tolist(),
        ],
        "tolfun": 0,
        "tolx": 0,
        "tolfunhist": 0,
        "tolstagnation": h + 10,
    }
    if popsize is not None:
        opts["popsize"] = int(popsize)

    es = cma.CMAEvolutionStrategy(
        start_xy.astype(np.float64).tolist(), float(sigma0), opts
    )

    eval_count = 0
    while eval_count < h and not es.stop():
        solutions = es.ask()
        fitnesses = []
        for sol in solutions:
            if eval_count >= h:
                fitnesses.append(float("inf"))
                continue
            x = np.asarray(sol, dtype=np.float32)
            obj = float(obj_fn(x))
            if obj < best_obj:
                best_obj = obj
            eval_count += 1
            if eval_count <= h:
                curve[eval_count] = best_obj
            fitnesses.append(obj)
        es.tell(solutions, fitnesses)

    # Fill remaining curve entries if CMA-ES stopped early
    for i in range(eval_count + 1, h + 1):
        curve[i] = best_obj

    return curve


def rollout_multistart_gd_curve(
    *,
    grad_fn,
    obj_fn,
    clip_fn,
    start_xy: np.ndarray,
    horizon: int,
    base_lr: float,
    n_restarts: int,
    rng: np.random.Generator,
    domain_low: np.ndarray | float,
    domain_high: np.ndarray | float,
) -> np.ndarray:
    """Multi-start GD: split budget among n_restarts random initializations.

    Args:
        grad_fn: callable(x) -> gradient array, shape (d,).
        obj_fn: callable(x) -> float, normalized objective.
        clip_fn: callable(x) -> clipped x (boundary enforcement).
        start_xy: initial point (used as one of the starts).
        horizon: total step budget.
        base_lr: learning rate.
        n_restarts: number of restarts (including original start).
        rng: random number generator.
        domain_low / domain_high: domain bounds for random starts.

    Returns:
        curve of shape (horizon+1,) with best-so-far objective.
    """
    d = start_xy.shape[0]
    h = max(1, int(horizon))
    steps_per_restart = max(1, h // max(1, n_restarts))

    curve = np.empty(h + 1, dtype=np.float32)
    best_obj = float(obj_fn(start_xy.astype(np.float32)))
    curve[0] = best_obj

    # Generate starting points: original + random
    starts = [start_xy.astype(np.float32).copy()]
    for _ in range(n_restarts - 1):
        low = np.broadcast_to(np.asarray(domain_low, dtype=np.float32), (d,))
        high = np.broadcast_to(np.asarray(domain_high, dtype=np.float32), (d,))
        starts.append(rng.uniform(low, high).astype(np.float32))

    step_idx = 0
    for state in starts:
        run_steps = min(steps_per_restart, h - step_idx)
        for local_step in range(run_steps):
            grad = grad_fn(state)
            # Simple cosine schedule within this restart
            frac = local_step / max(1, run_steps - 1) if run_steps > 1 else 0.0
            lr_t = base_lr * 0.5 * (1.0 + np.cos(np.pi * frac))
            update = (-lr_t * grad).astype(np.float32)
            state = clip_fn(state + update)
            obj = float(obj_fn(state))
            if obj < best_obj:
                best_obj = obj
            step_idx += 1
            curve[step_idx] = best_obj

    # Fill remaining if budget not exhausted
    for i in range(step_idx + 1, h + 1):
        curve[i] = best_obj

    return curve


def rollout_multistart_adam_curve(
    *,
    grad_fn,
    obj_fn,
    clip_fn,
    start_xy: np.ndarray,
    horizon: int,
    base_lr: float,
    n_restarts: int,
    rng: np.random.Generator,
    domain_low: np.ndarray | float,
    domain_high: np.ndarray | float,
) -> np.ndarray:
    """Multi-start Adam: split budget among n_restarts random initializations."""
    d = start_xy.shape[0]
    h = max(1, int(horizon))
    steps_per_restart = max(1, h // max(1, n_restarts))
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    curve = np.empty(h + 1, dtype=np.float32)
    best_obj = float(obj_fn(start_xy.astype(np.float32)))
    curve[0] = best_obj

    starts = [start_xy.astype(np.float32).copy()]
    for _ in range(n_restarts - 1):
        low = np.broadcast_to(np.asarray(domain_low, dtype=np.float32), (d,))
        high = np.broadcast_to(np.asarray(domain_high, dtype=np.float32), (d,))
        starts.append(rng.uniform(low, high).astype(np.float32))

    step_idx = 0
    for state in starts:
        m = np.zeros(d, dtype=np.float64)
        v = np.zeros(d, dtype=np.float64)
        run_steps = min(steps_per_restart, h - step_idx)
        for local_step in range(run_steps):
            grad = grad_fn(state).astype(np.float64)
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            t = local_step + 1
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            frac = local_step / max(1, run_steps - 1) if run_steps > 1 else 0.0
            lr_t = base_lr * 0.5 * (1.0 + np.cos(np.pi * frac))
            update = (-lr_t * (m_hat / (np.sqrt(v_hat) + eps))).astype(np.float32)
            state = clip_fn(state + update)
            obj = float(obj_fn(state))
            if obj < best_obj:
                best_obj = obj
            step_idx += 1
            curve[step_idx] = best_obj

    for i in range(step_idx + 1, h + 1):
        curve[i] = best_obj

    return curve


def rollout_jacobian_controller_curve(
    *,
    hidden_grad_fn,
    jacobian_fn,
    obj_fn,
    clip_fn,
    start_xy: np.ndarray,
    horizon: int,
    base_lr: float,
) -> np.ndarray:
    """Jacobian pseudoinverse controller: uses J^+ * g_hidden as visible-space update.

    This is the natural non-RL alternative when you have the hidden gradient
    and the Jacobian: compute the pseudoinverse update z_{t+1} = z_t - lr * J^+ g_h.

    Args:
        hidden_grad_fn: callable(x) -> hidden gradient, shape (D,).
        jacobian_fn: callable(x) -> Jacobian dF/dz, shape (D, d).
        obj_fn: callable(x) -> float, normalized objective.
        clip_fn: callable(x) -> clipped x.
        start_xy: initial point, shape (d,).
        horizon: number of steps.
        base_lr: learning rate.

    Returns:
        curve of shape (horizon+1,) with objective values.
    """
    state = start_xy.astype(np.float32).copy()
    h = max(1, int(horizon))
    curve = np.empty(h + 1, dtype=np.float32)
    curve[0] = float(obj_fn(state))

    for step in range(h):
        g_h = hidden_grad_fn(state).astype(np.float64)
        J = jacobian_fn(state).astype(np.float64)
        # Pseudoinverse: J^+ = (J^T J)^{-1} J^T, but use lstsq for stability
        # Solve J @ delta = g_h in least-squares sense
        delta, _, _, _ = np.linalg.lstsq(J, g_h, rcond=None)
        frac = step / max(1, h - 1) if h > 1 else 0.0
        lr_t = base_lr * 0.5 * (1.0 + np.cos(np.pi * frac))
        update = (-lr_t * delta).astype(np.float32)
        state = clip_fn(state + update)
        curve[step + 1] = float(obj_fn(state))

    return curve
