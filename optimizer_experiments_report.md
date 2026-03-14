# Optimizer Experiments: Current Readout

Date: 2026-03-14

## Scope

This report summarizes the current results in `plots/spatial_optimizer_studies` for:

- `meta_optimizer`
- `search_algorithm`

The current suite uses 3 seeds (`1705`, `2626`, `62224`).

Important context from the stored manifest/code:

- Meta study: 500 evaluation tasks per seed, horizon 60.
- Search study: one fixed task per seed, environment-step budgets up to 300,000.
- Search-policy rollouts are stochastic by default (`search_rl_deterministic = false`).
- Search wall-clock curves measure evaluation time only, not policy training cost.

## Executive Read

So far, the results tell a fairly clean story:

- The hidden-gradient RL policy is the strongest **meta-optimizer** by a clear margin.
- Random search is the strongest **search algorithm** under the current fixed-task, anytime-budget setup.
- The two RL results are not contradictory. They suggest the hidden-gradient policy learned a strong amortized optimizer for one rollout on a fresh task, but not a strong repeated-episode search procedure.

That is the main conclusion I would trust at this point.

## Meta-Optimizer Results

Mean final objective across seeds:

| Method | Mean final objective | 95% CI |
| --- | ---: | ---: |
| RL hidden gradient | 0.0137 | 0.0065 |
| Random search | 0.0351 | 0.0031 |
| Adam | 0.2013 | 0.0095 |
| GD | 0.2146 | 0.0130 |
| RL visible oracle | 0.3009 | 0.0116 |
| RL no oracle | 0.5131 | 0.0127 |

### What stands out

`RL hidden gradient` is the clear winner. In the curve plot it collapses from about `0.58` to about `0.02` within the first `6-8` optimization steps and finishes around `0.014`. That is materially better than:

- `random_search` at `0.035`
- `adam` at `0.201`
- `gd` at `0.215`

This is strong evidence that the oracle-gradient communication channel is carrying real optimization value, and that the hidden-gradient policy is learning to use it.

`Random search` is surprisingly strong as a meta baseline. It is much better than GD/Adam on final objective and is the second-best method overall. That means the Fourier-induced landscape is local-minima-heavy enough that unguided global sampling beats local visible-space descent.

`RL visible oracle` and especially `RL no oracle` are not competitive as final optimizers. The hidden oracle signal seems to matter much more than simply training an RL controller in the same environment.

### Trajectory quality vs final-state quality

There is also a useful stability signal in the gap between each method's best visited state and final state:

- `GD`: no gap
- `Adam`: essentially no gap
- `Random search`: no gap, by construction
- `RL visible oracle`: final-best gap `0.1069`
- `RL no oracle`: final-best gap `0.3158`
- `RL hidden gradient`: final-best gap `0.0124`

Interpretation:

- The classical optimizers and random search are effectively monotone in the metric being plotted.
- The RL policies often pass through better states than where they end up.
- `RL hidden gradient` is much better than the other RL variants, but it still looks slightly under-damped near good regions. It finds very good states on average (`mean best objective = 0.00125`) and then gives some of that back by the end of the rollout (`mean final objective = 0.01366`).

That suggests a concrete improvement direction: better stopping behavior, smaller terminal step sizes, or a policy head that can explicitly choose to stop/refine.

## Search-Algorithm Results

The search study tells a different story.

At selected environment-step budgets:

| Budget | Random search | RL no oracle | RL visible oracle | RL hidden gradient |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.1329 | 0.2342 | 0.1192 | 0.3293 |
| 20 | 0.1072 | 0.0554 | 0.1119 | 0.2544 |
| 100 | 0.0104 | 0.0418 | 0.0683 | 0.1734 |
| 220 | 0.0026 | 0.0098 | 0.0118 | 0.0326 |
| 526 | 0.0005 | 0.0056 | 0.0061 | 0.0280 |
| 1730 | 0.000241 | 0.000681 | 0.001869 | 0.009542 |
| 300000 | 0.000002 | 0.000014 | 0.000005 | 0.000069 |

Lower is better.

### What stands out

`Random search` dominates most of the budget frontier and remains best even at the maximum budget.

Threshold crossings show the same pattern:

| Method | First budget with mean objective <= 0.1 | <= 0.01 | <= 0.001 |
| --- | ---: | ---: | ---: |
| Random search | 38 | 117 | 526 |
| RL no oracle | 19 | 220 | 1164 |
| RL visible oracle | 42 | 526 | 4481 |
| RL hidden gradient | 220 | 1730 | 14727 |

`RL no oracle` has one narrow advantage: it reaches `<= 0.1` earlier than the others. But after that, `random_search` becomes the best anytime method and opens a large gap.

`RL hidden gradient`, which was the best meta-optimizer, is the weakest search algorithm for most of the curve and still the worst at the maximum budget. That is the most important discrepancy in the whole study.

### Speed-quality frontier

The wall-clock plot makes the same point more strongly:

- At `100` steps, `random_search` gets to `0.0104` in about `0.95 ms`.
- `RL no oracle` needs about `10.23 ms` to reach `0.0418`.
- `RL visible oracle` needs about `11.83 ms` to reach `0.0683`.
- `RL hidden gradient` needs about `13.31 ms` to reach `0.1734`.

At the full `300000`-step budget:

- `random_search`: about `3.18 s`
- RL methods: about `31.7-34.8 s`

So under the current evaluation protocol, random search is not just better in objective. It is also about an order of magnitude cheaper in wall-clock time.

## Interpretation

My current read is:

1. The hidden-gradient policy has learned a strong **amortized optimizer**.
2. It has not learned a strong **search procedure**.

Those are different capabilities.

In the meta study, the policy gets one rollout on a fresh task and is judged by the quality of that trajectory. That setup rewards a controller that can rapidly exploit the hidden oracle signal and move to a good region in one pass. The hidden-gradient policy does that very well.

In the search study, the method is judged as an anytime repeated-episode search algorithm on a fixed task under a strict environment-step budget. That setup rewards broad coverage, restart efficiency, and strong best-so-far behavior across many trials. On that metric, random search is better.

So the current evidence supports:

- "oracle-guided RL is a strong meta-optimizer"

but does not support:

- "oracle-guided RL is a strong generic search algorithm"

at least not in the present form.

## Why The Meta/Search Split Probably Happened

The simplest explanation is that the hidden-gradient policy is learning a good direct controller, not a good proposal distribution for repeated search.

More specifically:

- It seems optimized to turn a hidden gradient signal into a strong single rollout.
- It does not seem calibrated for repeated stochastic restarts on one fixed task.
- It still overshoots after finding good states, which hurts search-style best-use-of-budget performance.
- Random search benefits from being extremely cheap and from covering the 2D control space broadly.

There is also a methodological caveat: the search evaluation uses stochastic policy actions by default. If the intended use case is "deploy the learned optimizer once on a task," deterministic evaluation may be the more faithful measurement. The current search result is therefore partly measuring the quality of the policy's stochastic rollout distribution, not just its nominal control law.

## Caveats

The current conclusions are directionally strong, but I would still treat them as provisional because:

- The suite only has 3 seeds.
- The search study uses one fixed task per seed, not a task distribution.
- Search wall-clock excludes RL training cost. Including training would only make RL look less favorable as a search baseline.
- The visible/no-oracle RL confidence bands are large at small budgets, so early-budget ranking is noisy.

## What I Would Do Next

If the goal is to strengthen the paper/story, I would run these next:

1. Re-run the search study with `--search_rl_deterministic`.
   This will tell us whether the policy is actually a strong one-shot optimizer but is being dragged down by stochastic sampling.

2. Increase to at least 10 seeds.
   The current 3-seed setup is enough to see the broad pattern, but not enough to make fine claims.

3. Evaluate search on a distribution of tasks, not one fixed task per seed.
   That would separate "amortized optimizer across tasks" from "search procedure on one task."

4. Add a hybrid baseline.
   Two obvious options:
   - RL hidden-gradient rollout followed by local refinement
   - RL proposals plus random restarts

5. Add explicit stopping/refinement behavior to the policy.
   The hidden-gradient policy is already finding very good states; it just is not holding them reliably to the end of the rollout.

## Bottom Line

The current optimizer experiments are promising, but in a specific way:

- `RL hidden gradient` looks genuinely strong for meta-optimization.
- `Random search` is still the best search baseline under the current budgeted fixed-task protocol.

So far, I would present this as evidence that oracle communication helps learn an amortized optimizer, not yet as evidence that the learned policy beats simple search in an anytime setting.
