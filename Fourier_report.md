# Fourier Mapping Report

## Overview

In this project, the "Fourier mapping" is the hidden function `F(z)` that turns the agent's visible 2D coordinate `z` into a high-dimensional hidden state `s`.

The key idea is simple:

- The agent moves in a small, visible space: `z in R^2`
- The objective is defined in a larger, hidden space: `s in R^D`
- The connection between them is nonlinear:

```math
F(z) = A z + a \odot \sin(W z + b) + c \odot \cos(V z + d)
```

This map is what makes the task interesting. In hidden space, the objective is just a convex quadratic. But once that objective is pulled back through `F`, the 2D landscape the agent experiences becomes bumpy, folded, and non-convex.

## Intuition First

### What problem is the map solving?

If the hidden state were just `s = z`, then the task would be ordinary 2D optimization. The agent could directly follow a local gradient and likely solve the problem easily.

The Fourier mapping prevents that. It hides the true geometry behind a nonlinear embedding:

- nearby visible points can move in complicated ways in hidden space
- moving in a straight line in `z` does not usually produce a straight line in `s`
- the same small 2D move can help in one region and hurt in another

So the agent is not optimizing a clean bowl in 2D. It is moving over the shadow cast by a high-dimensional bowl through a warped surface.

### Why "Fourier"?

It is called Fourier because the nonlinear part is built from sine and cosine waves.

Sine and cosine are the standard building blocks for periodic structure. By summing many sinusoidal components with different frequencies, signs, amplitudes, and phases, the map can create:

- ripples
- ridges
- valleys
- repeated motifs
- many local minima in the visible landscape

This is a lightweight way to generate a complicated but still fully differentiable hidden map.

### A geometric picture

You can think of `F` as drawing a 2D sheet inside a `D`-dimensional hidden space.

- `z` picks a point on the sheet
- `F(z)` tells you where that point sits in hidden space
- the target is another hidden point `s* = F(z*)`
- the objective is how far the current hidden point is from the target:

```math
E(s) = \frac{1}{2}\|s - s^*\|^2
```

In hidden space this is just distance to a target, so it is convex and simple.

But the agent cannot move directly in hidden space. It can only slide around on the 2D sheet. If that sheet is wavy, then "move toward the target in hidden space" does not translate into an obvious local move in `z`.

That is exactly the communication challenge in the experiment.

## Technical Definition

For each hidden coordinate `i = 1, ..., D`, the environment defines

```math
F_i(z) = A_i z + a_i \sin(W_i z + b_i) + c_i \cos(V_i z + d_i)
```

where:

- `z in R^2` is the visible coordinate
- `A_i` is the `i`th row of a linear matrix `A in R^(D x 2)`
- `W_i` and `V_i` are 2D frequency vectors for the sine and cosine terms
- `a_i` and `c_i` are amplitudes
- `b_i` and `d_i` are phase offsets

Stacking all `D` coordinates gives the full hidden vector `F(z) in R^D`.

In the actual implementation, this is stored as:

- `linear_w` for `A`
- `sin_w` for `W`
- `cos_w` for `V`
- `sin_amp` for `a`
- `cos_amp` for `c`
- `sin_phase` for `b`
- `cos_phase` for `d`

## How The Parameters Are Sampled

When `f_type == "FOURIER"`, the map parameters are sampled randomly once per run by default, then shared across environments unless `refresh_map_each_episode=True`.

### Linear term

The linear component is sampled as

- `A_ij ~ Normal(0, 0.18)`

This gives the map a globally sloped component so it is not purely periodic.

### Frequency vectors

Each frequency entry is sampled as an integer magnitude and then assigned a random sign:

- `|W_ij| in {1, ..., basis_complexity}`
- `|V_ij| in {1, ..., basis_complexity}`
- sign chosen independently from `{-1, +1}`

So with the default `basis_complexity = 3`, each coordinate uses frequencies from `{-3, -2, -1, 1, 2, 3}`.

These integer frequencies matter because they control how quickly the sinusoidal terms oscillate over the visible space. Larger `basis_complexity` means a rougher, more rapidly varying map.

### Phases

The phase offsets are sampled independently:

- `b_i ~ Uniform(0, 2pi)`
- `d_i ~ Uniform(0, 2pi)`

Phases shift the waves left or right, preventing all hidden dimensions from lining up in the same pattern.

### Amplitudes

The amplitudes are sampled independently:

- `a_i ~ Uniform(0.8, 1.3)`
- `c_i ~ Uniform(0.6, 1.1)`

These ranges ensure the nonlinear sinusoidal part is strong enough to shape the landscape, rather than being a tiny perturbation.

## Why This Produces A Hard Optimization Problem

The objective is defined in hidden space:

```math
E(z) = \frac{1}{2}\|F(z) - s^*\|^2
```

with `s* = F(z*)` for some sampled target point `z*`.

Although this is just squared distance in hidden space, the composition with `F` makes the visible-space objective non-convex.

The reason is that `F` can fold and warp the 2D plane:

- multiple visible regions can map near each other in hidden space
- the local slope in `z` can change direction quickly
- steepest descent in 2D can get trapped by visible-space ripples

This is why the project compares the learned policy to a 2D gradient descent baseline. Even with the true visible-space gradient, local descent can be much worse than using the oracle's hidden-space signal in a more global way.

## What If Visible Dimension Is Greater Than 2?

Short answer: yes in hidden space, no in visible space in general.

The convexity statement in the project is about the hidden-space objective

```math
E(s) = \frac{1}{2}\|s - s^*\|^2
```

and that remains convex no matter what the visible dimension is. If the agent controlled `z in R^m` for any `m >= 1`, the function of `s` is still just a quadratic bowl.

What changes is the pulled-back objective as a function of the visible variable:

```math
E(z) = \frac{1}{2}\|F(z) - s^*\|^2
```

If `F` is the nonlinear Fourier map, this is generally not convex, whether `z` is 2-dimensional or higher-dimensional. Increasing the visible dimension does not restore convexity by itself. It only gives the agent more directions in which to move on the warped manifold defined by `F`.

So:

- convex in `s`: always yes
- convex in `z`: generally no for nonlinear Fourier `F`

There is one important special case. If `F` were purely linear, say `F(z) = Az`, then

```math
E(z) = \frac{1}{2}\|Az - s^*\|^2
```

would be convex in `z` for any visible dimension, because it is a quadratic form. The loss of convexity comes from the sine and cosine terms, not from the fact that the visible dimension is 2.

## The Jacobian

The Jacobian of the Fourier map is central to the technical story because it converts hidden-space information into visible-space gradients.

For the Fourier map,

```math
J(z) = \frac{\partial F(z)}{\partial z}
```

and each hidden coordinate contributes

```math
\nabla_z F_i(z)
= A_i
+ a_i \cos(W_i z + b_i) W_i
- c_i \sin(V_i z + d_i) V_i
```

So the full Jacobian is

```math
J(z)
= A
+ \big(a \odot \cos(Wz+b)\big) \odot W
- \big(c \odot \sin(Vz+d)\big) \odot V
```

written row-wise.

This matches the implementation:

- sine terms differentiate to cosine
- cosine terms differentiate to `-sin`
- each derivative is scaled by its frequency vector

An important consequence is that higher frequencies do not just make the map wigglier. They also increase local sensitivity, because derivatives scale with the frequency values themselves.

## Hidden Gradient vs Visible Gradient

The oracle provides the hidden-space gradient:

```math
g_h(z) = F(z) - s^*
```

because for

```math
E(s) = \frac{1}{2}\|s-s^*\|^2
```

the gradient with respect to `s` is exactly `s - s*`.

But the true visible-space gradient is

```math
\nabla_z E(z) = J(z)^T (F(z) - s^*)
```

This is the chain rule.

So the Fourier map creates a separation between:

- the oracle's message: "where the target is in hidden space"
- the control gradient: "which 2D move decreases the objective locally"

The agent receives the first, not the second. To act well, it must learn an implicit approximation to how hidden gradients and local position combine into useful visible motions.

## Role Of `basis_complexity`

`basis_complexity` controls the maximum absolute frequency used in `sin_w` and `cos_w`.

Low values:

- smoother map
- slower oscillations
- simpler visible landscape

High values:

- rougher map
- faster oscillations
- more local traps and sharper curvature

So this parameter is effectively a knob for how difficult the hidden-to-visible translation problem is.

## Why Include Both Linear And Fourier Terms?

The map is not just a pure sinusoidal basis. It includes a linear term `Az` as well.

That helps in two ways:

1. It gives the map a coarse global structure, so the surface is not only repeating waves.
2. It prevents the geometry from being too symmetric or too degenerate.

The result is a landscape that is structured but not trivial: globally organized, locally rugged.

## Why Sample The Map Once Per Run?

By default, the same Fourier map is reused across episodes within a run.

That means the policy can gradually learn the geometry of one hidden world:

- how oracle gradients relate to good 2D moves
- where local minima tend to appear
- how to exploit regularities in the same map over many episodes

If `refresh_map_each_episode=True`, that regularity disappears. Then the policy would need to generalize across many different sampled Fourier worlds, which is a harder problem.

## Practical Interpretation In This Project

The Fourier mapping is doing three jobs at once:

1. It hides the true optimization problem from the agent.
2. It creates a nonlinear communication bottleneck between oracle signals and visible actions.
3. It turns a simple convex hidden objective into a hard non-convex visible task.

Without this map, the experiment would be much less interesting. The main question of the project is whether a learned policy can use oracle messages that are meaningful in hidden space, even when the visible action consequences are warped by a complicated transformation. The Fourier map is the mechanism that creates that mismatch.

## Bottom Line

Intuitively, the Fourier mapping is a randomly generated wavy embedding from 2D control space into high-dimensional hidden space.

Technically, it is a differentiable random feature map

```math
F(z) = A z + a \odot \sin(W z + b) + c \odot \cos(V z + d)
```

with integer-valued frequencies, random amplitudes, and random phases.

Its purpose is to make the visible optimization landscape difficult while preserving a simple hidden-space objective. That lets the project test whether the agent can learn to interpret hidden-space oracle gradients and translate them into effective visible-space behavior.
