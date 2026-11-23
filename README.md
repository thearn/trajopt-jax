# trajopt-jax

Trajectory optimisation written entirely in JAX, built to run on a GPU.

```bash
pip install git+https://github.com/thearn/trajopt-jax
```

Example: [`examples/brachistochrone.py`](examples/brachistochrone.py), and as a notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thearn/trajopt-jax/blob/main/examples/brachistochrone.ipynb)

## Key features

- **All JAX.** Your model's dynamics are a JAX function; derivatives (Jacobians and exact Hessians) come from JAX,
  so there are no hand-written partials and no finite differences.
- **GPU native, with its own solver.** A primal-dual interior-point method whose linear algebra is a Riccati
  recursion, run in parallel over time on a GPU (sequential on a CPU); the whole iteration compiles to one XLA program.
- **Radau collocation for vehicle models.** Multi-leg problems with free leg durations, controls linear across
  stages with box and rate limits, path constraints on any model output.
- **Missions as events.** Legs end at gates, points, boxes or state caps and can hold constraints along their
  length (an orbit); several missions can be chained into one problem sharing a design.
- **Co-design.** Design variables carried as states, so the trajectory and the vehicle are optimised together.
- **Mesh refinement.** Passes that split the stages where the solution needs resolution.
- **Checked** against closed-form optima (double integrator, brachistochrone) and IPOPT on the same problems.

## FAQ

**Why use this over OpenMDAO/Dymos?** You probably shouldn't. This is an experimental repository aimed at hobby
problems (small drones and similar). Dymos is far more mature, general and validated.

**Does it include the MPCC pilot models?** No. Model-predictive contouring control (iLQR in a receding horizon)
solves a different problem, flying a path in closed loop like a pilot; it will live in a separate control library.
