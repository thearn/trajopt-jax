# Solving: the default and what to try when it does not converge

How the vehicle projects (projects/wing, projects/whoop) solve their missions and co-design, from what worked and what
did not on both; each step cites the experiment that set it. A new vehicle starts here.

## The default

1. **One compiled problem.** The missions chained as stages (`chained`, `Mission`, junctions), the design in the model
   data (`data_width`, a design-as-data model), so one compilation serves every design and every mission's conditions
   (wind, throttle cap) are stage data, not new problems.
2. **A fixed-design solve first, kept.** `path_seed`, then `ipm.solve(..., parallel=False)`; save `X`, `U`. Every later
   solve starts from it. A chained problem solved cold from the seed stalled on its first solve (whoop lap and course,
   runq 6312: objective 2.03 against ~0.22) where the same chain at a fixed design converged in 209 iterations (6369).
   The sequential Riccati scan is the default: the parallel one's steps were too inexact on 784-stage chains (ROADMAP).
3. **Co-design nested** (`trajopt_jax.nested`): an outer loop over the design whose inner solve is the problem at a
   fixed design, each warm from the last converged one (multipliers kept, `WARM_MU`), a far design reached in steps of
   at most `MAX_STEP` of each variable's size (`scale`), the gradient the optimum's sensitivity to the data (envelope
   theorem), the outer loop `trust_region` on the design over its size. The whoop: course 1.5775 and lap 3.7440 against
   all-at-once 1.5766 and 3.7438 (runq 6041, 6042).
   Several missions or scenarios (a chained problem): their weighted sum by default; `missions=True` with
   `optimise(risk=beta)` their conditional value at risk (`cvar`, the mean of the worst `beta` share), each mission's
   objective and gradient from its own stages' multipliers (`envelope.mission_parts`), the outer loop on the bounds alone.
   Scenarios that do not converge chained (the whoop's lap in four winds, 800-1500 iterations) go as separate problems
   sharing the design: `nested.scenarios`, one `Nested` each, each warm from its own last solution. **Start from a
   design that flies every scenario**: `scenarios` solves each at the start design first and stops on any that does
   not converge. The stock whoop cannot fly the lap into a 6 m/s headwind (stalled at feasibility 0.25 even from the
   4 m/s solve), the lap co-design flies all four winds at once (whoop runq 7710-7723): three co-designs from stock
   went nowhere. Reach a hard scenario by continuation in its own parameter (the wind in steps) at that design.
   From the lap co-design the CVaR of the four winds matched the old bilevel co-design (whoop's SNOPT, one process
   per mission): worst lap 4.3565 s against 4.3566 s, in 2,266 s against 5,655 s (whoop runq 7724).
4. **All-at-once co-design** (`codesign=True`) only as a cross-check on small problems. On the wing it stalled at two
   designs a fixed one beat and took 5.5 GB a 128-stage block with the design's map in every node (wing E39-E41).

## Before a result is trusted

- **The gradient against re-solves** either side in one design variable, from the same converged state (the vehicles'
  `codesign_nested.py --check`). Near an optimum judge the absolute error: the wing at rev b, 4 % relative, 8e-6
  absolute, the same as 0.35 % at the stock design (wing 6029, E39-E41).
- **The status kept with the result** (converged, acceptable, iteration limit) in each design point's results.yaml, so
  the missions table shows an unconverged one (the wing's E60, an iteration limit read as a result until then).
- **A finite gradient**: `Nested` stops on one that is not (a NaN sent the whoop's outer loop to an all-NaN design).
  The usual cause is a square root or power at 0 behind a clip or `where`: its infinite slope times zero is NaN even in
  the branch not taken (designloop-propulsion's vortex-ring fade, 2026-11-25). Use the double `where`.

## When it does not converge, in order

1. **Start from a converged neighbour**: the same design, a nearby one, or each mission alone before chaining.
2. **Smaller design steps**: `MAX_STEP` and `scale` (a variable at 0 needs its `scale`: the whoop's prop pitch took a
   1e-7 step in 2400 solves without it).
3. **More iterations from where it stopped**: `--warm` the last result (the wing's E60 iteration limit, then E74
   converged at -7.9 %).
4. **Cap what the solver wanders in**: a leg with no weight sat at its maximum time (the wing's takeoff and landing at
   15-20 s, E51): cap it (8 s).
5. **Keep the multipliers on a restart**: restarted at mu 0.1 a design re-converged into another local optimum ~1 % away,
   which a design gradient cannot see (wing E39-E41).
6. **A better seed, or several**: spline seeds (`path_seed`); multistart to find the local optima. Lattice A* seeds are
   planned, not built.
7. **The mesh**: refinement re-seeds from the coarse solution and can carry its artefacts into a worse local optimum
   (wing E24, ROADMAP); compare against the uniform mesh.

## Tried and dropped

- SLSQP as the outer loop: its first step moved the wing's span 33 % (314 iterations, 20 min) where 5 % steps take ~20.
- scipy's trust-constr: its barrier, as large as the objective's gradient, held every step to 1-2 %.
- The design as constant states in every node: second order through the design's map, out of memory (above).
