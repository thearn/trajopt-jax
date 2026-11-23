# Open items

- **Free start states with first-stage events.** The start state is data today (every component fixed but the first
  leg's time and the design variables). A start box (chosen components free within bounds) and events judged at stage 0
  let a problem pin its start by constraints: the Earth-Mars example's departure, at Earth's position on a free launch
  date with the launch vehicle's v-infinity.
- **Polynomial controls per leg.** Controls are piecewise linear between stage knots, so their freedom grows with the
  mesh. A leg may instead carry one low-order polynomial (or a few spline coefficients) per control: a steady leg
  cannot then represent an oscillation, and refinement adds stages for state accuracy without control freedom. Test:
  the wing's recon (E24 in projects/wing), whose refined cruise waved and drew 19 % more charge than the uniform mesh.
- **Refinement: smooth re-seeding and the tolerance's scale.** Each pass starts from the previous pass's solution on the
  new mesh, so a coarse solution's artefacts carry forward (E24: converged at every pass, a worse local optimum from
  the 580-stage pass on); seed from a smoothed trajectory instead. The midpoint estimate stayed 0.11-0.38 with hundreds
  of stages above 1e-3 while the uniform mesh's was 0.022: set the tolerance from the estimator's measured scale.
