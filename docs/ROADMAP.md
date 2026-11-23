# Open items

Done: free start states with first-stage events (`Stages.start_free`, `start_events`; the rendezvous example's
departure), 2026-11-23.


- **Polynomial controls per leg.** Controls are piecewise linear between stage knots, so their freedom grows with the
  mesh. A leg may instead carry one low-order polynomial (or a few spline coefficients) per control: a steady leg
  cannot then represent an oscillation, and refinement adds stages for state accuracy without control freedom. Test:
  the wing's recon (E24 in projects/wing), whose refined cruise waved and drew 19 % more charge than the uniform mesh.
- **Refinement: smooth re-seeding and the tolerance's scale.** Each pass starts from the previous pass's solution on the
  new mesh, so a coarse solution's artefacts carry forward (E24: converged at every pass, a worse local optimum from
  the 580-stage pass on); seed from a smoothed trajectory instead. The midpoint estimate stayed 0.11-0.38 with hundreds
  of stages above 1e-3 while the uniform mesh's was 0.022: set the tolerance from the estimator's measured scale.
- **The real-ephemeris transfer example** (astropy planets, any pair Mercury-Neptune): converges for Earth-Mars arriving
  2027-09-01 (launch 2026-08-09, 387 days, 252 kg) but its bang-bang optimum re-propagates 55-76 thousand km off at
  120-240 stages (the switches need refinement or switch times as variables), and Mars arriving 2033-07-08 is out of
  phase with its launch windows (to retry with a 0.3-2.6 yr flight window). Not committed until those close.
