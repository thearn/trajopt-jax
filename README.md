# trajopt-jax

Trajectory optimisation in JAX, built to run on a GPU. A primal-dual interior-point solver whose linear algebra is a
Riccati recursion (parallel in time on a GPU), and a Radau collocation transcription for vehicle models: events at leg
ends and along legs (gates, points, orbits, boxes), chained missions sharing one design, and co-design of design
variables carried as states. Checked against closed-form optima (double integrator, brachistochrone) and IPOPT.

Extracted from the whoop and wing vehicle projects (designloop); no designloop dependency.
