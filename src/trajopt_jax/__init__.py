"""trajopt-jax: trajectory optimisation in JAX on the GPU. A primal-dual interior-point solver with Riccati linear
algebra (parallel in time on a GPU), and a Radau collocation transcription for vehicle models with events, chained
missions and co-design. Modules: ``ipm`` (solve), ``batch`` (multi-start), ``riccati``, ``stages`` and ``events`` (the
transcription), ``specs`` (model declarations), ``reference`` (IPOPT and a numpy Riccati, for checks)."""
