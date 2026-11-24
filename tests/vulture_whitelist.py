"""Names vulture reports on src/trajopt_jax alone that tests, the projects or a framework use."""

# ruff: noqa: B018
from trajopt_jax import batch, envelope, nested, reference, refine, result, specs, stages

envelope.with_data
envelope.data_sensitivity
reference._IpoptStages.objective
reference._IpoptStages.gradient
reference._IpoptStages.jacobianstructure
reference._IpoptStages.intermediate
refine.solve_refined
stages.Model.Model
stages.Stages.design
stages.Stages.path_seed
stages.Stages.trajectory  # (the vehicles export solutions with it)
nested.Nested  # (the vehicles' co-design drivers)
nested.trust_region
stages.chained

# whitelist-audit: off  (record fields and constants a caller reads by attribute)
batch.BatchResult.iterations
batch.BatchResult.failed
batch.BatchResult.wall_s
batch.BatchResult.restart
reference._IpoptStages.alg_mod
result.IPMResult.iterations
result.IPMResult.wall_s
specs.StateSpec.unit
stages.Model.u_rate
# whitelist-audit: on
