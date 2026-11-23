"""What the interior-point solver reads from a problem (trajopt_jax.ipm): sizes, the raw stage functions it compiles,
the free start components, the objective's state and scale, the per-stage data, and an evaluation with Jacobians.
trajopt_jax.stages.Stages is one; any object with these attributes is another."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np


class StageProblem(Protocol):
    N: int  # stages
    nx: int  # stage state size
    nu: int  # stage control size
    ng: int  # inequalities a stage
    nh: int  # stage equalities a stage (Radau collocation defects; 0 for explicit steps)
    nl: int  # design variables carried as states (co-design; 0 otherwise)
    ci: int  # the objective's state (its final value, times ref[ci] / 10, is minimised)
    ref: np.ndarray  # state scales
    free: np.ndarray  # start components the solver chooses (the rest fixed at x0)
    masks: np.ndarray  # per-stage data for the dynamics and defects
    ends: np.ndarray  # per-stage data for the inequalities (N + 1 rows)
    node_second_order: bool  # (the node's own derivatives: a lighter Hessian pass)
    raw: dict[str, Callable[..., Any]]  # dyn, ineq, defects, hessian, hessian_eq

    def evaluate(self, X: np.ndarray, U: np.ndarray, jac: bool, data: tuple | None = None) -> dict: ...
