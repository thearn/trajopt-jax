"""Mesh refinement on the brachistochrone: from a coarse mesh, passes split the stages whose error is above
tolerance, and the time approaches the cycloid's with fewer stages than a uniform mesh needs."""

from __future__ import annotations

import math
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from test_standard import Brachistochrone, G  # noqa: E402

from trajopt_jax import ipm  # noqa: E402
from trajopt_jax.events import Leg, Point  # noqa: E402
from trajopt_jax.refine import solve_refined, stage_errors  # noqa: E402
from trajopt_jax.stages import Stages  # noqa: E402

D, H = 10.0, 5.0
START, U0 = np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.5])


def cycloid_time() -> float:
    phi = brentq(lambda f: (f - math.sin(f)) / (1 - math.cos(f)) - D / H, 0.1, 2 * math.pi - 0.1)
    return phi * math.sqrt(H / (1 - math.cos(phi)) / G)


def build(legs: tuple[Leg, ...]) -> Stages:
    return Stages(Brachistochrone(), START, U0, legs, segments=4)


def test_refinement_reaches_the_cycloid_with_fewer_stages_than_a_uniform_mesh() -> None:
    """From 4 stages, passes at tol 1e-5 reach 0.055 % of the cycloid on 19 stages (2026-11-22); a uniform 30-stage
    solve is held to 0.2 % (test_standard)."""
    legs = (Leg((Point(D, -H, None, tol=1e-3),), max_s=30.0),)
    coarse = build(legs)
    X, U = coarse.path_seed(5.0)
    r0 = ipm.solve(coarse, X, U, coarse.x0(X[0, -1]), max_iter=300, verbose=False)
    e0 = stage_errors(coarse, r0.X, r0.U)
    prob, res, record = solve_refined(build, legs, r0.X, r0.U, passes=5, tol=1e-5, max_iter=300, verbose=False)
    exact = cycloid_time()
    assert res.converged and coarse.N < prob.N < 30
    assert abs(res.X[0, -1] - exact) / exact < 1e-3 < abs(r0.X[0, -1] - exact) / exact
    assert record[-1]['err_max'] < 1e-5 < e0.max()
