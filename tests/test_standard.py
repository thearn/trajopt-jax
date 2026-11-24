"""Standard optimal-control problems with closed-form optima, through the transcription (trajopt_jax.stages) and the
interior-point solver: the minimum-time double integrator and the brachistochrone."""

from __future__ import annotations

import math
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from trajopt_jax import ipm  # noqa: E402
from trajopt_jax._jax import jnp  # noqa: E402
from trajopt_jax.events import Leg, Point  # noqa: E402
from trajopt_jax.specs import ControlSpec, PathConstraint, StateSpec  # noqa: E402
from trajopt_jax.stages import Stages  # noqa: E402

G = 9.81


class _Base:
    outputs = ('zero',)
    data_width = 0
    seed_integrated = ('cost',)
    angles = ()
    constraints: tuple[PathConstraint, ...] = ()

    def params(self) -> np.ndarray:
        return jnp.zeros(0)


class DoubleIntegrator(_Base):
    """x'' = a, |a| <= 1, from rest; the cost is time."""

    states = (StateSpec('x', 'm', 10.0), StateSpec('v', 'm/s', 1.0), StateSpec('cost', 's', 10.0))
    controls = (ControlSpec('a', -1.0, 1.0),)

    def rhs(self, s, u, u_rate, p, data):  # noqa: ANN001, ANN201 - the model protocol
        return jnp.stack([s[1], u[0], jnp.ones(())]), jnp.zeros(1)

    def pose(self, s):  # noqa: ANN001, ANN201 - the model protocol
        return jnp.stack([s[0], 0.0, 0.0]), jnp.stack([s[1], 0.0, 0.0])

    def state_from(self, pos: np.ndarray, vel: np.ndarray, base: np.ndarray) -> np.ndarray:
        out = np.array(base, float)
        out[:2] = pos[0], vel[0]
        return out


class Brachistochrone(_Base):
    """A bead from rest under gravity, x' = v sin(theta), y' = -v cos(theta), v' = g cos(theta); the cost is time."""

    states = (StateSpec('x', 'm', 10.0), StateSpec('y', 'm', 10.0), StateSpec('v', 'm/s', 10.0), StateSpec('cost', 's', 10.0))
    controls = (ControlSpec('theta', 0.0, math.pi),)

    def rhs(self, s, u, u_rate, p, data):  # noqa: ANN001, ANN201 - the model protocol
        th = u[0]
        return jnp.stack([s[2] * jnp.sin(th), -s[2] * jnp.cos(th), G * jnp.cos(th), jnp.ones(())]), jnp.zeros(1)

    def pose(self, s):  # noqa: ANN001, ANN201 - the model protocol
        th = jnp.arctan2(1.0, 1.0)
        return jnp.stack([s[0], s[1], 0.0]), jnp.stack([s[2] * jnp.sin(th), -s[2] * jnp.cos(th), 0.0])

    def state_from(self, pos: np.ndarray, vel: np.ndarray, base: np.ndarray) -> np.ndarray:
        out = np.array(base, float)
        out[:3] = pos[0], pos[1], max(float(np.linalg.norm(vel)), 0.1)
        return out


def solve(model: _Base, start: np.ndarray, u0: np.ndarray, end: Point, speed: float, segments: int = 30) -> float:
    prob = Stages(model, start, u0, (Leg((end,), max_s=30.0),), segments=segments)
    X, U = prob.path_seed(speed)
    res = ipm.solve(prob, X, U, prob.x0(X[0, -1]), max_iter=300, verbose=False)
    assert res.converged, res.status
    return float(res.X[0, -1])


def test_min_time_double_integrator_is_sqrt_2d() -> None:
    d = 10.0
    t = solve(DoubleIntegrator(), np.array([0.0, 0.0, 0.0]), np.array([1.0]), Point(d, None, None, tol=1e-3), speed=3.0)
    assert t == pytest.approx(math.sqrt(2 * d), rel=2e-3)


def test_brachistochrone_matches_the_cycloid() -> None:
    D, H = 10.0, 5.0
    phi = brentq(lambda f: (f - math.sin(f)) / (1 - math.cos(f)) - D / H, 0.1, 2 * math.pi - 0.1)
    R = H / (1 - math.cos(phi))
    t_cycloid = phi * math.sqrt(R / G)
    t = solve(Brachistochrone(), np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.5]), Point(D, -H, None, tol=1e-3), speed=5.0)
    assert t == pytest.approx(t_cycloid, rel=2e-3)


class ScaledIntegrator(DoubleIntegrator):
    """x'' = p a, |a| <= 1: the minimum time to ``d`` from rest is sqrt(2 d / p), so d(T)/d(p) = -T / (2 p)."""

    def __init__(self, p: float) -> None:
        self.p = p

    def params(self) -> np.ndarray:
        return jnp.array([self.p])

    def design_bounds(self) -> list[tuple[float, float]]:
        return [(0.1, 10.0)]

    def design_constraints(self, p):  # noqa: ANN001, ANN201 - the model protocol
        return []

    def rhs(self, s, u, u_rate, p, data):  # noqa: ANN001, ANN201 - the model protocol
        return jnp.stack([s[1], p[0] * u[0], jnp.ones(())]), jnp.zeros(1)


@pytest.mark.parametrize('p', [1.0, 2.5])
def test_pinned_design_gradient_matches_the_closed_form(p: float) -> None:
    d = 10.0
    model = ScaledIntegrator(p)
    prob = Stages(model, np.array([0.0, 0.0, 0.0]), np.array([1.0]), (Leg((Point(d, None, None, tol=1e-3),), max_s=30.0),), segments=30,
                  codesign=True, design_free=False)  # fmt: skip
    X, U = prob.path_seed(3.0)
    res = ipm.solve(prob, X, U, prob.x0(X[0, -1]), max_iter=300, verbose=False)
    assert res.converged, res.status
    assert float(res.X[0, -1]) == pytest.approx(math.sqrt(2 * d / p), rel=2e-3)  # (pinned: the design did not move)
    np.testing.assert_allclose(prob.design(res.X[0]), [p])
    grad = prob.design_gradient(res.lam0)
    assert grad[0] == pytest.approx(-float(res.obj) / (2 * p), rel=1e-4)
