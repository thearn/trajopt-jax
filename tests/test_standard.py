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


class DataScaledIntegrator(DoubleIntegrator):
    """x'' = p a, |a| <= 1, the scale p as each stage's model data: the minimum time to ``d`` from rest is sqrt(2 d / p),
    so d(T)/d(p) = -T / (2 p)."""

    data_width = 1

    def rhs(self, s, u, u_rate, p, data):  # noqa: ANN001, ANN201 - the model protocol
        return jnp.stack([s[1], data[0] * u[0], jnp.ones(())]), jnp.zeros(1)


@pytest.mark.parametrize('p', [1.0, 2.5])
def test_data_sensitivity_matches_the_closed_form(p: float) -> None:
    from trajopt_jax.envelope import data_sensitivity, with_data

    d = 10.0
    prob = Stages(
        DataScaledIntegrator(), np.array([0.0, 0.0, 0.0]), np.array([1.0]), (Leg((Point(d, None, None, tol=1e-3),), max_s=30.0),), segments=30
    )
    X, U = prob.path_seed(3.0)
    res = ipm.solve(prob, X, U, prob.x0(X[0, -1]), max_iter=300, verbose=False, data=with_data(prob, np.array([p])))
    assert res.converged, res.status
    assert float(res.X[0, -1]) == pytest.approx(math.sqrt(2 * d / p), rel=2e-3)
    grad = data_sensitivity(prob, res.X, res.U, res.state, chunk=7)  # (7: blocks that do not divide the 31 stages)
    assert grad[0] == pytest.approx(-float(res.obj) / (2 * p), rel=1e-4)


def test_refs_override_the_models_scales_and_leave_the_optimum() -> None:
    d = 10.0
    start, u0, legs = np.array([0.0, 0.0, 0.0]), np.array([1.0]), (Leg((Point(d, None, None, tol=1e-3),), max_s=30.0),)
    prob = Stages(DoubleIntegrator(), start, u0, legs, segments=30, refs={'x': 1.0, 'v': 5.0})
    assert prob.ref[0] == 1.0 and prob.ref[1] == 5.0
    X, U = prob.path_seed(3.0)
    res = ipm.solve(prob, X, U, prob.x0(X[0, -1]), max_iter=300, verbose=False)
    assert res.converged and float(res.X[0, -1]) == pytest.approx(math.sqrt(2 * d), rel=2e-3)
    with pytest.raises(ValueError, match='no such states'):
        Stages(DoubleIntegrator(), start, u0, legs, segments=30, refs={'height': 1.0})


def test_stop_ends_at_rest_and_matches_the_bang_bang_time() -> None:
    from trajopt_jax.events import Stop

    d = 10.0  # (from rest to rest with |a| <= 1: accelerate half the way, brake half: T = 2 sqrt(d))
    prob = Stages(
        DoubleIntegrator(), np.array([0.0, 0.0, 0.0]), np.array([1.0]), (Leg((Point(d, None, None, tol=1e-3), Stop()), max_s=30.0),), segments=30
    )
    X, U = prob.path_seed(3.0)
    res = ipm.solve(prob, X, U, prob.x0(X[0, -1]), max_iter=300, verbose=False)
    assert res.converged, res.status
    assert float(res.X[0, -1]) == pytest.approx(2.0 * math.sqrt(d), rel=5e-3)
    assert abs(float(res.X[-1, 1] * prob.ref[1])) < 1e-3


def test_trajectory_gives_times_and_named_unscaled_states() -> None:
    d = 10.0
    prob = Stages(DoubleIntegrator(), np.array([0.0, 0.0, 0.0]), np.array([1.0]), (Leg((Point(d, None, None, tol=1e-3),), max_s=30.0),), segments=30)
    X, U = prob.path_seed(3.0)
    tr = prob.trajectory(X)
    assert tr['t'][0] == 0.0 and tr['t'][-1] == pytest.approx(float(X[0, -1]))
    assert tr[prob.model.states[0].name][-1] == pytest.approx(float(X[-1, 0] * prob.ref[0]))
    assert len(tr['t']) == prob.N + 1 and set(tr['mission']) == {0}


def test_a_chained_trajectory_ends_one_mission_and_starts_the_next_at_its_own_start() -> None:
    from trajopt_jax.stages import Mission, chained

    leg = Leg((Point(10.0, None, None, tol=1e-3),), max_s=30.0)
    first = Mission(np.array([0.0, 0.0, 0.0]), np.array([1.0]), (leg,))
    second = Mission(np.array([5.0, 0.0, 0.0]), np.array([1.0]), (leg,))
    prob = chained(DoubleIntegrator(), [first, second], segments=10)
    X, U = prob.path_seed(3.0)
    tr = prob.trajectory(X)
    (k,) = prob.junctions
    x = tr[prob.model.states[0].name]
    assert len(tr['t']) == prob.N + 2 and list(tr['mission']).count(0) == k + 1 and list(tr['stage'][k : k + 2]) == [k, k]
    assert tr['mission'][k] == 0 and tr['t'][k] > 0.0 and x[k] == pytest.approx(float(X[k, 0] * prob.ref[0]))  # (the first's end)
    assert tr['mission'][k + 1] == 1 and tr['t'][k + 1] == 0.0 and x[k + 1] == pytest.approx(5.0)  # (the second's own start)
    assert np.all(np.diff(tr['t'][: k + 1]) > 0) and np.all(np.diff(tr['t'][k + 1 :]) > 0)


def test_seed_from_a_flown_trajectory() -> None:
    """Stages.seed from the exact optimum (full acceleration from rest: x = t^2 / 2): its states and controls land at
    the nodes and the solve starts there, converging at once."""
    d = 10.0
    T = math.sqrt(2 * d)
    start, u0, legs = np.array([0.0, 0.0, 0.0]), np.array([1.0]), (Leg((Point(d, None, None, tol=1e-3),), max_s=30.0),)
    prob = Stages(DoubleIntegrator(), start, u0, legs, segments=30)

    def state_at(_: int, w: float) -> np.ndarray:
        t = w * T
        return np.array([t * t / 2, t, t])

    X, U = prob.seed([T], state_at, lambda _, w: np.array([1.0]))
    assert np.allclose(X[:, 3], 1.0) and X[-1, 0] * prob.ref[0] == pytest.approx(d)  # (the controls, the end's position)
    res = ipm.solve(prob, X, U, prob.x0(X[0, -1]), max_iter=50, verbose=False)
    assert res.iterations <= 15 and float(res.X[0, -1]) == pytest.approx(T, rel=2e-3)
