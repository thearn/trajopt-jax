"""trajopt_jax.nested on the double integrator whose acceleration scale is model data: the minimum time to d from rest
is T(p) = sqrt(2 d / p) and the objective T / 10 (the cost state's scale), so with an outer cost c p the optimum is
p* = (sqrt(2 d) / (20 c))^(2/3)."""

import math

import numpy as np
import pytest

from tests.test_standard import DataScaledIntegrator, DoubleIntegrator
from trajopt_jax import ipm
from trajopt_jax._jax import jnp
from trajopt_jax.envelope import data_sensitivity, with_data
from trajopt_jax.events import Leg, Point
from trajopt_jax.nested import Nested, trust_region
from trajopt_jax.stages import Stages

D, C = 10.0, 0.1


class TwoColumnIntegrator(DoubleIntegrator):
    """x'' = a0 a1 u: the first column a stage's own (held), the second the design."""

    data_width = 2

    def rhs(self, s, u, u_rate, p, data):  # noqa: ANN001, ANN201 - the model protocol
        return jnp.stack([s[1], data[0] * data[1] * u[0], jnp.ones(())]), jnp.zeros(1)


def _problem(model: object) -> Stages:
    return Stages(model, np.array([0.0, 0.0, 0.0]), np.array([1.0]), (Leg((Point(D, None, None, tol=1e-3),), max_s=30.0),), segments=30)


def test_sensitivity_to_some_columns_holds_the_others() -> None:
    prob = _problem(TwoColumnIntegrator())
    base = with_data(prob, np.array([2.0, 1.0]))
    data = with_data(prob, np.array([1.5]), base, slice(1, 2))
    X, U = prob.path_seed(3.0)
    res = ipm.solve(prob, X, U, prob.x0(X[0, -1]), max_iter=300, verbose=False, data=data)
    assert res.converged and float(res.X[0, -1]) == pytest.approx(math.sqrt(2 * D / 3.0), rel=2e-3)
    grad = data_sensitivity(prob, res.X, res.U, res.state, columns=slice(1, 2))
    assert grad.shape == (1,) and grad[0] == pytest.approx(-float(res.obj) / (2 * 1.5), rel=1e-4)


def test_nested_trust_region_finds_the_closed_form_design() -> None:
    prob = _problem(DataScaledIntegrator())
    X, U = prob.path_seed(3.0)
    inner = Nested(prob, X, U, lambda p: p, max_iter=300)
    p0 = np.array([1.0])

    def fg(z: np.ndarray) -> tuple[float, np.ndarray]:
        obj, grad = inner(z * p0)
        return obj + C * float(z[0] * p0[0]), (grad + C) * p0

    out = trust_region(fg, np.ones(1), np.array([0.5]), np.array([20.0]), maxiter=40)
    p_star = (math.sqrt(2 * D) / (20 * C)) ** (2 / 3)
    assert out['z'][0] * p0[0] == pytest.approx(p_star, rel=2e-3)
    assert any(line['step'] for line in inner.log)  # (the far designs reached in warm steps of at most MAX_STEP)


def test_a_design_variable_at_zero_takes_no_continuation_steps() -> None:
    # (sized against its own value, a variable at 0 took a 1e-7 step in thousands of warm solves: the whoop's prop pitch,
    # 2026-11-25)
    prob = _problem(TwoColumnIntegrator())
    X, U = prob.path_seed(3.0)
    inner = Nested(prob, X, U, lambda p: 1.0 + p, base=with_data(prob, np.array([2.0, 1.0])), columns=slice(1, 2), max_iter=300)
    inner(np.zeros(1))
    inner(np.array([1e-3]))
    assert len(inner.log) == 2 and not any(line['step'] for line in inner.log)


def test_a_nan_design_gradient_stops_the_run() -> None:
    prob = _problem(DataScaledIntegrator())
    X, U = prob.path_seed(3.0)
    inner = Nested(prob, X, U, lambda p: jnp.sqrt(p - 1.0), max_iter=50)  # (sqrt at 0: an infinite slope)
    with pytest.raises(FloatingPointError):
        inner(np.array([1.0]))


def test_check_agrees_with_the_re_solves_and_the_closed_form() -> None:
    prob = _problem(DataScaledIntegrator())
    X, U = prob.path_seed(3.0)
    out = Nested(prob, X, U, lambda p: p, max_iter=300).check(np.array([2.0]), np.array([0.05]))
    exact = -0.5 * math.sqrt(2 * D / 2.0) / 2.0 / 10.0 * 0.05  # (d(T/10)/dp along the step: T = sqrt(2 d / p))
    assert out['envelope'] == pytest.approx(exact, rel=5e-3) and out['resolves'] == pytest.approx(exact, rel=5e-3)


def test_optimise_runs_to_the_bound_a_falling_objective_meets() -> None:
    prob = _problem(DataScaledIntegrator())
    X, U = prob.path_seed(3.0)
    out = Nested(prob, X, U, lambda p: p, max_iter=300).optimise(np.array([1.0]), np.array([0.5]), np.array([2.0]), maxiter=20)
    assert out['design'][0] == pytest.approx(2.0) and out['objective'] == pytest.approx(math.sqrt(2 * D / 2.0) / 10.0, rel=2e-3)


def _two_missions() -> Stages:
    """Two missions chained on the data-scaled integrator: 10 m from rest, then 5 m from rest (from x = 5)."""
    from trajopt_jax.stages import Mission, chained

    leg = Leg((Point(D, None, None, tol=1e-3),), max_s=30.0)
    first = Mission(np.array([0.0, 0.0, 0.0]), np.array([1.0]), (leg,))
    second = Mission(np.array([5.0, 0.0, 0.0]), np.array([1.0]), (leg,))
    return chained(DataScaledIntegrator(), [first, second], segments=20)


def test_each_missions_objective_and_gradient_match_the_closed_form() -> None:
    prob = _two_missions()
    X, U = prob.path_seed(3.0)
    inner = Nested(prob, X, U, lambda p: p, max_iter=300, missions=True)
    p = 2.0
    objs, grads = inner.parts(np.array([p]))
    t = np.array([math.sqrt(2 * D / p), math.sqrt(2 * (D - 5.0) / p)]) / 10.0
    np.testing.assert_allclose(objs, t, rtol=3e-3)
    np.testing.assert_allclose(grads[:, 0], -t / (2 * p), rtol=3e-3)  # (d/dp sqrt(2 d / p) = -T / (2 p))
    obj, grad = inner(np.array([p]))
    assert objs.sum() == pytest.approx(obj, rel=1e-9) and grads.sum(0) == pytest.approx(grad, rel=1e-6)


def test_cvar_is_the_mean_of_the_worst_share() -> None:
    from trajopt_jax.nested import cvar

    objs, grads = np.array([1.0, 4.0, 2.0, 3.0]), np.eye(4)
    assert cvar(objs, grads, 1.0)[0] == pytest.approx(2.5)  # (the mean)
    v, g = cvar(objs, grads, 0.25)  # (the worst)
    assert v == pytest.approx(4.0) and g == pytest.approx([0, 1, 0, 0])
    v, g = cvar(objs, grads, 0.375)  # (the worst and half the next: (4 + 0.5 x 3) / 1.5)
    assert v == pytest.approx(5.5 / 1.5) and g == pytest.approx([0, 1 / 1.5, 0, 0.5 / 1.5])
    with pytest.raises(ValueError):
        cvar(objs, grads, 0.0)


def test_optimise_on_the_worst_mission_runs_it_to_the_bound() -> None:
    prob = _two_missions()
    X, U = prob.path_seed(3.0)
    inner = Nested(prob, X, U, lambda p: p, max_iter=300, missions=True)
    out = inner.optimise(np.array([1.0]), np.array([0.5]), np.array([2.0]), maxiter=20, risk=0.5)
    assert out['design'][0] == pytest.approx(2.0, rel=1e-6)  # (both times fall as p grows: the bound)
    assert out['objective'] == pytest.approx(math.sqrt(2 * D / 2.0) / 10.0, rel=3e-3)  # (the 10 m mission's, the worse)
