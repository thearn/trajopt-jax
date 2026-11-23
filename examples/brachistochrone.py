"""The brachistochrone with trajopt-jax: a vehicle model, one leg ending at a point, the interior-point solve, then
mesh refinement; the descent time against the cycloid's.

python examples/brachistochrone.py
"""

from __future__ import annotations

import math

import numpy as np

from trajopt_jax import ipm
from trajopt_jax._jax import jnp
from trajopt_jax.events import Leg, Point
from trajopt_jax.refine import solve_refined
from trajopt_jax.specs import ControlSpec, StateSpec
from trajopt_jax.stages import Stages

G = 9.81


class Bead:
    """A bead from rest under gravity: x' = v sin(theta), y' = -v cos(theta), v' = g cos(theta); the cost is time.
    A model declares its states (with scales), controls (with bounds) and outputs, and gives its dynamics (``rhs``), its
    position and velocity (``pose``, for events such as Point) and a state from a position and velocity (for seeds)."""

    states = (StateSpec('x', 'm', 10.0), StateSpec('y', 'm', 10.0), StateSpec('v', 'm/s', 10.0), StateSpec('cost', 's', 10.0))
    controls = (ControlSpec('theta', 0.0, math.pi),)
    outputs = ('zero',)
    constraints = ()
    data_width = 0
    seed_integrated = ('cost',)
    angles = ()

    def params(self):  # noqa: ANN201 - no design variables
        return jnp.zeros(0)

    def rhs(self, s, u, u_rate, p, data):  # noqa: ANN001, ANN201 - the model protocol
        th = u[0]
        return jnp.stack([s[2] * jnp.sin(th), -s[2] * jnp.cos(th), G * jnp.cos(th), jnp.ones(())]), jnp.zeros(1)

    def pose(self, s):  # noqa: ANN001, ANN201 - the model protocol
        return jnp.stack([s[0], s[1], 0.0]), jnp.zeros(3)

    def state_from(self, pos: np.ndarray, vel: np.ndarray, base: np.ndarray) -> np.ndarray:
        out = np.array(base, float)
        out[:3] = pos[0], pos[1], max(float(np.linalg.norm(vel)), 0.1)
        return out


def cycloid_time(D: float, H: float) -> float:
    lo, hi = 0.1, 2 * math.pi - 0.1
    for _ in range(100):  # (bisect (phi - sin phi) / (1 - cos phi) = D / H)
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if (mid - math.sin(mid)) / (1 - math.cos(mid)) < D / H else (lo, mid)
    return lo * math.sqrt(H / (1 - math.cos(lo)) / G)


def main(D: float = 10.0, H: float = 5.0) -> dict:
    start, u0 = np.zeros(4), np.array([0.5])
    legs = (Leg((Point(D, -H, None, tol=1e-3),), max_s=30.0),)

    def build(legs: tuple[Leg, ...]) -> Stages:
        return Stages(Bead(), start, u0, legs, segments=4)

    prob = build(legs)
    X, U = prob.path_seed(speed=5.0)
    coarse = ipm.solve(prob, X, U, prob.x0(X[0, -1]), max_iter=300, verbose=False)
    fine, res, passes = solve_refined(build, legs, coarse.X, coarse.U, passes=5, tol=1e-5, max_iter=300, verbose=False)
    exact = cycloid_time(D, H)
    s = res.X[:, :4] * fine.ref
    out = dict(cycloid=exact, coarse=float(coarse.X[0, -1]), refined=float(res.X[0, -1]), stages=fine.N, path=s[:, :2])
    print(f'cycloid {exact:.5f} s; 4 stages {out["coarse"]:.5f} s; refined to {fine.N} stages {out["refined"]:.5f} s')
    return out


if __name__ == '__main__':
    main()
