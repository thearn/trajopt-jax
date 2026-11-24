"""A low-thrust transfer between two planets with trajopt-jax: arrive on a given date, the launch date solved for.

The spacecraft starts at the origin planet (Earth by default) on a free launch date inside a window, leaving with at
most the launch vehicle's hyperbolic excess speed, flies a heliocentric two-body arc under a bounded electric thrust,
and meets the target planet on the arrival date: a rendezvous (its position and velocity) or a flyby (its position).
The objective is the least propellant. Planet positions come from astropy's built-in ephemeris: the origin's, over
the launch window, as Chebyshev series fitted once and evaluated in JAX (smooth and differentiable in the launch
date); the target's at the one arrival date. Patched conics: no escape spiral or capture, no other planet's gravity.

What it shows of the library: a start pinned by constraints (``Stages.start_free`` and ``start_events``: the launch
date and the departure state free, held to the origin's ephemeris), a free leg duration, path constraints (the thrust
bound), and end events (the rendezvous).

python examples/transfer.py [ORIGIN] [TARGET] [ARRIVAL_DATE] [--flyby] [--thrust N] [--isp S] [--mass KG] [--vinf KM_S]
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from trajopt_jax import ipm
from trajopt_jax._jax import jax, jnp
from trajopt_jax.events import Leg
from trajopt_jax.radau import radau_iia3
from trajopt_jax.specs import ControlSpec, StateSpec
from trajopt_jax.stages import Stages

AU_M, YEAR_S, G0 = 1.495978707e11, 365.25 * 86400.0, 9.80665
MU_SUN = 4 * math.pi**2  # AU^3/yr^2 (Kepler's third law in these units)
ACCEL = YEAR_S**2 / AU_M  # (AU/yr^2) per (m/s^2)
KMS = YEAR_S / AU_M * 1000.0  # (AU/yr) per (km/s)
PLANETS = ('mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')
HOMOTOPY = (0.0, 0.5, 0.9, 1.0)  # the cost's fuel weight, solved in turn (0: throttle^2, 1: propellant)
EPHEMERIS = 'builtin'  # (astropy's; 'de440s' adds Pluto's barycentre, a download)
_EPS = math.radians(23.4392911)  # the obliquity of the ecliptic at J2000
ECLIPTIC = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(_EPS), math.sin(_EPS)], [0.0, -math.sin(_EPS), math.cos(_EPS)]])


# ---------------------------------------------------------------------------------------------------- ephemerides
def heliocentric(body: str, iso_dates: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:
    """(position AU, velocity AU/yr) of ``body`` relative to the Sun at TDB ``iso_dates``, in ecliptic axes (ICRS turned
    by the obliquity: the planets' orbits lie within a few degrees of the x-y plane)."""
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time

    t = Time(iso_dates, scale='tdb')
    p, v = get_body_barycentric_posvel(body, t, ephemeris=EPHEMERIS)
    ps, vs = get_body_barycentric_posvel('sun', t, ephemeris=EPHEMERIS)
    pos = (p - ps).xyz.to('au').value.T @ ECLIPTIC.T
    vel = (v - vs).xyz.to('au/d').value.T @ ECLIPTIC.T * 365.25
    return np.atleast_2d(pos), np.atleast_2d(vel)


@dataclass(frozen=True)
class Chebyshev:
    """Position and velocity of a planet over [t_lo, t_hi] (years from the arrival date) as Chebyshev series, fitted to
    the ephemeris at Chebyshev nodes; ``__call__`` evaluates them in JAX by Clenshaw's recurrence (a ``scan``)."""

    t_lo: float
    t_hi: float
    coef: np.ndarray  # (degree + 1, 6): position then velocity

    @classmethod
    def fit(cls, body: str, arrival: str, t_lo: float, t_hi: float, degree: int) -> Chebyshev:
        from astropy.time import Time, TimeDelta

        k = np.arange(4 * degree)
        x = np.cos(math.pi * (k + 0.5) / len(k))  # (Chebyshev nodes: no Runge oscillation)
        t = t_lo + 0.5 * (x + 1.0) * (t_hi - t_lo)
        dates = Time(arrival, scale='tdb') + TimeDelta(t * 365.25, format='jd')
        pos, vel = heliocentric(body, dates.iso)
        coef = np.polynomial.chebyshev.chebfit(x, np.hstack([pos, vel]), degree)
        return cls(t_lo, t_hi, coef)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        x = (2.0 * t - self.t_lo - self.t_hi) / (self.t_hi - self.t_lo)
        c = jnp.asarray(self.coef)

        def step(carry: tuple, ck: jnp.ndarray) -> tuple:
            b1, b2 = carry
            return (2.0 * x * b1 - b2 + ck, b1), None

        (b1, b2), _ = jax.lax.scan(step, (jnp.zeros(6), jnp.zeros(6)), c[:0:-1])
        return x * b1 - b2 + c[0]


# ---------------------------------------------------------------------------------------------------------- model
class Transfer:
    """Heliocentric two-body flight under thrust: position (AU), velocity (AU/yr), mass (a fraction of the start's),
    time (years from the arrival date), cost. The controls are the throttle (0 to 1, a fraction of the engine's
    maximum thrust) and the thrust's direction as two angles in ecliptic axes, alpha in the plane and beta out of it.
    The propellant's flow is then linear in the throttle, smooth where the thrust is zero (a thrust vector u with the
    flow as |u| has a corner there, at which the least-propellant solve failed), and the direction has no free scale (a
    vector d with the thrust along d / |d| left |d| undetermined, and the solve stalled, 2026-11-23)."""

    states = (
        *(StateSpec(n, 'AU', 1.0) for n in ('x', 'y', 'z')),
        *(StateSpec(n, 'AU/yr', 6.0) for n in ('vx', 'vy', 'vz')),
        StateSpec('m', '', 1.0),
        StateSpec('t', 'yr', 1.0),
        StateSpec('cost', '', 0.1),
    )
    controls = (ControlSpec('throttle', 0.0, 1.0), ControlSpec('alpha', -40.0, 40.0), ControlSpec('beta', -1.2, 1.2))
    outputs = ('throttle',)
    constraints = ()
    data_width = 0

    def __init__(self, thrust_n: float, isp_s: float, mass_kg: float, fuel_weight: float = 1.0) -> None:
        self.a_max = thrust_n / mass_kg * ACCEL  # AU/yr^2 at full mass
        self.mdot_max = thrust_n / (isp_s * G0 * mass_kg) * YEAR_S  # mass fraction per year at full throttle
        # (the cost's rate: propellant (the throttle) at fuel_weight 1, the energy-like throttle^2 at 0. Least
        # propellant flies full thrust or none, a solve from a rough seed creeps; solved at 0 first, the smooth
        # problem's answer seeds the next weight)
        self.fuel_weight = fuel_weight

    def params(self) -> jnp.ndarray:
        return jnp.zeros(0)

    def rhs(self, s: jnp.ndarray, u: jnp.ndarray, u_rate: jnp.ndarray, p: jnp.ndarray, data: jnp.ndarray) -> tuple:
        del u_rate, p, data
        r, v, m = s[0:3], s[3:6], s[6]
        throttle, alpha, beta = u[0], u[1], u[2]
        d = jnp.stack([jnp.cos(beta) * jnp.cos(alpha), jnp.cos(beta) * jnp.sin(alpha), jnp.sin(beta)])
        acc = -MU_SUN * r / (r @ r) ** 1.5 + self.a_max * throttle * d / m
        mdot = self.mdot_max * throttle
        cost = self.mdot_max * (self.fuel_weight * throttle + (1.0 - self.fuel_weight) * throttle * throttle)
        return jnp.concatenate([v, acc, jnp.stack([-mdot, jnp.ones(()), cost])]), jnp.stack([throttle])

    def pose(self, s: jnp.ndarray) -> tuple:
        return s[0:3], s[3:6]


def departure(eph: Chebyshev) -> type:
    """The start event's kind for the origin's ephemeris ``eph``: the start within ``tol`` of the origin's position on
    its launch date ``t`` (the time state), its speed relative to the origin at most ``vinf``, ``t`` inside the window."""

    @dataclass(frozen=True)
    class Departure:
        vinf: float  # AU/yr
        tol: float  # AU
        t_lo: float
        t_hi: float
        width: ClassVar[int] = 4

        def columns(self) -> list[float]:
            return [self.vinf, self.tol, self.t_lo, self.t_hi]

        @staticmethod
        def rows(s: jnp.ndarray, pos: jnp.ndarray, vel: jnp.ndarray, e: jnp.ndarray) -> list:
            pv = eph(s[7])
            dr, dv = pos - pv[:3], vel - pv[3:]
            return [*(dr - e[1]), *(-dr - e[1]), (dv @ dv - e[0] ** 2) / 10.0, e[2] - s[7], s[7] - e[3]]

    return Departure


@dataclass(frozen=True)
class Arrival:
    """The end event: the target's position within ``tol_r`` (AU), its velocity within ``tol_v`` (AU/yr; a flyby sets
    it large), on the arrival date (the time state at 0 within ``tol_t`` years)."""

    target: tuple[float, ...]  # position then velocity at arrival
    tol_r: float
    tol_v: float
    tol_t: float = 1e-6
    width: ClassVar[int] = 9

    def columns(self) -> list[float]:
        return [*self.target, self.tol_r, self.tol_v, self.tol_t]

    @staticmethod
    def rows(s: jnp.ndarray, pos: jnp.ndarray, vel: jnp.ndarray, e: jnp.ndarray) -> list:
        dr, dv = pos - e[0:3], vel - e[3:6]
        return [*(dr - e[6]), *(-dr - e[6]), *(dv - e[7]), *(-dv - e[7]), s[7] - e[8], -s[7] - e[8]]


# ----------------------------------------------------------------------------------------------------------- seed
def start_controls(v0: np.ndarray) -> np.ndarray:
    """Throttle 0 and the direction along the velocity (the first knot's controls are fixed: no thrust at launch)."""
    return np.array([0.0, math.atan2(v0[1], v0[0]), math.asin(v0[2] / np.linalg.norm(v0))])


def seed(prob: Stages, r0: np.ndarray, v0: np.ndarray, r1: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """(X, U): a prograde sweep from ``r0`` to ``r1`` over ``tau`` years in the origin's orbital plane, radius and angle
    interpolated (whole revolutions added so the mean angular rate is the mean motion), mass and controls constant."""
    h = np.cross(r0, v0)
    k = h / np.linalg.norm(h)
    e1 = r0 / np.linalg.norm(r0)
    e2 = np.cross(k, e1)
    rho0, rho1 = np.linalg.norm(r0), np.linalg.norm(r1)
    p1 = r1 - (r1 @ k) * k
    dth = math.atan2(p1 @ e2, p1 @ e1) % (2 * math.pi)
    n_mean = math.sqrt(MU_SUN / (0.5 * (rho0 + rho1)) ** 3)
    dth += 2 * math.pi * max(0, round((n_mean * tau - dth) / (2 * math.pi)))

    def at(w: float) -> np.ndarray:
        rho, th = rho0 + w * (rho1 - rho0), w * dth
        return rho * (math.cos(th) * e1 + math.sin(th) * e2) + w * (r1 @ k) * k

    def state(w: float) -> np.ndarray:
        dw = 1e-6
        pos = at(w)
        vel = (at(min(w + dw, 1.0)) - at(max(w - dw, 0.0))) / ((min(w + dw, 1.0) - max(w - dw, 0.0)) * tau)
        return np.concatenate([pos, vel, [1.0, -tau * (1.0 - w), 0.0]]) / prob.ref

    c = np.concatenate([[0.0], np.asarray(radau_iia3()[0])])
    N = prob.N
    u0 = start_controls(v0)
    X, U = np.zeros((N + 1, prob.nx)), np.zeros((N, prob.nu))

    def controls(y: np.ndarray, prev: np.ndarray) -> np.ndarray:  # (thrust along the velocity, alpha unwrapped)
        v = y[3:6] * prob.ref[3:6]
        a = math.atan2(v[1], v[0])
        a -= 2 * math.pi * round((a - prev[1]) / (2 * math.pi))
        return np.array([0.0, a, math.asin(v[2] / np.linalg.norm(v))])

    k_prev = u0
    for k_ in range(N):
        w0, fk = k_ / N, 1.0 / N
        Y = [state(w0 + ck * fk) for ck in c]
        k_next = controls(state(w0 + fk), k_prev)
        X[k_] = np.concatenate([Y[0], k_prev, [tau]])
        U[k_] = np.concatenate([k_next, [0.0], *Y[1:]])
        k_prev = k_next
    X[N] = np.concatenate([state(1.0), k_prev, [tau]])
    return X, U


# ----------------------------------------------------------------------------------------------------------- solve
def hohmann_years(origin: str, target: str, iso: str) -> float:
    r0 = np.linalg.norm(heliocentric(origin, [iso])[0][0])
    r1 = np.linalg.norm(heliocentric(target, [iso])[0][0])
    return 0.5 * math.sqrt((0.5 * (r0 + r1)) ** 3)


@dataclass(frozen=True)
class Spacecraft:
    thrust_n: float = 0.5
    isp_s: float = 3000.0
    mass_kg: float = 1500.0
    vinf_kms: float = 3.0  # the launch vehicle's hyperbolic excess speed (C3 = 9 km^2/s^2)


DEFAULT_CRAFT = Spacecraft()


def solve(
    origin: str = 'earth',
    target: str = 'mars',
    arrival: str = '2033-07-08',
    flyby: bool = False,
    craft: Spacecraft = DEFAULT_CRAFT,
    segments: int = 120,
    window: tuple[float, float] | None = None,
    tau_seed: float | None = None,
) -> dict:
    """The least-propellant transfer from ``origin`` to ``target`` arriving on ``arrival`` (ISO date, TDB). ``window``:
    the flight time's bounds in years (default 0.5 to 1.6 Hohmann transfers); ``tau_seed``: the seed's flight time
    (default the Hohmann transfer's, held inside the window). A long window has several launch opportunities, and the
    solve settles near the one its seed starts from."""
    from astropy.time import Time, TimeDelta

    t_h = hohmann_years(origin, target, arrival)
    lo, hi = window or (0.5 * t_h, 1.6 * t_h)
    pad, degree = 0.1 * (hi - lo), 24 + int(40 * (hi - lo) / max(t_h, 0.2))
    body = 'earth-moon-barycenter' if origin == 'earth' else origin  # (the patched conic's origin: Earth's with its Moon)
    eph = Chebyshev.fit(body, arrival, -hi - pad, -lo + pad, degree=degree)
    rT, vT = (a[0] for a in heliocentric(target, [arrival]))
    tau0 = min(max(tau_seed or t_h, lo), hi)
    pv0 = np.asarray(eph(jnp.asarray(-tau0)))
    start = np.concatenate([pv0, [1.0, -tau0, 0.0]])
    leg = Leg((Arrival((*rT, *vT), 1e-6, 1e3 if flyby else 1e-5),), min_s=lo, max_s=hi, segments=segments)
    dep = departure(eph)(craft.vinf_kms * KMS, 1e-5, -hi, -lo)

    def problem(fuel_weight: float) -> Stages:
        model = Transfer(craft.thrust_n, craft.isp_s, craft.mass_kg, fuel_weight)
        free = ('x', 'y', 'z', 'vx', 'vy', 'vz', 't')
        return Stages(model, start, start_controls(pv0[3:]), (leg,), segments=segments, start_free=free, start_events=(dep,))

    # (a homotopy in the cost's fuel weight, each solve seeded by the last: the smooth problem first, then steps toward
    # least propellant; one jump from 0 to 1 crept to the iteration cap, 2026-11-23)
    prob = problem(HOMOTOPY[0])
    X, U = seed(prob, pv0[:3], pv0[3:], rT, tau0)
    steps = []
    for w in HOMOTOPY:
        prob = problem(w)
        res = ipm.solve(prob, X, U, prob.x0(float(X[0, -1])), max_iter=500, verbose=False)
        steps.append((w, res.status, int(res.iterations)))
        X, U = res.X, res.U
    model = prob.model
    s = res.X[:, : prob.ns] * prob.ref
    t_launch = float(s[0, 7])
    launch = (Time(arrival, scale='tdb') + TimeDelta(t_launch * 365.25, format='jd')).iso[:10]
    pv_launch = np.asarray(eph(jnp.asarray(t_launch)))
    out = dict(origin=origin, target=target, arrival=arrival, mode='flyby' if flyby else 'rendezvous', status=res.status,
               converged=bool(res.converged), iterations=int(res.iterations), launch=launch, flight_days=-t_launch * 365.25,
               propellant_kg=float(1.0 - s[-1, 6]) * craft.mass_kg, homotopy=steps, vinf_kms=float(np.linalg.norm(s[0, 3:6] - pv_launch[3:]) / KMS),
               arrival_miss_km=float(np.linalg.norm(s[-1, :3] - rT) * AU_M / 1000))  # fmt: skip
    out['check'] = propagate_check(model, prob, res)
    return {**out, 'X': res.X, 'U': res.U, 'prob': prob}


def propagate_check(model: Transfer, prob: Stages, res: ipm.IPMResult) -> dict:
    """The solution re-flown by SciPy's integrator (DOP853, tight tolerances) from its own start with its controls
    (linear in each stage): the end's distance from the transcription's (km) and its speed difference (m/s)."""
    from scipy.integrate import solve_ivp

    ns, nc, N = prob.ns, prob.nc, prob.N
    s = res.X[:, :ns] * prob.ref
    tau = float(res.X[0, -1])
    t_nodes = np.linspace(0.0, tau, N + 1)
    u_nodes = np.vstack([res.X[:N, ns : ns + nc], res.U[-1:, :nc]])
    f = jax.jit(lambda y, u: model.rhs(y, u, jnp.zeros(nc), jnp.zeros(0), jnp.zeros(0))[0])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        u = np.array([np.interp(t, t_nodes, u_nodes[:, i]) for i in range(nc)])
        return np.asarray(f(jnp.asarray(y), jnp.asarray(u)))

    sol = solve_ivp(rhs, (0.0, tau), s[0], method='DOP853', rtol=1e-10, atol=1e-12)
    end = sol.y[:, -1]
    return dict(
        position_km=float(np.linalg.norm(end[:3] - s[-1, :3]) * AU_M / 1000), speed_ms=float(np.linalg.norm(end[3:6] - s[-1, 3:6]) / KMS * 1000)
    )


def main(argv: list[str]) -> dict:
    pos = [a for a in argv if not a.startswith('--') and not _is_value(argv, a)]
    origin, target = (pos + ['earth', 'mars'])[:2] if len(pos) < 2 else pos[:2]
    arrival = pos[2] if len(pos) > 2 else '2033-07-08'
    for body in (origin, target):
        if body not in PLANETS:
            raise SystemExit(f'{body!r}: one of {", ".join(PLANETS)}')

    def opt(name: str, default: float) -> float:
        return float(argv[argv.index(name) + 1]) if name in argv else default

    craft = Spacecraft(opt('--thrust', 0.5), opt('--isp', 3000.0), opt('--mass', 1500.0), opt('--vinf', 3.0))
    out = solve(origin, target, arrival, '--flyby' in argv, craft)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k not in ('X', 'U', 'prob')})
    return out


def _is_value(argv: list[str], a: str) -> bool:
    i = argv.index(a)
    return i > 0 and argv[i - 1] in ('--thrust', '--isp', '--mass', '--vinf')


if __name__ == '__main__':
    main(sys.argv[1:])
