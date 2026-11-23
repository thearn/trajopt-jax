"""A low-thrust rendezvous between two planets on circular, coplanar orbits, the departure date solved for.

The spacecraft leaves the inner planet on a free date, at most at the launch vehicle's hyperbolic excess speed, flies a
heliocentric arc under a bounded engine, and meets the outer planet (its position and velocity) on a fixed arrival
date, using the least propellant. The planets' positions are one line of trigonometry each, so the problem is small
(seven states, two controls, planar) and solves in seconds. With a strong engine and no launch speed the answer tends
to the Hohmann transfer: its total speed change, flight time and the phase angle at launch are the checks.

What it shows of the library: a start pinned by constraints (``Stages.start_free``, ``start_events``: the departure
date and state free, held to the inner planet's orbit), a free leg duration, an end event (the rendezvous), and the
per-stage model data (``ipm.solve(..., data=...)``) carrying the homotopy's weight, so one compiled solver serves
every step.

python examples/rendezvous.py [--hohmann]
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from trajopt_jax import ipm
from trajopt_jax._jax import jax, jnp
from trajopt_jax.events import Leg
from trajopt_jax.radau import radau_iia3
from trajopt_jax.specs import ControlSpec, PathConstraint, StateSpec
from trajopt_jax.stages import Stages

AU_M, YEAR_S, G0 = 1.495978707e11, 365.25 * 86400.0, 9.80665
MU = 4 * math.pi**2  # the Sun's, AU^3/yr^2
ACCEL = YEAR_S**2 / AU_M  # (AU/yr^2) per (m/s^2)
KMS = YEAR_S / AU_M * 1000.0  # (AU/yr) per (km/s)


@dataclass(frozen=True)
class Orbit:
    """A circular orbit of radius ``radius`` (AU), at angle ``phase`` (rad) on the arrival date (t = 0)."""

    radius: float
    phase: float

    @property
    def rate(self) -> float:
        return math.sqrt(MU / self.radius**3)  # rad/yr

    def state(self, t: jnp.ndarray) -> jnp.ndarray:
        """(x, y, vx, vy) at ``t`` years from the arrival date."""
        th = self.phase + self.rate * t
        r, v = self.radius, self.radius * self.rate
        return jnp.stack([r * jnp.cos(th), r * jnp.sin(th), -v * jnp.sin(th), v * jnp.cos(th)])


class Craft:
    """Planar two-body flight under thrust. States: position (AU), velocity (AU/yr), mass (a fraction of the start's),
    time (years from the arrival date), cost. Controls: the throttle (0 to 1) and the thrust's angle ``alpha`` from the
    x axis. The per-stage model data is the cost's propellant weight w: the cost's rate is w throttle + (1 - w)
    throttle^2, times the full-throttle flow."""

    states = (
        StateSpec('x', 'AU', 1.0),
        StateSpec('y', 'AU', 1.0),
        StateSpec('vx', 'AU/yr', 6.0),
        StateSpec('vy', 'AU/yr', 6.0),
        StateSpec('m', '', 1.0),
        StateSpec('t', 'yr', 1.0),
        StateSpec('cost', '', 0.1),
    )
    controls = (ControlSpec('throttle', 0.0, 1.0), ControlSpec('alpha', -40.0, 40.0))
    outputs = ('throttle', 'm')
    # (the mass kept above a fifth of the start's: without it an iterate of a strong engine burned it to zero, the
    # thrust's acceleration T / m blew up and the solve failed, 2026-11-23)
    constraints = (PathConstraint('m', 0.2, None),)
    data_width = 1

    def __init__(self, accel_au_yr2: float, flow_per_yr: float, time_weight: float = 0.0) -> None:
        self.a_max, self.mdot_max, self.time_weight = accel_au_yr2, flow_per_yr, time_weight

    def params(self) -> jnp.ndarray:
        return jnp.zeros(0)

    def rhs(self, s: jnp.ndarray, u: jnp.ndarray, u_rate: jnp.ndarray, p: jnp.ndarray, data: jnp.ndarray) -> tuple:
        del u_rate, p
        r, v, m, w = s[0:2], s[2:4], s[4], data[0]
        throttle, alpha = u[0], u[1]
        acc = -MU * r / (r @ r) ** 1.5 + self.a_max * throttle / m * jnp.stack([jnp.cos(alpha), jnp.sin(alpha)])
        flow = self.mdot_max * throttle
        cost = self.mdot_max * (w * throttle + (1.0 - w) * throttle * throttle) + self.time_weight
        return jnp.concatenate([v, acc, jnp.stack([-flow, jnp.ones(()), cost])]), jnp.stack([throttle, m])

    def pose(self, s: jnp.ndarray) -> tuple:
        return jnp.concatenate([s[0:2], jnp.zeros(1)]), jnp.concatenate([s[2:4], jnp.zeros(1)])


def departure(origin: Orbit) -> type:
    """The start event's kind for ``origin``: the start on its orbit at the departure date (the time state), the speed
    relative to it at most ``vinf``, the date inside the window."""

    @dataclass(frozen=True)
    class Departure:
        vinf: float  # AU/yr
        t_lo: float
        t_hi: float
        tol: float = 1e-7
        width: ClassVar[int] = 4

        def columns(self) -> list[float]:
            return [self.vinf, self.t_lo, self.t_hi, self.tol]

        @staticmethod
        def rows(s: jnp.ndarray, pos: jnp.ndarray, vel: jnp.ndarray, e: jnp.ndarray) -> list:
            o = origin.state(s[5])
            dr, dv = pos[:2] - o[:2], vel[:2] - o[2:]
            return [*(dr - e[3]), *(-dr - e[3]), dv @ dv - e[0] ** 2 - 1e-12, e[1] - s[5], s[5] - e[2]]

    return Departure


@dataclass(frozen=True)
class Rendezvous:
    """The end event: the target's position and velocity at the arrival date (t = 0)."""

    target: tuple[float, float, float, float]
    tol: float = 1e-7
    width: ClassVar[int] = 5

    def columns(self) -> list[float]:
        return [*self.target, self.tol]

    @staticmethod
    def rows(s: jnp.ndarray, pos: jnp.ndarray, vel: jnp.ndarray, e: jnp.ndarray) -> list:
        d = jnp.concatenate([pos[:2], vel[:2]]) - e[0:4]
        return [*(d - e[4]), *(-d - e[4]), s[5] - e[4], -s[5] - e[4]]


@dataclass(frozen=True)
class Case:
    """The problem's numbers. The default: Earth to Mars at their mean radii, Mars placed on the arrival date so the
    Hohmann transfer's departure falls inside the window; a 1500 kg craft, 0.5 N at 3000 s, launched at up to 3 km/s."""

    origin_au: float = 1.0  # the departure planet's orbit radius
    target_au: float = 1.524  # the target's (smaller for an inward transfer, e.g. 0.723 for Venus)
    arrive_phase_deg: float = 180.0  # the target's angle on the arrival date
    lead_deg: float = 0.0  # the departure planet's angle on the arrival date, from where Hohmann would need it
    thrust_n: float = 0.5
    isp_s: float = 3000.0
    mass_kg: float = 1500.0
    vinf_kms: float = 3.0
    segments: int = 60
    window: tuple[float, float] = (0.3, 1.2)  # years of flight
    # the cost's propellant weight, solved in turn (0: throttle^2, smooth; 1: propellant). A strong engine starts at 1:
    # throttle^2 rewards spreading the thrust over the longest flight allowed, and the next steps stayed there (the
    # Hohmann check went to the window's 0.9 yr limit at 6.4-6.8 km/s, 2026-11-23)
    homotopy: tuple[float, ...] = (0.0, 0.5, 1.0)
    # a tie-break added to the cost's rate (per year): with no launch speed the craft may leave early and coast beside
    # the planet, every departure date before the burn costing the same, a flat direction the solve slid along to the
    # window's edge; a small charge on flight time picks the shortest
    time_weight: float = 0.0


def hohmann(case: Case) -> dict:
    """The impulsive two-burn transfer between the orbits: speed changes (km/s), flight time (yr), and the target's
    lead over the inner planet at departure (deg)."""
    r1, r2 = case.origin_au, case.target_au
    a = 0.5 * (r1 + r2)
    v1, v2 = math.sqrt(MU / r1), math.sqrt(MU / r2)
    dv1 = abs(math.sqrt(MU / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1))  # (abs: an inward transfer brakes)
    dv2 = abs(math.sqrt(MU / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2))))
    tof = math.pi * math.sqrt(a**3 / MU)
    lead = math.degrees(math.pi - math.sqrt(MU / r2**3) * tof)
    return dict(dv_kms=(dv1 + dv2) / KMS, dv1_kms=dv1 / KMS, dv2_kms=dv2 / KMS, flight_yr=tof, lead_deg=lead, v1=v1, v2=v2)


def build(case: Case) -> tuple[Stages, Orbit, Orbit]:
    h = hohmann(case)
    target = Orbit(case.target_au, math.radians(case.arrive_phase_deg))
    # (the departure planet on the arrival date: where it must be a Hohmann flight time after leaving 180 deg behind the
    # target's arrival point, then moved on by lead_deg)
    origin_at_arrival = target.phase - math.pi + Orbit(case.origin_au, 0.0).rate * h['flight_yr'] + math.radians(case.lead_deg)
    origin = Orbit(case.origin_au, origin_at_arrival)
    craft = Craft(case.thrust_n / case.mass_kg * ACCEL, case.thrust_n / (case.isp_s * G0 * case.mass_kg) * YEAR_S, case.time_weight)
    lo, hi = case.window
    tau0 = min(max(h['flight_yr'], lo), hi)
    o0 = np.asarray(origin.state(jnp.asarray(-tau0)))
    start = np.concatenate([o0, [1.0, -tau0, 0.0]])
    u0 = np.array([0.0, math.atan2(o0[3], o0[2])])
    leg = Leg((Rendezvous(tuple(float(x) for x in target.state(jnp.zeros(())))),), min_s=lo, max_s=hi, segments=case.segments)
    dep = departure(origin)(case.vinf_kms * KMS, -hi, -lo)
    free = ('x', 'y', 'vx', 'vy', 't')
    return Stages(craft, start, u0, (leg,), segments=case.segments, start_free=free, start_events=(dep,)), origin, target


def seed(prob: Stages, origin: Orbit, target: Orbit) -> tuple[np.ndarray, np.ndarray]:
    """(X, U): a prograde spiral from the inner planet's position at the seed's departure to the target's at arrival,
    radius and angle interpolated, the velocity the spiral's own, no thrust and the thrust angle along the velocity."""
    tau = float(prob.start[5]) * -1.0
    th0 = math.atan2(prob.start[1], prob.start[0])
    th1 = target.phase
    dth = (th1 - th0) % (2 * math.pi)
    r0, r1 = origin.radius, target.radius

    def state(w: float) -> np.ndarray:
        r, th = r0 + w * (r1 - r0), th0 + w * dth
        dr, dthdt = (r1 - r0) / tau, dth / tau
        vx = dr * math.cos(th) - r * dthdt * math.sin(th)
        vy = dr * math.sin(th) + r * dthdt * math.cos(th)
        return np.array([r * math.cos(th), r * math.sin(th), vx, vy, 1.0, -tau * (1.0 - w), 0.0]) / prob.ref

    c = np.concatenate([[0.0], np.asarray(radau_iia3()[0])])
    N, ns = prob.N, prob.ns
    X, U = np.zeros((N + 1, prob.nx)), np.zeros((N, prob.nu))

    def controls(y: np.ndarray, prev: float) -> np.ndarray:
        a = math.atan2(y[3], y[2])
        return np.array([0.0, a - 2 * math.pi * round((a - prev) / (2 * math.pi))])

    k_prev = np.asarray(prob.start_u, float)
    for k in range(N):
        Y = [state((k + ck) / N) for ck in c]
        k_next = controls(Y[-1] * prob.ref, k_prev[1])
        X[k] = np.concatenate([Y[0], k_prev, [tau]])
        U[k] = np.concatenate([k_next, [0.0], *Y[1:]])
        k_prev = k_next
    X[N] = np.concatenate([state(1.0), k_prev, [tau]])
    del ns
    return X, U


def with_weight(prob: Stages, w: float) -> tuple[np.ndarray, np.ndarray]:
    """The problem's per-stage data with the cost's propellant weight set to ``w`` (one compiled solver for all)."""
    masks, ends = prob.masks.copy(), prob.ends.copy()
    masks[:, prob._im + 1] = w
    ends[:, prob._dcol] = w
    return masks, ends


def solve(case: Case = Case(), verbose: bool = False) -> dict:  # noqa: B008 - a frozen dataclass default
    """The least-propellant rendezvous for ``case``: the homotopy's solves in turn, each seeded by the last."""
    prob, origin, target = build(case)
    X, U = seed(prob, origin, target)
    steps, t0 = [], time.time()
    res = None
    for w in case.homotopy:
        res = ipm.solve(prob, X, U, prob.x0(float(X[0, -1])), max_iter=300, verbose=verbose, data=with_weight(prob, w))
        steps.append((w, res.status, int(res.iterations), round(time.time() - t0, 1)))
        X, U = res.X, res.U
    s = res.X[:, : prob.ns] * prob.ref
    t_dep = float(s[0, 5])
    o, tg = np.asarray(origin.state(jnp.asarray(t_dep))), np.asarray(target.state(jnp.asarray(t_dep)))
    lead = (math.degrees(math.atan2(tg[1], tg[0]) - math.atan2(o[1], o[0])) + 180.0) % 360.0 - 180.0
    m_end = float(s[-1, 4])
    # (the first burn's centre, its throttle-weighted mean time: a finite burn straddles the impulse it stands for, so
    # its centre, not its start, is what compares with the impulsive transfer's departure; with no launch speed the
    # departure itself may precede the burn by a coast beside the planet, which costs nothing)
    thr = res.X[:, prob.ns]
    on = thr > 0.05
    k0 = int(np.argmax(on)) if on.any() else 0
    k1 = k0 + int(np.argmax(~on[k0:])) if (~on[k0:]).any() else len(on)
    t_burn = float(np.sum(thr[k0:k1] * s[k0:k1, 5]) / max(np.sum(thr[k0:k1]), 1e-12))
    ob, tb = np.asarray(origin.state(jnp.asarray(t_burn))), np.asarray(target.state(jnp.asarray(t_burn)))
    lead_burn = (math.degrees(math.atan2(tb[1], tb[0]) - math.atan2(ob[1], ob[0])) + 180.0) % 360.0 - 180.0
    out = dict(status=res.status, converged=bool(res.converged), steps=steps, seconds=round(time.time() - t0, 1),
               flight_yr=-t_dep, lead_deg=lead, burn_centre_to_arrival_yr=-t_burn, lead_at_burn_centre_deg=lead_burn,
               propellant_kg=(1 - m_end) * case.mass_kg,
               dv_kms=case.isp_s * G0 * math.log(1.0 / m_end) / 1000.0,
               vinf_kms=float(np.linalg.norm(s[0, 2:4] - o[2:]) / KMS))  # fmt: skip
    return {**out, 'X': res.X, 'U': res.U, 'prob': prob, 'origin': origin, 'target': target}


# The Hohmann check: no launch speed, a strong engine and a window round the Hohmann flight time.
# - With no launch speed any earlier departure is the same transfer after a coast beside the planet: a flat direction
#   the solve slides along to the window's edge (a tie-break on flight time stopped it converging), so the window is
#   narrow and the check compares the first burn's centre, not the departure.
# - 5 N: burns of about 11 days, which the 3-day stages resolve (at 50 N a day's burn was not, and the full-throttle
#   flow burned an iterate's mass away).
# - The specific impulse stays realistic: at 1e6 s the whole propellant cost was 0.0014, too weak against the throttle
#   bounds' barrier, which held the throttle at 0.18 throughout (a min-energy shape, 13.8 km/s) and still met the
#   tolerance, 2026-11-23.
HOHMANN_CASE = replace(Case(), thrust_n=5.0, vinf_kms=0.0, window=(0.65, 0.78), segments=100, homotopy=(0.5, 1.0))


def main(argv: list[str]) -> dict:
    case = HOHMANN_CASE if '--hohmann' in argv else Case()
    out = solve(case, verbose='--verbose' in argv)
    h = hohmann(case)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k not in ('X', 'U', 'prob', 'origin', 'target')})
    print('Hohmann:', {k: round(h[k], 4) for k in ('dv_kms', 'flight_yr', 'lead_deg')})
    return out


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    main(sys.argv[1:])
