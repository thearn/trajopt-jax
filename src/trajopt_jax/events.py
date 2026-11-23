"""Mission events and legs (doc/SPEC-vehicle-interface.md section 3, events as objects; misfit 6: rows over a whole
leg, not only at its end).

An event kind writes its parameters as data ``columns()`` and gives its ``rows(s, pos, vel, e)`` (each <= 0) from the
model's state ``s``, position and velocity at a stage and its columns ``e``. A leg ends at its ``end`` events (rows at
its last stage) and keeps its ``along`` events at every stage after its first (the hold's orbit); its duration lies in
[``min_s``, ``max_s``] (a hold: both its length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from trajopt_jax._jax import Array, jnp

BAND = 1e-4  # m: an equality on a state (a gate's plane) as a band: the Riccati form takes inequalities at states
T_LEG_MIN = 0.2  # s


@dataclass(frozen=True)
class Gate:
    """A square window flown through: centre, heading (rad, from +x toward +y: the direction of flight, the window's
    plane normal to it), half its usable side (the opening's half less the vehicle's half span) and a through-speed."""

    x: float
    y: float
    z: float
    heading: float
    half: float
    v_min: float = 0.5
    width: ClassVar[int] = 7

    def columns(self) -> list[float]:
        c, s = math.cos(self.heading), math.sin(self.heading)
        return [c, s, c * self.x + s * self.y, -s * self.x + c * self.y, self.z, self.half, self.v_min]

    @staticmethod
    def rows(s: Array, pos: Array, vel: Array, e: Array) -> list:
        c, sn, plane, lat, z, half, v_min = (e[i] for i in range(7))
        along, side = c * pos[0] + sn * pos[1], -sn * pos[0] + c * pos[1]
        rows = [along - plane - BAND, plane - along - BAND, side - lat - half, lat - side - half]
        return rows + [pos[2] - z - half, z - pos[2] - half, v_min - (c * vel[0] + sn * vel[1])]

    def aim(self, start: np.ndarray) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass(frozen=True)
class Point:
    """A point passed within ``tol`` on each axis (a None coordinate free)."""

    x: float | None
    y: float | None
    z: float | None
    tol: float = 0.5
    width: ClassVar[int] = 7

    def columns(self) -> list[float]:
        coords = [self.x, self.y, self.z]
        return [0.0 if c is None else c for c in coords] + [0.0 if c is None else 1.0 for c in coords] + [self.tol]

    @staticmethod
    def rows(s: Array, pos: Array, vel: Array, e: Array) -> list:
        rows = []
        for i in range(3):
            on = e[3 + i] > 0
            rows += [jnp.where(on, pos[i] - e[i] - e[6], -1.0), jnp.where(on, e[i] - pos[i] - e[6], -1.0)]
        return rows

    def aim(self, start: np.ndarray) -> np.ndarray:
        return np.array([float(start[i]) if v is None else v for i, v in enumerate((self.x, self.y, self.z))])


@dataclass(frozen=True)
class Orbit:
    """Held within ``radius`` (m, horizontally) of a centre and within ``dz`` of its height: the recon hold."""

    x: float
    y: float
    z: float
    radius: float
    dz: float = 5.0
    width: ClassVar[int] = 5

    def columns(self) -> list[float]:
        return [self.x, self.y, self.z, self.radius, self.dz]

    @staticmethod
    def rows(s: Array, pos: Array, vel: Array, e: Array) -> list:
        d = jnp.sqrt((pos[0] - e[0]) ** 2 + (pos[1] - e[1]) ** 2 + 1e-9)
        return [d - e[3], pos[2] - e[2] - e[4], e[2] - pos[2] - e[4]]

    def aim(self, start: np.ndarray) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass(frozen=True)
class StateCap:
    """A model state held at or below ``upper`` (the pack's reserve: the charge drawn at the mission's end)."""

    index: int
    upper: float
    width: ClassVar[int] = 2

    def columns(self) -> list[float]:
        return [float(self.index), self.upper]

    @staticmethod
    def rows(s: Array, pos: Array, vel: Array, e: Array) -> list:
        return [s[e[0].astype(int)] - e[1]]


@dataclass(frozen=True)
class Box:
    """Inside an axis-aligned box (x, y, z up), each bound optional (None: open): the takeoff's 'above 5 m before
    x = D', the landing's touchdown zone."""

    lo: tuple[float | None, float | None, float | None]
    hi: tuple[float | None, float | None, float | None]
    width: ClassVar[int] = 12

    def columns(self) -> list[float]:
        vals = [0.0 if v is None else v for v in (*self.lo, *self.hi)]
        return vals + [0.0 if v is None else 1.0 for v in (*self.lo, *self.hi)]

    @staticmethod
    def rows(s: Array, pos: Array, vel: Array, e: Array) -> list:
        rows = []
        for i in range(3):
            rows.append(jnp.where(e[6 + i] > 0, e[i] - pos[i], -1.0))
            rows.append(jnp.where(e[9 + i] > 0, pos[i] - e[3 + i], -1.0))
        return rows

    def aim(self, start: np.ndarray) -> np.ndarray:
        mid = [
            start[i] if lo is None and hi is None else (hi if lo is None else lo if hi is None else 0.5 * (lo + hi))
            for i, (lo, hi) in enumerate(zip(self.lo, self.hi, strict=True))
        ]
        return np.array(mid, float)


@dataclass(frozen=True)
class Touchdown:
    """At the leg's end: speed over the ground at most ``v_max`` and sinking at most ``sink_max`` (m/s): a belly
    landing the airframe survives."""

    v_max: float
    sink_max: float
    width: ClassVar[int] = 2

    def columns(self) -> list[float]:
        return [self.v_max, self.sink_max]

    @staticmethod
    def rows(s: Array, pos: Array, vel: Array, e: Array) -> list:
        return [jnp.sqrt(vel[0] ** 2 + vel[1] ** 2 + 1e-9) - e[0], -vel[2] - e[1]]


Event = Gate | Point | Orbit | StateCap | Box | Touchdown


@dataclass(frozen=True)
class Leg:
    end: tuple[Event, ...] = ()
    along: tuple[Event, ...] = ()
    min_s: float = T_LEG_MIN
    max_s: float = 30.0
    segments: int | None = None  # this leg's stage count (None: the problem's); refine a short or fast-changing leg
    mesh: tuple[float, ...] | None = None  # this leg's stage fractions of its time (sum 1): a refined mesh, over segments
    cost_weight: float = 1.0  # the cost's rate on this leg times this (0: the leg costs nothing, e.g. the return in range)

    def seed_path(self, a: np.ndarray, speed: float, start: np.ndarray) -> tuple[np.ndarray, float, object]:
        """(where the leg's seed ends, its duration, position and velocity at fraction w): a hold circles its orbit at
        0.8 of its radius from the point nearest ``a``; any other leg flies straight at the aim of its first placed end
        event."""
        orbit = next((e for e in self.along if isinstance(e, Orbit)), None)
        if orbit is not None:
            c, r = np.array([orbit.x, orbit.y, orbit.z]), 0.8 * orbit.radius
            th0 = math.atan2(a[1] - c[1], a[0] - c[0])
            d0 = float(np.hypot(a[0] - c[0], a[1] - c[1]))
            tau = max(self.min_s, T_LEG_MIN)
            om = speed / r

            def at(w: float) -> tuple[np.ndarray, np.ndarray]:
                # (the radius ramps from where the leg starts to 0.8 R over its first quarter: no jump at the start)
                th, rad = th0 + om * w * tau, d0 + (r - d0) * min(1.0, 4.0 * w)
                pos = c + rad * np.array([math.cos(th), math.sin(th), 0.0])
                pos[2] = a[2] + (c[2] - a[2]) * min(1.0, 4.0 * w)
                return pos, speed * np.array([-math.sin(th), math.cos(th), 0.0])

            return at(1.0)[0], tau, at
        placed = [e for e in self.end if not isinstance(e, StateCap)]
        b = placed[0].aim(start) if placed else a
        dist = float(np.linalg.norm(b - a))
        tau = float(np.clip(dist / speed, max(self.min_s, T_LEG_MIN), self.max_s))
        direction = (b - a) / max(dist, 1e-9) if dist > 1e-9 else np.array([1.0, 0.0, 0.0])

        def line(w: float) -> tuple[np.ndarray, np.ndarray]:
            return a + w * (b - a), speed * direction

        return b, tau, line
