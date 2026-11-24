"""The generic stage problem (doc/SPEC-vehicle-interface.md sections 3-5): any declared vehicle model on a list of legs,
each ending at an event, transcribed as Radau IIA stages for whoop.ipm (the device solver) to solve unchanged.

Stages k = 0..N-1, N = legs x segments, one 3-point Radau IIA collocation segment a stage:
- state x = (the model's states over their scales, the control knot entering the stage, the leg's duration tau);
- control u = (the knot leaving the stage, d_tau (moves tau on a leg's last stage only), the segment's three point
  states Z1..Z3 scaled); the controls are linear across a stage, their rates the knots' slope;
- stage equalities: the collocation defects; inequalities at each stage start: the model's path constraints, the control box
  and rate limits, the leg time, and the leg-end event's rows (masked where no leg ends);
- the objective the final cost state (whoop.ipm reads X[N, ci] ref[ci] / 10).

Events (the user's decision, 2026-11-22: objects writing their own data columns and rows): each event kind present
in the course gets an (active, columns...) block in the per-stage ``ends`` data, so one compiled step serves every
course with the same kinds. The start state is data too: whoop.ipm's ``x0`` (every component fixed but tau).

Built from whoop.ocp.StageProblem's Radau form, without its whoop specifics (sticks, packs, posts, chained courses).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np

from trajopt_jax._jax import Array, jax, jnp
from trajopt_jax.events import T_LEG_MIN, Leg
from trajopt_jax.radau import radau_iia3
from trajopt_jax.specs import ControlSpec, PathConstraint, StateSpec


class Model(Protocol):
    """What the transcription reads from a vehicle model (wing.model's declarations)."""

    states: tuple[StateSpec, ...]
    outputs: tuple[str, ...]
    data_width: int

    @property
    def controls(self) -> tuple[ControlSpec, ...]: ...
    @property
    def constraints(self) -> tuple[PathConstraint, ...]: ...
    def params(self) -> Array: ...
    def rhs(self, s: Array, u: Array, u_rate: Array, p: Array, data: Array) -> tuple[Array, Array]: ...
    def pose(self, s: Array) -> tuple[Array, Array]: ...


@dataclass
class Stages:
    """The stage problem for ``model`` on ``legs`` from the start state ``start`` (unscaled) and control ``start_u``."""

    model: Any
    start: np.ndarray
    start_u: np.ndarray
    legs: tuple[Leg, ...]
    segments: int = 20
    # co-designing: the model's design variables as constant states, free at the start, held within the model's
    # ``design_bounds()`` at stage 0 (as whoop.ocp.StageProblem does)
    codesign: bool = False
    # with ``codesign``: False pins the design variables at the start's (``x0(design=...)``), so the solve is a fixed
    # design's and -lam0 on their components is d(obj*)/d(design) (envelope theorem): the gradient a nested co-design's
    # outer loop takes. (All-at-once co-design of the wing's recon chain stalled at "acceptable" at two different
    # designs, both beaten by a fixed design 2.3 % lower, 2026-11-23; the fixed solves themselves converge cleanly)
    design_free: bool = True
    # chained missions: {first stage of a later mission: (its start state, unscaled; its start controls)}
    junctions: dict = field(default_factory=dict)
    # a start pinned by constraints: these states (by name) left free at stage 0, which ``start_events`` (events judged
    # on the start state) hold; ``start`` is then only their first guess (the Earth-Mars example's departure: a free
    # launch date, the position Earth's on it, the speed within the launch vehicle's)
    start_free: tuple[str, ...] = ()
    start_events: tuple = ()
    info: dict = field(default_factory=dict)
    node_second_order: ClassVar[bool] = False

    def __post_init__(self) -> None:
        m = self.model
        self.ns, self.nc = len(m.states), len(m.controls)
        self.ref = np.array([s.ref for s in m.states])
        self.si = {s.name: i for i, s in enumerate(m.states)}
        self.ci = self.si['cost']
        p0 = np.asarray(m.params(), float)
        self.nl = len(p0) if self.codesign else 0
        self.design_var_ref = np.where(np.abs(p0) > 1.0, np.abs(p0), 1.0)  # (design variables over their start, at least 1)
        self._il = slice(self.ns + self.nc, self.ns + self.nc + self.nl)
        self.nx, self.nu = self.ns + self.nc + self.nl + 1, self.nc + 1 + 3 * self.ns
        free_states = [self.si[n] for n in self.start_free]
        if len(free_states) != len(set(free_states)) or self.si['cost'] in free_states:
            raise ValueError('start_free: each state once, never the cost')
        # (the design variables when co-designing, the leg time, and the start states left free)
        moving = np.arange(self.ns + self.nc, self.nx) if self.design_free else np.array([self.nx - 1])
        self.free = np.concatenate([np.asarray(free_states, int), moving])
        # (each leg's stage count: its own, or the problem's; a short or fast-changing leg can be refined alone)
        # (each leg's stages: its mesh (fractions of its time), or its own count, or the problem's, uniform)
        meshes = [
            np.asarray(leg.mesh, float) if leg.mesh else np.full(leg.segments or self.segments, 1.0 / (leg.segments or self.segments))
            for leg in self.legs
        ]
        for p, mesh in enumerate(meshes):
            if abs(mesh.sum() - 1.0) > 1e-9 or np.any(mesh <= 0):
                raise ValueError(f'leg {p}: mesh fractions must be positive and sum to 1')
        self.frac = np.concatenate(meshes)  # (each stage's share of its leg's time)
        self.leg_S = [len(mesh) for mesh in meshes]
        self.leg_start = [0, *np.cumsum(self.leg_S).tolist()]
        self.S, self.N = self.segments, int(self.leg_start[-1])
        self.stage_leg = np.repeat(np.arange(len(self.legs)), self.leg_S)
        self.nh = 3 * self.ns
        self._c, self._D = (jnp.asarray(a) for a in radau_iia3())
        self._p = jnp.asarray(m.params())
        self.kinds = list(dict.fromkeys([*(type(ev) for leg in self.legs for ev in (*leg.end, *leg.along)), *map(type, self.start_events)]))
        self._make_data()
        dyn, defects, ineq = self._dynamics(), self._defects(), self._ineq()
        self._compile(dyn, defects, ineq)
        self.info.update(stages=self.N, nx=self.nx, nu=self.nu, ng=self.ng, nh=self.nh)

    # ---------------------------------------------------------------------------------------------------------- data
    def _make_data(self) -> None:
        """masks (N, 2 + data): [d_tau mask, the leg's cost weight, the model's data]; ends (N + 1, ...): [terminal flag, the stage's leg time
        bounds (2), each event kind's (active, columns), the model's data]. Stage k's state is leg p's end when
        k = (p + 1) S; a leg's ``along`` events hold at its stages pS + 1 .. (p + 1) S (its start is the last leg's end)."""
        nd, N, L0 = self.model.data_width, self.N, self.leg_start
        self._kcol: dict[type, int] = {}
        # ends: 0 terminal, 1-2 leg time bounds, 3 the first stage (the design variables' bounds), then the junction
        # block (flag, reset state, reset controls: a chained mission's start), then the event kinds, then model data
        # (then each stage's inverse stage count, 1 / S of its leg: the stage's length is the leg time over it)
        self._ie = 4 + 1 + self.ns + self.nc
        col = self._ie + 1
        for kind in self.kinds:
            self._kcol[kind] = col
            col += 1 + kind.width
        self._dcol = col
        self._im = 3 + self.ns + self.nc  # (masks: the stage's fraction of its leg's time, then the model's data)
        masks = np.zeros((N, self._im + 1 + nd))
        last = {L0[p + 1] - 1 for p in range(len(self.legs))}
        masks[:, 0] = [1.0 if k in last else 0.0 for k in range(N)]
        masks[:, 1] = [self.legs[self.stage_leg[k]].cost_weight for k in range(N)]
        masks[:, self._im] = self.frac
        ends = np.zeros((N + 1, col + nd))
        ends[N, 0] = 1.0
        ends[0, 3] = 1.0
        for k, (s0, u0) in self.junctions.items():  # (chained missions: the stage that starts each later one)
            block = np.concatenate([[1.0], np.asarray(s0) / self.ref, u0])
            masks[k, 2 : 3 + self.ns + self.nc] = block
            ends[k, 4 : 5 + self.ns + self.nc] = block
        for p, leg in enumerate(self.legs):
            # (the leg time's bounds at its first stage only, loose elsewhere: the time is one state held through the
            # leg, and a hold's min_s = max_s repeated at every stage made hundreds of dependent active rows, which
            # stalled the wing's recon at feasibility 0.1 on every mesh, 2026-11-23)
            ends[L0[p] : L0[p + 1] + 1, 1:3] = (T_LEG_MIN, 1e6)
            ends[L0[p], 1:3] = (leg.min_s, leg.max_s)
            ends[L0[p] : L0[p + 1], self._ie] = self.frac[L0[p] : L0[p + 1]]
            for ev in leg.end:
                self._write(ends, L0[p + 1], ev)
            for ev in leg.along:
                for k in range(L0[p] + 1, L0[p + 1] + 1):
                    self._write(ends, k, ev)
        for ev in self.start_events:  # (judged at stage 0, on the start state)
            self._write(ends, 0, ev)
        ends[N, self._ie] = self.frac[-1]
        self.masks, self.ends = masks, ends

    def _write(self, ends: np.ndarray, k: int, ev: Any) -> None:
        c = self._kcol[type(ev)]
        if ends[k, c] > 0:
            raise ValueError(f'stage {k}: two {type(ev).__name__} events (one block a kind a stage)')
        ends[k, c] = 1.0
        ends[k, c + 1 : c + 1 + ev.width] = ev.columns()

    def _stage_start(self, x: Array, row: Array, j: int) -> tuple[Array, Array]:
        """The stage's starting (scaled states, controls): the state's, or at a chained mission's junction (``row[j]``
        set) that mission's start from the data, the cost carried across."""
        ns, nc, ci = self.ns, self.nc, self.ci
        jr = row[j] > 0
        s0 = jnp.where(jr, row[j + 1 : j + 1 + ns].at[ci].set(x[ci]), x[:ns])
        return s0, jnp.where(jr, row[j + 1 + ns : j + 1 + ns + nc], x[ns : ns + nc])

    def x0(self, tau: float, design: np.ndarray | None = None) -> np.ndarray:
        """The start (scaled states, start controls, [scaled design variables: ``design``, unscaled, or the model's],
        the first leg's duration ``tau``)."""
        p = np.asarray(self.model.params() if design is None else design, float)
        lev = p[: self.nl] / self.design_var_ref[: self.nl]
        return np.concatenate([np.asarray(self.start) / self.ref, self.start_u, lev, [tau]])

    def design_gradient(self, lam0: np.ndarray) -> np.ndarray:
        """d(obj*)/d(design), unscaled, from a solve with the design pinned (``design_free=False``): -lam0 on the design
        components (the start rows' multipliers, envelope theorem) over their scale."""
        if not self.nl or self.design_free:
            raise ValueError('design_gradient needs codesign=True and design_free=False')
        return -np.asarray(lam0)[self._il] / self.design_var_ref

    def design(self, x: np.ndarray) -> np.ndarray:
        """The design variables a stage state carries (unscaled), or the model's when not co-designing."""
        return np.asarray(x)[self._il] * self.design_var_ref if self.nl else np.asarray(self.model.params())

    # -------------------------------------------------------------------------------------------------- the stages
    def _node(self, s: Array, u: Array, rate: Array, data: Array, x: Array | None = None) -> tuple[Array, Array]:
        """The model at unscaled ``s``; its design variables from the stage state ``x`` when co-designing."""
        p = x[self._il] * jnp.asarray(self.design_var_ref) if (self.nl and x is not None) else self._p
        return self.model.rhs(s, u, rate, p, data)

    def _dynamics(self) -> Callable[..., Any]:
        ns, nc, il = self.ns, self.nc, self._il

        def dyn(x: Array, u: Array, m: Array) -> Array:
            """The next stage's state: the segment's last point, the knot leaving, the leg time (moved at a leg's end)."""
            return jnp.concatenate([u[nc + 1 + 2 * ns :], u[:nc], x[il], jnp.atleast_1d(x[-1] + m[0] * u[nc])])

        return dyn

    def _defects(self) -> Callable[..., Any]:
        ns, nc, im = self.ns, self.nc, self._im
        ref, c, D = jnp.asarray(self.ref), self._c, self._D
        ci = self.ci

        def defects(x: Array, u: Array, m: Array) -> Array:
            """D Y - h f(Y) / ref at the three Radau points, Y = (the stage's start, Z1..Z3), scaled."""
            s0, k0 = self._stage_start(x, m, 2)
            k1, h = u[:nc], x[-1] * m[im]
            rate = (k1 - k0) / h
            Z = u[nc + 1 :].reshape(3, ns)
            Y = jnp.concatenate([s0[None], Z])
            ctrl = (1.0 - c[:, None]) * k0[None] + c[:, None] * k1[None]
            fz = jax.vmap(lambda z, k: self._node(z * ref, k, rate, m[im + 1 :], x)[0])(Z, ctrl)
            fz = fz.at[:, ci].multiply(m[1])  # (the cost's rate times the leg's weight)
            return (D @ Y - h * fz / ref).ravel()

        return defects

    def _ineq(self) -> Callable[..., Any]:
        ns, nc, ie = self.ns, self.nc, self._ie
        ref, model = jnp.asarray(self.ref), self.model
        out_i = {n: i for i, n in enumerate(model.outputs)}
        lo = jnp.array([ct.lo for ct in model.controls])
        hi = jnp.array([ct.hi for ct in model.controls])
        rmax = jnp.array([ct.rate_max if ct.rate_max is not None else 1e6 for ct in model.controls])
        kcol, dcol = self._kcol, self._dcol
        constraints = model.constraints  # (read once, before tracing: a model's constraints are constants of the problem)
        nl = self.nl
        lev_lo = lev_hi = jnp.zeros(0)
        if nl:
            bounds = np.asarray(model.design_bounds(), float)
            lev_lo, lev_hi = jnp.asarray(bounds[:, 0] / self.design_var_ref), jnp.asarray(bounds[:, 1] / self.design_var_ref)

        def ineq(x: Array, u: Array, e: Array) -> Array:
            """g <= 0 at a stage start (the terminal stage's control rows inactive: it has no control leaving)."""
            s0, k0 = self._stage_start(x, e, 4)
            s, tau = s0 * ref, x[-1]
            k1, h, term = u[:nc], tau * e[ie], e[0] > 0
            _, out = self._node(s, k0, (k1 - k0) / h, e[dcol:], x)
            g = []
            for lim in constraints:
                v = out[out_i[lim.output]]
                g += ([lim.lo - v] if lim.lo is not None else []) + ([v - lim.hi] if lim.hi is not None else [])
            box = [*(lo - k1), *(k1 - hi), *(k1 - k0 - rmax * h), *(k0 - k1 - rmax * h)]
            g += [jnp.where(term, -1.0, r) for r in box] + [e[1] - tau, tau - e[2]]
            s_end = x[:ns] * ref  # (the state arriving: at a junction, the last mission's end, which its end events judge)
            pos, vel = model.pose(s_end)
            for kind, col in kcol.items():
                on = e[col] > 0
                g += [jnp.where(on, r, -1.0) for r in kind.rows(s_end, pos, vel, e[col + 1 : col + 1 + kind.width])]
            if nl and self.design_free:  # (pinned, the outer loop holds them. The design variables' bounds, at the first stage only: repeated, active rows would be dependent)
                lev = x[self._il]
                g += [jnp.where(e[3] > 0, r, -1.0) for r in [*(lev_lo - lev), *(lev - lev_hi)]]
                # (the model's design constraints on the design variables alone, e.g. a wing-loading cap: at stage 0)
                for val, lo_c, hi_c in model.design_constraints(lev * jnp.asarray(self.design_var_ref)):
                    g += [
                        jnp.where(e[3] > 0, r, -1.0) for r in ([lo_c - val] if lo_c is not None else []) + ([val - hi_c] if hi_c is not None else [])
                    ]
            return jnp.stack(g)

        return ineq

    # ------------------------------------------------------------------------------------------------- compiled
    def _compile(self, dyn: Callable[..., Any], defects: Callable[..., Any], ineq: Callable[..., Any]) -> None:
        """The raw forms whoop.ipm compiles (dyn, ineq, defects and the exact stage Hessians) and the vmapped,
        jitted evaluations with Jacobians for ``evaluate``."""
        nx = self.nx

        def lagrangian(z: Array, m: Array, e: Array, lam: Array, nu: Array) -> Array:
            return -lam @ dyn(z[:nx], z[nx:], m) + nu @ ineq(z[:nx], z[nx:], e)

        def hessian(x: Array, u: Array, m: Array, e: Array, lam: Array, nu: Array) -> Array:
            return jax.jacfwd(jax.jacfwd(lagrangian))(jnp.concatenate([x, u]), m, e, lam, nu)

        def hessian_eq(x: Array, u: Array, m: Array, e: Array, lam: Array, nu: Array, nuh: Array) -> Array:
            def f(z: Array) -> Array:
                return lagrangian(z, m, e, lam, nu) + nuh @ defects(z[:nx], z[nx:], m)

            return jax.jacfwd(jax.jacfwd(f))(jnp.concatenate([x, u]))

        self.raw: dict[str, Callable[..., Any]] = dict(dyn=dyn, ineq=ineq, defects=defects, hessian=hessian, hessian_eq=hessian_eq)

        def with_jac(fn: Callable[..., Any]) -> Callable[..., Any]:
            return jax.jit(jax.vmap(lambda a, b, c: (fn(a, b, c), jax.jacfwd(fn, argnums=(0, 1))(a, b, c))))

        self._f, self._g, self._h = (jax.jit(jax.vmap(fn)) for fn in (dyn, ineq, defects))
        self._fj, self._gj, self._hj = (with_jac(fn) for fn in (dyn, ineq, defects))
        self.ng = int(ineq(jnp.ones(nx), jnp.ones(self.nu), jnp.asarray(self.ends[0])).shape[0])

    def evaluate(self, X: np.ndarray, U: np.ndarray, jac: bool, data: tuple | None = None) -> dict:
        """Dynamics defects c, inequalities g (stages 0..N, the last with zero controls), stage equalities h and, with
        ``jac``, their Jacobians, as whoop.ocp.StageProblem.evaluate."""
        N = self.N
        masks, ends = (self.masks, self.ends) if data is None else data
        Uz = np.vstack([U, np.zeros((1, self.nu))])
        args_f, args_g = (jnp.asarray(X[:-1]), jnp.asarray(U), jnp.asarray(masks)), (jnp.asarray(X), jnp.asarray(Uz), jnp.asarray(ends))
        out: dict[str, Any] = {}
        if jac:
            F, (A, B) = self._fj(*args_f)
            G, (Gx, Gu) = self._gj(*args_g)
            H, (Hx, Hu) = self._hj(*args_f)
            out |= dict(A=np.asarray(A), B=np.asarray(B), Gx=np.asarray(Gx), Gu=np.asarray(Gu)[:N], Hx=np.asarray(Hx), Hu=np.asarray(Hu))
        else:
            F, G, H = self._f(*args_f), self._g(*args_g), self._h(*args_f)
        out |= dict(c=X[1:] - np.asarray(F), g=np.asarray(G), h=np.asarray(H))
        out['obj'] = float(X[-1, self.ci] * self.ref[self.ci] / 10.0)
        return out

    # ------------------------------------------------------------------------------------------------- the seed
    def path_seed(self, speed: float) -> tuple[np.ndarray, np.ndarray]:
        """(X, U) along each leg's seed path (``Leg.seed_path``: straight at its end event, or circling its hold) at
        constant ``speed``, the controls held at the start's: the model's ``state_from`` maps each point's position and
        velocity to a state, the model's ``angles`` unwrapped so turns do not jump by 2 pi."""
        ns, nc, ref, L0 = self.ns, self.nc, self.ref, self.leg_start
        a, paths, mission_start = np.asarray(self.start[:3], float), [], np.asarray(self.start, float)
        for i, leg in enumerate(self.legs):
            if L0[i] in self.junctions:  # (a chained mission: its seed starts at its own start)
                mission_start = np.asarray(self.junctions[L0[i]][0], float)
                a = mission_start[:3].copy()
            b, tau, at = leg.seed_path(a, speed, mission_start)
            paths.append((tau, at))
            a = b
        taus = [tau for tau, _ in paths]
        c = np.concatenate([[0.0], np.asarray(self._c)])
        X, U = np.zeros((self.N + 1, self.nx)), np.zeros((self.N, self.nu))
        integ = [self.si[n] for n in getattr(self.model, 'seed_integrated', ('cost',))]
        ang = [self.si[n] for n in getattr(self.model, 'angles', ())]
        rate = jax.jit(lambda s: self._node(s, jnp.asarray(self.start_u), jnp.zeros(nc), jnp.zeros(self.model.data_width))[0])
        t0, prev, t_prev = 0.0, None, 0.0
        lev0 = self.x0(0.0)[self._il]

        def state(p: int, w: float) -> np.ndarray:
            """The seed's state at fraction ``w`` of leg ``p``: the path's kinematics, the integrated states advanced
            from the previous node by its rates (explicit Euler through the nodes, in time order)."""
            nonlocal prev, t_prev
            if w == 0.0 and L0[p] in self.junctions and prev is not None:  # (the junction: the next mission's start)
                cost = prev[self.ci]
                prev = np.asarray(self.junctions[L0[p]][0], float).copy()
                prev[self.ci] = cost
            pos, vel = paths[p][1](w)
            t = t0 + w * taus[p]
            st = self.model.state_from(pos, vel, np.array(self.start, float) if prev is None else prev)
            if prev is not None:
                st[ang] -= 2 * math.pi * np.round((st[ang] - prev[ang]) / (2 * math.pi))  # (only the angles: positions jump)
                # (sub-stepped at 0.2 s at most: the pack's polarisation relaxes in ~2 s, and nodes of a long leg are tens
                # of seconds apart, where one explicit step diverged, 2026-11-22)
                n_sub = max(1, int(math.ceil((t - t_prev) / 0.2)))
                cur = prev.copy()
                for _ in range(n_sub):
                    cur[integ] = cur[integ] + (t - t_prev) / n_sub * np.asarray(rate(jnp.asarray(cur)))[integ]
                st[integ] = cur[integ]
            prev, t_prev = st, t
            return st / ref

        for k in range(self.N):
            p = int(self.stage_leg[k])
            j, S = k - L0[p], self.leg_S[p]
            w0, fk = float(self.frac[L0[p] : k].sum()), float(self.frac[k])
            Y = [state(p, w0 + ck * fk) for ck in c]
            last = j == S - 1
            nxt = taus[min(p + 1, len(taus) - 1)]
            # (at a junction the stage's state is the last mission's end, its start reset from the data)
            y0 = U[k - 1][nc + 1 + 2 * ns :] if k in self.junctions else Y[0]
            X[k] = np.concatenate([y0, self.start_u, lev0, [taus[p]]])
            U[k] = np.concatenate([self.start_u, [nxt - taus[p] if last else 0.0], *Y[1:]])
            if last:
                t0 += taus[p]
        X[self.N] = np.concatenate([U[-1][nc + 1 + 2 * ns :], self.start_u, lev0, [taus[-1]]])
        X[0, :ns] = np.asarray(self.start) / ref
        return X, U


@dataclass(frozen=True)
class Mission:
    """One mission of a chained problem: its start state (unscaled) and controls, its legs, its weight in the
    objective (0: a constraint mission, flown only to be feasible)."""

    start: np.ndarray
    start_u: np.ndarray
    legs: tuple[Leg, ...]
    weight: float = 1.0


def chained(model: Any, missions: list[Mission], segments: int = 20, codesign: bool = False, design_free: bool = True) -> Stages:
    """Every mission in one stage problem (one design across them when co-designing): the legs in sequence, each later
    mission's first stage a junction that resets the state to its start (the cost carried), each leg's cost weight
    times its mission's (whoop.ocp.StageProblem.chained's form)."""
    import dataclasses

    legs, junctions, n_stage = [], {}, 0
    for i, mis in enumerate(missions):
        if i:
            junctions[n_stage] = (np.asarray(mis.start, float), np.asarray(mis.start_u, float))
        legs += [dataclasses.replace(leg, cost_weight=leg.cost_weight * mis.weight) for leg in mis.legs]
        n_stage += sum(len(leg.mesh) if leg.mesh else (leg.segments or segments) for leg in mis.legs)
    first = missions[0]
    return Stages(
        model, first.start, first.start_u, tuple(legs), segments=segments, codesign=codesign, junctions=junctions, design_free=design_free
    )
