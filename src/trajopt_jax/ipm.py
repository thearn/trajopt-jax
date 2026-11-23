"""The exact-Hessian interior-point SQP on whoop.ocp.StageProblem as one jitted iteration on the device, a GPU or the
CPU (doc/PLAN-interop.md, "a structured solver"). RK4 stages, or Radau (its collocation defects as stage equalities, eliminated per
stage before the Riccati: whoop.riccati.eliminate_equalities).

Each iteration: the stage evaluations and their Jacobians, the exact stage Hessians (forward over forward), the KKT
solve by whoop.riccati (parallel in time on a GPU, lax.scan on a CPU), the step for IPOPT's monotone barrier, then
the trial points. Where the first numpy solver searched serially, this one evaluates on the device: a ladder of inertia shifts
(the smallest with every Cholesky pivot positive and positive curvature in the free start) and a ladder of step
lengths (the first that passes the filter-like test), each one batched evaluation. Levenberg-Marquardt damping on the
accepted step's fraction and a second-order correction. One host synchronisation an iteration (the convergence test).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from trajopt_jax._jax import Array, jax, jnp
from trajopt_jax.problem import StageProblem
from trajopt_jax.result import IPMResult
from trajopt_jax.riccati import LQR, eliminate_equalities, guarded, restore, riccati_scan

SHIFTS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)  # the inertia ladder (added to the damping)
BACKTRACK = 12  # step lengths a_p 2^-j
FILTER = 32  # the filter's ring of past (theta, phi) pairs
HESSIAN_BUDGET = 550_000  # stages x (stage variables)^2 the stage passes may take whole (the 16 GB card)
PROX_MIN = 1e-3  # the damping's restart value after a short step (times the up factor). 1e-8 everywhere or in the final
# barrier problem only (2026-11-18): lap co-design converged (130 / 80 iterations) but the hard start and 2-gate
# co-design failed; the end game needs a better method (the active-set finish), not this knob
PROX_MAX = 1e3  # the damping's ceiling (uncapped, a hard start drove it past 1e4 and the iterate to NaN)
DEAD = 3  # consecutive iterations with the iterate unmoved before the multipliers are reset (recovery)
# a crawl: CRAWL failed line searches running with the damping at PROX_MAX (each a 1e-3 fallback step). The two-course
# bilevel's cold 2-gate solve from another design's trajectory (2026-11-21) sat so from iteration ~60 to its 600 cap,
# infeasibility 1.1e-2 -> 6.5e-3, mu at 1.8e-6; the dead rule never fired (alpha 1e-3, not 0). Recovery as a dead
# iterate's, the damping back to 1, and mu raised to CRAWL_MU (None: kept)
CRAWL = 10
CRAWL_MU: float | None = None
_STEPS: dict = {}  # (id(prob), options) -> (prob, the jitted step): a bilevel outer loop re-solves one problem
S_MAX = 100.0  # IPOPT's s_max: the multiplier size above which the optimality error is scaled down
# IPOPT's acceptable termination (acceptable_tol / acceptable_iter, as the reference IPOPT runs set them): stationarity
# and complementarity below ACCEPT_TOL for ACCEPT_ITER iterations running, feasibility at the full tolerance (stricter
# than IPOPT's 1e-2: the defects must close). Lap co-design held stationarity at 2-3e-6 for 300 iterations, feasible to
# 1e-14 (flat design variable directions: an undamped step overshoots, a damped one creeps; an active-set finish did not help)
ACCEPT_TOL = 1e-4
ACCEPT_ITER = 15
STALL = 30  # consecutive iterations with the iterate unmoved (no finite trial point): the solve stops, 'stalled'


def chunked(fn: Any, chunk: int | None) -> Any:
    """``fn`` vmapped over its arguments' leading (stage) axis, in blocks of ``chunk`` stages by lax.map when given: each
    block one vectorised call, peak memory bounded by the block (the forward-over-forward stage Hessians carry 25 x 25
    tangents through the propulsion's root solves: 900 stages at once needed more than 16 GB)."""
    vf = jax.vmap(fn)
    if not chunk:
        return vf

    def run(*args: Any) -> Any:
        n = args[0].shape[0]
        pad = (-n) % chunk
        padded = [jnp.concatenate([a, jnp.repeat(a[-1:], pad, 0)]) if pad else a for a in args]
        blocks = [a.reshape(-1, chunk, *a.shape[1:]) for a in padded]
        out = jax.lax.map(lambda b: vf(*b), tuple(blocks))
        return jax.tree.map(lambda o: o.reshape(-1, *o.shape[2:])[:n], out)

    return run


def make_step(
    prob: StageProblem,
    parallel: bool,
    tol: float,
    chunk: int | None = None,
    damping: tuple = (4.0, 4.0),
    line_search: str = 'loop',
    hessian: str = 'exact',
    model_dtype: Any = None,
    inertia: str = 'adaptive',
    jit: bool = True,
) -> Any:
    """The jitted iteration ``step(state) -> (state, info)`` for ``prob`` (the start ``x0`` in the state: one step serves every start); ``chunk``: the
    stage passes (Jacobians, Hessians) in blocks of that many stages (None: all at once). ``line_search``: 'loop' (the trial points in turn on the
    device, stopping at the first accepted: memory for one) or 'batched' (all in one vmapped evaluation, as mpx and DiffMPC do). ``hessian``: 'exact'
    (forward over forward) or 'gauss_newton' (no curvature of the dynamics or constraints: with a linear objective only the barrier's, plus the
    inertia shift). ``model_dtype``: the model's evaluations, Jacobians and Hessians in this precision (float32), the KKT in float64."""
    step = _Step(prob, _StepOptions(parallel, tol, chunk, damping, line_search, hessian, model_dtype, inertia))
    return step.step if not jit else jax.jit(step.step)


class _StepOptions(NamedTuple):
    """make_step's options (see it): the parallel Riccati, the tolerance, the stage passes' chunk, the damping's
    factors, the line search's form, the Hessian's form, the model's precision and the inertia rule."""

    parallel: bool
    tol: float
    chunk: int | None
    damping: tuple
    line_search: str
    hessian: str
    model_dtype: Any
    inertia: str


class _Step:
    """One iteration of the device solver, its phases as methods: the KKT residuals and the barrier (``_kkt``), the
    stage Hessians (``_hessians``), the inertia-corrected direction (``_direction``), the second-order correction
    (``_correction``), the filter line search (``_search``) and the state's update (``_update``)."""

    def __init__(self, prob: StageProblem, opts: _StepOptions) -> None:
        parallel, tol, chunk, damping, line_search, hessian, model_dtype, inertia = opts
        self.N, self.nx, self.nu, self.ng = prob.N, prob.nx, prob.nu, prob.ng
        # the start: every component fixed but these (the first leg's duration; the design variables when co-designing)
        self.free = np.asarray(prob.free)
        self.fixed = np.setdiff1d(np.arange(self.nx), self.free)
        self.up, self.down = damping  # the damping's factors after a short step (fraction < 1/4) and a full one (> 3/4)
        self.tol, self.line_search, self.hessian, self.inertia = tol, line_search, hessian, inertia
        raw = dict(prob.raw)
        if model_dtype is not None:  # the model in model_dtype, everything around it in float64
            raw = {k: _cast(f, model_dtype) for k, f in raw.items()}
        self.dyn, self.ineq = (jax.vmap(raw[k]) for k in ('dyn', 'ineq'))
        self.nh = nh = prob.nh  # the Radau form's stage equalities (its collocation defects); 0: RK4
        self.hess = chunked(raw['hessian'], chunk)
        # (the Radau form's defects and their Jacobian; RK4 has none, its stand-ins never called)
        self.hess_eq = chunked(raw['hessian_eq'], chunk) if nh else self.hess
        self.defects = jax.vmap(raw['defects']) if nh else self.dyn
        self.jac_h = chunked(jax.jacfwd(raw['defects'], argnums=(0, 1)), chunk) if nh else self.hess
        self.jac_f = chunked(jax.jacfwd(raw['dyn'], argnums=(0, 1)), chunk)
        self.jac_g = chunked(jax.jacfwd(raw['ineq'], argnums=(0, 1)), chunk)
        self.ci, self.cscale = prob.ci, float(prob.ref[prob.ci] / 10.0)
        # the parallel scan refined and checked, the sequential recursion when its residual is too large (riccati.guarded)
        self.solve_lqr = guarded() if parallel else riccati_scan
        self.eye_v, self.eye_x = jnp.eye(self.nx + self.nu), jnp.eye(self.nx)

    def evaluate(self, X: Array, U: Array, st: Array) -> tuple:
        """(F, G, c0, the defects, objective, the stage equalities): the start and the per-stage data (masks, ends: the
        legs' ends and the wind) from the state, not constants, so one compiled step serves any start and data."""
        N, nu = self.N, self.nu
        Uz = jnp.concatenate([U, jnp.zeros((1, nu))])
        F = self.dyn(X[:-1], U, st['masks'])
        G = self.ineq(X, Uz, st['ends'])
        c0 = (X[0] - st['x0'])[self.fixed]
        Hr = self.defects(X[:-1], U, st['masks']) if self.nh else jnp.zeros((N, 0))
        return F, G, c0, X[1:] - F, X[-1, self.ci] * self.cscale, Hr

    def merit(self, X: Array, U: Array, s: Array, mu: Array, st: Array) -> tuple:
        """(infeasibility theta, barrier objective) at a trial point."""
        _, G, c0, cd, obj, Hr = self.evaluate(X, U, st)
        theta = jnp.abs(c0).sum() + jnp.abs(cd).sum() + jnp.abs(G + s).sum() + jnp.abs(Hr).sum()
        return theta, obj - mu * jnp.log(s).sum()

    def step(self, st: dict) -> tuple:
        """One iteration: ``(state, info)`` from ``state``."""
        k = self._kkt(st)
        self._direction(st, k)
        self._correction(st, k)
        self._search(st, k)
        return self._update(st, k)

    def _kkt(self, st: dict) -> dict:
        """The residuals at the iterate, the scaled optimality error (IPOPT's, Waechter and Biegler 2006, eq. 5:
        stationarity over s_d and complementarity over s_c, each the multipliers' mean magnitude over S_MAX, at least 1;
        unscaled, lap co-design sat at 1.3e-6 against tol 1e-6 for 300 iterations, 2026-11-18), IPOPT's monotone barrier
        (a smaller mu once the barrier problem is solved to 10 mu), and the condensed KKT's stage blocks."""
        N, nx, nu, nh, ci = self.N, self.nx, self.nu, self.nh, self.ci
        X, U, s, v, lam, mu, nuh = st['X'], st['U'], st['s'], st['v'], st['lam'], st['mu'], st['nuh']
        Uz = jnp.concatenate([U, jnp.zeros((1, nu))])
        masks, ends = st['masks'], st['ends']
        F, G, c0, cd, obj, Hr = self.evaluate(X, U, st)
        A, B = self.jac_f(X[:-1], U, masks)
        Gx, Gu = self.jac_g(X, Uz, ends)
        Hx, Hu = self.jac_h(X[:-1], U, masks) if nh else (jnp.zeros((N, 0, nx)), jnp.zeros((N, 0, nu)))
        # the Lagrangian's gradient (lam[0]: the start's, its last component 0: the leg duration is free)
        gx = jnp.zeros((N + 1, nx)).at[-1, ci].add(self.cscale)
        gx = gx.at[:-1].add(-jnp.einsum('kij,ki->kj', A, lam[1:])).at[1:].add(lam[1:]).at[0].add(lam[0])
        gx = gx + jnp.einsum('kij,ki->kj', Gx, v)
        gu = -jnp.einsum('kij,ki->kj', B, lam[1:]) + jnp.einsum('kij,ki->kj', Gu[:N], v[:N])
        if nh:
            gx = gx.at[:N].add(jnp.einsum('kij,ki->kj', Hx, nuh))
            gu = gu + jnp.einsum('kij,ki->kj', Hu, nuh)
        stat = jnp.maximum(jnp.abs(gx).max(), jnp.abs(gu).max())
        feas = jnp.maximum(jnp.maximum(jnp.abs(c0).max(), jnp.abs(cd).max()), jnp.maximum(G, 0.0).max())
        feas = jnp.maximum(feas, jnp.abs(Hr).max()) if nh else feas
        comp = jnp.abs(v * s - mu).max()
        n_mult = lam.size + v.size + nuh.size
        s_d = jnp.maximum(S_MAX, (jnp.abs(lam).sum() + jnp.abs(v).sum() + jnp.abs(nuh).sum()) / n_mult) / S_MAX
        s_c = jnp.maximum(S_MAX, jnp.abs(v).sum() / v.size) / S_MAX
        stat, comp = stat / s_d, comp / s_c
        mu = jnp.where(jnp.maximum(jnp.maximum(stat, feas), comp) < 10.0 * mu, jnp.maximum(self.tol / 10, jnp.minimum(0.2 * mu, mu**1.5)), mu)
        r_g = G + s
        d = v / s
        Hs = self._hessians(st, X, Uz)
        Gs = jnp.concatenate([Gx, Gu], axis=2)  # (N + 1, ng, nv)
        K = Hs + jnp.einsum('kji,kj,kjl->kil', Gs, d, Gs)
        w = (-(v * s - mu) + v * r_g) / s
        rx = -gx - jnp.einsum('kji,kj->ki', Gx, w)
        ru = -gu - jnp.einsum('kji,kj->ki', Gu[:N], w[:N])
        return dict(
            X=X, U=U, s=s, v=v, mu=mu, c0=c0, cd=cd, obj=obj, Hr=Hr, A=A, B=B, Gx=Gx, Gu=Gu, Hx=Hx, Hu=Hu, gx=gx, gu=gu,
            stat=stat, feas=feas, comp=comp, s_d=s_d, r_g=r_g, Qs=K[:N], QN=K[N][:nx, :nx], rx=rx, ru=ru,
        )  # fmt: skip

    def _hessians(self, st: dict, X: Array, Uz: Array) -> Any:
        """The stage Lagrangian Hessians (N + 1, nv, nv): exact, or none (Gauss-Newton). The defects' term on the N
        stages only: the terminal stage's padded zero points (a zero quaternion) give NaN, and 0 x NaN is NaN."""
        N, nx = self.N, self.nx
        masks, lam, v = st['masks'], st['lam'], st['v']
        hargs = (X, Uz, jnp.concatenate([masks, jnp.zeros((1, masks.shape[1]))]), st['ends'], jnp.concatenate([lam[1:], jnp.zeros((1, nx))]), v)
        if self.hessian == 'gauss_newton':
            return jnp.zeros((N + 1, nx + self.nu, nx + self.nu))
        if self.nh:
            return jnp.concatenate([self.hess_eq(*(a[:N] for a in hargs), st['nuh']), self.hess(*(a[N:] for a in hargs))])
        return self.hess(*hargs)

    def _solve_at(self, k: dict, delta: Any, rhs: tuple) -> tuple:
        """The KKT step at inertia shift ``delta`` for right-hand side ``rhs`` = (q_st, qN, e, x0d, hres): (dX, dU, L,
        lam0, dnu) and whether every part is finite. The linearised defects Hx dx + Hu du + h = 0 removed stage by stage,
        their multipliers restored."""
        q_st, qN, e, x0d, hres = rhs
        Qd = k['Qs'] + delta * self.eye_v
        QNd = k['QN'] + delta * self.eye_x
        if self.nh:
            Qw, qw, Aw, Bw, ew, maps = eliminate_equalities(Qd, q_st, k['A'], k['B'], e, k['Hx'], k['Hu'], hres)
            dX_, W, L_, lam0_ = self.solve_lqr(LQR(Qw, QNd, qw, qN, Aw, Bw, ew), x0d, self.free)
            dU_, dnu_ = restore(dX_, W, L_, maps, Qd, q_st, k['B'], k['Hu'])
            out = (dX_, dU_, L_, lam0_, dnu_)
        else:
            out = (*self.solve_lqr(LQR(Qd, QNd, q_st, qN, k['A'], k['B'], e), x0d, self.free), jnp.zeros((self.N, 0)))
        ok = jnp.all(jnp.array([jnp.all(jnp.isfinite(o)) for o in out]))
        return out, ok

    def _direction(self, st: dict, k: dict) -> None:
        """The step by the inertia ladder on the device, smallest shift first, stopping at the first that solves (one
        live at a time: batched, 900 stages needed more than the card's 16 GB). 'adaptive': IPOPT's rule, no shift first,
        then a third of the last that worked (1e-4 with none), grown 8x (100x with none before); else a fixed ladder."""
        N, nx, prox = self.N, self.nx, st['prox']
        rhs = (-jnp.concatenate([k['rx'][:N], k['ru']], axis=1), -k['rx'][N], -k['cd'], jnp.zeros(nx).at[self.fixed].set(-k['c0']), k['Hr'])
        if self.inertia == 'adaptive':
            dlast = st['dlast']
            out0, ok0 = self._solve_at(k, prox, rhs)

            def body(c: tuple) -> tuple:
                shift = jnp.where(
                    c[0] == 0.0, jnp.where(dlast > 0.0, jnp.maximum(1e-20, dlast / 3.0), 1e-4), c[0] * jnp.where(dlast > 0.0, 8.0, 100.0)
                )
                out, ok = self._solve_at(k, prox + shift, rhs)
                return shift, out, ok

            shift_ok, out, _ = jax.lax.while_loop(lambda c: (~c[2]) & (c[0] < 1e10), body, (jnp.asarray(0.0), out0, ok0))
            delta_used = prox + shift_ok
        else:
            deltas = prox + jnp.asarray(SHIFTS)
            out0, ok0 = self._solve_at(k, deltas[0], rhs)

            def body_fixed(c: tuple) -> tuple:
                i = c[0] + 1
                out, ok = self._solve_at(k, deltas[i], rhs)
                return i, out, ok

            i_ok, out, _ = jax.lax.while_loop(lambda c: (~c[2]) & (c[0] < len(SHIFTS) - 1), body_fixed, (jnp.asarray(0), out0, ok0))
            delta_used = deltas[i_ok]
            shift_ok = delta_used - prox
        dX, dU, L, lam0, dnu = out
        s, v, mu = k['s'], k['v'], k['mu']
        ds = -k['r_g'] - jnp.einsum('kij,kj->ki', k['Gx'], dX) - jnp.concatenate([jnp.einsum('kij,kj->ki', k['Gu'][:N], dU), jnp.zeros((1, self.ng))])
        k.update(dX=dX, dU=dU, L=L, lam0=lam0, dnu=dnu, ds=ds, dv=(-(v * s - mu) - v * ds) / s, delta_used=delta_used, shift_ok=shift_ok)
        k['a_p'], k['a_d'] = _ftb(s, ds), _ftb(v, k['dv'])

    def _correction(self, st: dict, k: dict) -> None:
        """The full step's trial point (one evaluation site: each call to the model is a full copy of it in the compiled
        step) and IPOPT's second-order correction from its residuals, on the same factors' shift."""
        N, nx, mu = self.N, self.nx, k['mu']
        X, U, s, v, a_p = k['X'], k['U'], k['s'], k['v'], k['a_p']
        _, G1, c01, cd1, obj1, H1 = self.evaluate(X + a_p * k['dX'], U + a_p * k['dU'], st)
        s1 = s + a_p * k['ds']
        k['th_first'] = jnp.abs(c01).sum() + jnp.abs(cd1).sum() + jnp.abs(G1 + s1).sum() + jnp.abs(H1).sum()
        k['bar_first'] = obj1 - mu * jnp.log(s1).sum()
        r_soc = a_p * k['r_g'] + (G1 + s1)
        w_c = (-(v * s - mu) + v * r_soc) / s
        rx_c = -k['gx'] - jnp.einsum('kji,kj->ki', k['Gx'], w_c)
        ru_c = -k['gu'] - jnp.einsum('kji,kj->ki', k['Gu'][:N], w_c[:N])
        rhs = (
            -jnp.concatenate([rx_c[:N], ru_c], axis=1),
            -rx_c[N],
            -(a_p * k['cd'] + cd1),
            jnp.zeros(nx).at[self.fixed].set(-(a_p * k['c0'] + c01)),
            a_p * k['Hr'] + H1,
        )
        (dXc, dUc, Lc, lam0c, dnuc), _ = self._solve_at(k, k['delta_used'], rhs)
        dsc = -r_soc - jnp.einsum('kij,kj->ki', k['Gx'], dXc) - jnp.concatenate([jnp.einsum('kij,kj->ki', k['Gu'][:N], dUc), jnp.zeros((1, self.ng))])
        k.update(dXc=dXc, dUc=dUc, Lc=Lc, lam0c=lam0c, dnuc=dnuc, dsc=dsc, dvc=(-(v * s - mu) - v * dsc) / s, a_c=_ftb(s, dsc))

    def _search(self, st: dict, k: dict) -> None:
        """IPOPT's filter line search (Waechter and Biegler 2006), compact: a ring of FILTER augmented (theta, phi)
        pairs (``_acceptance``); the full step first, then one device loop (or one batched evaluation) over the
        correction point (j = 0) and the full step's backtracks a_p 2^-j (j >= 1). Replaced fixed and best-so-far floors
        on theta: a fixed floor of 1 let co-design cycle with a Radau defect at 5e-3 for 700 iterations; a floor at the
        best theta crept (steps 2^-7) on every course."""
        mu, X, U, s, a_p, a_c = k['mu'], k['X'], k['U'], k['s'], k['a_p'], k['a_c']
        passes = self._acceptance(st, k)
        first_ok = passes(k['th_first'], k['bar_first'], a_p)
        first_finite = jnp.isfinite(k['th_first']) & jnp.isfinite(k['bar_first'])
        pairs = ((k['dXc'], k['dX']), (k['dUc'], k['dU']), (k['dsc'], k['ds']))

        def trial(a: Any, sj: Any) -> tuple:
            Dx, Du, Ds = (jnp.where(sj, cc, oo) for cc, oo in pairs)
            return self.merit(X + a * Dx, U + a * Du, s + a * Ds, mu, st)

        if self.line_search == 'batched':  # every trial at once (memory: BACKTRACK evaluations of the model)
            js = jnp.arange(BACKTRACK)
            soc_j = js == 0
            a_j = jnp.where(soc_j, a_c, a_p * 0.5**js)
            ths, bas = jax.vmap(trial)(a_j, soc_j)
            oks = jax.vmap(passes)(ths, bas, jnp.where(soc_j, a_p, a_j))
            j_ok = jnp.where(first_ok, -1, jnp.where(oks.any(), jnp.argmax(oks), BACKTRACK - 1))
            any_ok = first_ok | oks.any()
            last_finite = jnp.where(first_ok, first_finite, jnp.isfinite(ths[-1]) & jnp.isfinite(bas[-1]))
        else:

            def ls_body(c: tuple) -> tuple:
                j = c[0] + 1
                soc = j == 0
                a = jnp.where(soc, a_c, a_p * 0.5**j)
                th, ba = trial(a, soc)
                return j, passes(th, ba, jnp.where(soc, a_p, a)), jnp.isfinite(th) & jnp.isfinite(ba)

            ls_cond = lambda c: (~c[1]) & (c[0] < BACKTRACK - 1)  # noqa: E731 - the loop's condition, a one-line test
            j_ok, any_ok, last_finite = jax.lax.while_loop(ls_cond, ls_body, (jnp.asarray(-1), first_ok, first_finite))
        use_c = (~first_ok) & any_ok & (j_ok == 0)
        # none accepted: a tiny step along the full step if the last trial point was finite, else none (a NaN iterate
        # cannot recover)
        alpha = jnp.where(
            first_ok, a_p, jnp.where(use_c, a_c, jnp.where(any_ok, a_p * 0.5**j_ok, jnp.where(last_finite, jnp.minimum(a_p, 1e-3), 0.0)))
        )
        k.update(alpha=alpha, use_c=use_c, any_ok=any_ok)

    def _acceptance(self, st: dict, k: dict) -> Callable[..., Any]:
        """passes(theta, phi, step) for the filter line search, with its records (the filter, its margins, the f-type
        test) set in ``k``. A trial must not be dominated by any filter pair nor by the current point (a fraction of
        theta less, or phi less by a fraction of theta), and theta below theta_max; near feasibility (theta below
        theta_min) along a descent direction, Armijo on phi instead (an f-type step: nothing added)."""
        mu, s, tol = k['mu'], k['s'], self.tol
        theta0 = jnp.abs(k['c0']).sum() + jnp.abs(k['cd']).sum() + jnp.abs(k['r_g']).sum() + jnp.abs(k['Hr']).sum()
        bar0 = k['obj'] - mu * jnp.log(s).sum()
        t_init = jnp.where(jnp.isfinite(st['tinit']), st['tinit'], theta0)
        in_final = mu <= 1.0001 * tol / 10
        tfin = jnp.where(in_final, jnp.minimum(st['tfin'], theta0), jnp.inf)  # the best infeasibility of the final phase
        theta_min, theta_max = 1e-4 * jnp.maximum(1.0, t_init), 1e4 * jnp.maximum(1.0, t_init)
        filt = jnp.where(mu < st['mu'], jnp.inf, st['filt'])  # (a new barrier problem: a new filter)
        gphi = self.cscale * k['dX'][-1, self.ci] - mu * jnp.sum(k['ds'] / s)  # the barrier objective's slope along the step
        f_type = (theta0 <= theta_min) & (gphi < 0.0)
        k.update(theta0=theta0, bar0=bar0, t_init=t_init, tfin=tfin, filt=filt, f_type=f_type)

        def passes(th: Any, ba: Any, a: Any) -> Any:
            gam = 1e-5
            not_dominated = jnp.all((th <= filt[:, 0]) | (ba <= filt[:, 1]))
            filter_ok = not_dominated & ((th <= (1.0 - gam) * theta0) | (ba <= bar0 - gam * theta0))
            armijo = ba <= bar0 + 1e-4 * a * gphi
            # an f-type (Armijo) step must also stay near-feasible: with the fine inertia shifts a near-singular step passed
            # Armijo on phi alone and threw 2-gate co-design from 2.6e-4 back to 0.1 infeasible (theta_max is 1e4 x the
            # start's); beyond theta_min it is judged by the filter
            f_ok = armijo & (th <= theta_min)
            # the final barrier problem (mu at its floor): the infeasibility may rise to 10x the best of this phase, no more.
            # Unbounded, the filter's trade of a little infeasibility for objective let 2-gate co-design creep from 8e-6
            # to 1e-3 infeasible, below the optimum; forbidding any rise failed the 2-gate course and tripled the laps
            # (floored at the tolerance: the 2-gate course reached 1e-14 and a cap of 1e-13 rejected every step, holding
            # stationarity at 1.3e-6 for 450 iterations)
            final_ok = ~in_final | (th <= jnp.maximum(theta0, jnp.maximum(10.0 * tfin, tol)))
            base = jnp.isfinite(ba) & jnp.isfinite(th) & (th <= theta_max) & final_ok
            return base & jnp.where(f_type, f_ok | filter_ok, filter_ok)

        return passes

    def _update(self, st: dict, k: dict) -> tuple:
        """The new state: the accepted step (the correction's when it won), the damping by the accepted fraction, the
        multipliers by the dual step, the filter's new pair; the recoveries (``_recover``); and the iteration's info."""
        nx, alpha, use_c, any_ok = self.nx, k['alpha'], k['use_c'], k['any_ok']
        X, U, s, v, mu = k['X'], k['U'], k['s'], k['v'], k['mu']
        lam, nuh, prox = st['lam'], st['nuh'], st['prox']
        sel = ('dX', 'dU', 'ds', 'dv', 'L', 'lam0', 'dnu')
        dX, dU, ds, dv, L, lam0, dnu = (jnp.where(use_c, k[n + 'c'], k[n]) for n in sel)
        a_p = jnp.where(use_c, k['a_c'], k['a_p'])
        a_d = jnp.where(use_c, _ftb(v, k['dvc']), k['a_d'])
        frac = alpha / jnp.maximum(a_p, 1e-12)
        prox = jnp.where(frac < 0.25, self.up * jnp.maximum(prox, PROX_MIN), jnp.where(frac > 0.75, prox / self.down, prox))
        prox = jnp.clip(jnp.where(prox > 1e-6, prox, 0.0), 0.0, PROX_MAX)
        dlam = jnp.concatenate([jnp.zeros(nx).at[self.fixed].set(lam0[self.fixed])[None], L])
        s_new = jnp.maximum(s + alpha * ds, 1e-12)
        a_y = jnp.where(any_ok, a_d, alpha)
        f_type, theta0, bar0, filt = k['f_type'], k['theta0'], k['bar0'], k['filt']
        new = dict(
            x0=st['x0'],
            masks=st['masks'],
            ends=st['ends'],
            X=jnp.where(alpha > 0.0, X + alpha * dX, X),  # (0 x a NaN direction is NaN: select, not scale)
            U=jnp.where(alpha > 0.0, U + alpha * dU, U),
            s=jnp.where(alpha > 0.0, s_new, s),
            # bound multipliers by their own fraction to the boundary, then kept within 1e10 of mu / s (IPOPT's kappa_sigma)
            v=jnp.where(alpha > 0.0, jnp.clip(v + a_d * dv, mu / (1e10 * s_new), 1e10 * mu / s_new), v),
            # equality multipliers by the dual step after an accepted step, by the primal step after a failed one:
            # by the dual step always, a failed search (primal 1e-3) still moved them fully, to a bad linearisation's
            # values, and a hard start (stock design variables on the co-design optimum's path) blew them to 1e10 in 20 iterations,
            # every trial then non-finite; by the primal step always (IPOPT's alpha_for_y = primal) it converged, but
            # 2-gate co-design took 840 iterations against 181 and the 2-gate course 362 against 116
            lam=jnp.where(alpha > 0.0, lam + a_y * dlam, lam),
            mu=mu,
            prox=prox,
            dlast=k['shift_ok'],
            tinit=k['t_init'],
            tfin=k['tfin'],
            # an accepted h-type step adds the current point, augmented by the margins, at the ring's next slot
            filt=jnp.where(
                any_ok & ~f_type, filt.at[st['fidx'].astype(int) % FILTER].set(jnp.stack([(1 - 1e-5) * theta0, bar0 - 1e-5 * theta0])), filt
            ),
            fidx=st['fidx'] + jnp.where(any_ok & ~f_type, 1.0, 0.0),
            nuh=jnp.where(alpha > 0.0, nuh + a_y * dnu, nuh),
        )
        reset = self._recover(st, k, new)
        info = dict(
            obj=k['obj'], stat=k['stat'], feas=k['feas'], comp=k['comp'], mu=new['mu'], alpha=alpha, a_p=a_p, delta=k['delta_used'],
            ok=any_ok, soc=use_c, reset=reset, s_d=k['s_d'],
        )  # fmt: skip
        # which of the step's parts was non-finite (a diagnostic: bit k set when output k of the KKT solve was not finite)
        info['nonfinite'] = sum((~jnp.all(jnp.isfinite(o))).astype(int) * 2**i for i, o in enumerate((dX, dU, L, lam0, dnu)))
        # one signature for every call: the state's leaves strong float64 (a weakly typed scalar out of jnp.where with a
        # Python float made the second call a second compile, 242 s cold)
        return {n: jax.lax.convert_element_type(val, jnp.float64) for n, val in new.items()}, info

    def _recover(self, st: dict, k: dict, new: dict) -> Any:
        """Recovery, in place on ``new``: from a dead iterate (no finite trial point DEAD times running), or a crawl
        (CRAWL failed searches at PROX_MAX), the multipliers reset (equalities zero, bounds mu / s), the filter cleared,
        the damping raised (a crawl's to 1). The hard start (stock design variables on the co-design optimum's path) blew the
        multipliers to 1e10 on accepted steps of 0.02-0.16 and then sat at alpha 0; changing the multiplier step itself
        (by the primal step always, or below a step of 0.25) recovered it but cost the regression problems 2-5x the
        iterations or convergence, so normal iterations are left as they were. Returns whether it reset."""
        alpha, any_ok, mu, s = k['alpha'], k['any_ok'], new['mu'], k['s']
        dead = jnp.where(alpha == 0.0, st['dead'] + 1.0, 0.0)
        crawl = jnp.where((~any_ok) & (new['prox'] >= PROX_MAX), st['crawl'] + 1.0, 0.0)
        crawled = crawl >= CRAWL
        if CRAWL_MU is not None:
            mu = jnp.where(crawled, jnp.maximum(mu, CRAWL_MU), mu)
            new['mu'] = mu
        reset = (dead >= DEAD) | crawled
        new['lam'] = jnp.where(reset, 0.0, new['lam'])
        new['nuh'] = jnp.where(reset, 0.0, new['nuh'])
        new['v'] = jnp.where(reset, mu / s, new['v'])
        new['filt'] = jnp.where(reset, jnp.inf, new['filt'])
        new['prox'] = jnp.where(crawled, 1.0, jnp.where(reset, jnp.maximum(new['prox'], 1.0), new['prox']))
        new['dead'] = jnp.where(reset, 0.0, dead)
        new['crawl'] = jnp.where(reset, 0.0, crawl)
        return reset


def _cast(fn: Callable[..., Any], model_dtype: Any) -> Callable[..., Any]:
    """``fn`` evaluated in ``model_dtype``, its outputs back in float64."""

    def run(*a: Any) -> Any:
        out = fn(*(x.astype(model_dtype) for x in a))
        return jax.tree.map(lambda o: o.astype(jnp.float64), out)

    return run


def _ftb(val: Array, dval: Array) -> Any:
    """The fraction to the boundary (0.995): the largest step in (0, 1] keeping ``val + a dval`` positive."""
    ratio = jnp.where(dval < 0, -0.995 * val / jnp.where(dval < 0, dval, -1.0), jnp.inf)
    return jnp.minimum(1.0, ratio.min())


def initial_state(prob: StageProblem, X: np.ndarray, U: np.ndarray, x0: np.ndarray, mu0: float = 0.1, data: tuple | None = None) -> dict:
    """The device iteration's state at (X, U) from start ``x0``: slacks from the inequalities (at least 1e-2), their
    multipliers mu0 / s, the rest zero or empty; every leaf strong float64 (one call signature). ``data``: the
    per-stage (masks, ends), default the problem's (another wind: ``StageProblem.with_wind``)."""
    X = np.array(X, float)
    X[0] = x0
    masks, ends = (prob.masks, prob.ends) if data is None else data
    ev = prob.evaluate(X, U, False, (masks, ends))
    s = np.maximum(-ev['g'], 1e-2)
    st = dict(
        X=jnp.asarray(X),
        U=jnp.asarray(U),
        s=jnp.asarray(s),
        v=jnp.asarray(mu0 / s),
        lam=jnp.zeros((prob.N + 1, prob.nx)),
        mu=jnp.asarray(mu0),
        prox=jnp.asarray(0.0),
        tinit=jnp.asarray(jnp.inf),
        tfin=jnp.asarray(jnp.inf),
        dlast=jnp.asarray(0.0),
        filt=jnp.full((FILTER, 2), jnp.inf),
        fidx=jnp.asarray(0.0),
        nuh=jnp.zeros((prob.N, prob.nh)),
        dead=jnp.asarray(0.0),
        crawl=jnp.asarray(0.0),
        x0=jnp.asarray(x0, dtype=jnp.float64),
        masks=jnp.asarray(masks),
        ends=jnp.asarray(ends[: prob.N + 1]),
    )
    return {k: jax.lax.convert_element_type(val, jnp.float64) for k, val in st.items()}


def warm_state(prob: StageProblem, prev: dict, X: np.ndarray, U: np.ndarray, x0: np.ndarray, mu0: float, data: tuple | None = None) -> dict:
    """A warm start from a converged neighbour's state ``prev`` (IPOPT's warm_start_init_point): the iterate (X, U)
    from ``x0``, the multipliers kept, each slack the larger of the inequality's margin and the barrier's ``mu0 / v``
    (on the central path), the filter and the shifts fresh."""
    st = initial_state(prob, X, U, x0, mu0, data)
    v = np.maximum(np.asarray(prev['v']), 1e-12)
    ev = prob.evaluate(np.asarray(st['X']), np.asarray(U), False, (np.asarray(st['masks']), np.asarray(st['ends'])))
    s = np.maximum(np.maximum(-ev['g'], mu0 / v), 1e-12)
    st.update(s=jnp.asarray(s), v=jnp.asarray(v), lam=prev['lam'], nuh=prev['nuh'])
    return {k: jax.lax.convert_element_type(val, jnp.float64) for k, val in st.items()}


def solve(
    prob: StageProblem,
    X: np.ndarray,
    U: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    mu0: float = 0.1,
    parallel: bool | None = None,
    verbose: bool = True,
    chunk: int | None = None,
    damping: tuple = (4.0, 4.0),
    line_search: str = 'loop',
    hessian: str = 'exact',
    model_dtype: Any = None,
    inertia: str = 'adaptive',
    warm: dict | None = None,
    data: tuple | None = None,
) -> IPMResult:
    """The device iteration from (X, U), the start ``x0``; ``parallel`` (default off; ``True`` opts in, guarded by its residual) picks the parallel-in-time
    Riccati. Converged when stationarity, feasibility and complementarity are all below ``tol``. ``warm``: a
    converged neighbour's ``res.state`` (see warm_state), with a small ``mu0``. ``data``: per-stage (masks, ends) in
    place of the problem's (``StageProblem.with_wind``): the same compiled step."""
    if parallel is None:
        # (the sequential recursion by default, on a GPU too: the parallel solve's steps were too inexact on the wing's
        # recon chains, and guarded it fell back to the scan nearly every step, 26 s an iteration against the scan's
        # 11; parallel=True opts in, guarded, where it is accurate, 2026-11-23)
        parallel = False
    t0 = time.time()
    st = initial_state(prob, X, U, x0, mu0, data) if warm is None else warm_state(prob, warm, X, U, x0, mu0, data)
    if chunk is None:  # the stage passes whole when they fit, else in blocks that do (chunking a pass that fits cost the
        # lap ~50 % an iteration): the forward-over-forward Hessians' memory goes as stages x (stage variables)^2; on the
        # 16 GB card 110 x 70^2 and 300 x 25^2 fitted, 110 x 80^2 and 900 x 25^2 did not
        per_stage = (prob.nx + prob.nu) ** 2
        if prob.node_second_order:  # the node's own derivatives: the Hessian pass measured 1.6x (3.75x co-designing) lighter
            per_stage = int(per_stage / (3.0 if prob.nl else 1.5))
        if (prob.N + 1) * per_stage > HESSIAN_BUDGET:
            chunk = max(8, HESSIAN_BUDGET // (2 * per_stage))  # (half: 85 stages x 80^2, at the budget, still ran out)
    key = (id(prob), parallel, tol, chunk, damping, line_search, hessian, model_dtype, inertia)
    if _STEPS.get(key, (None,))[0] is not prob:  # one compiled step a problem and options, reused across starts
        _STEPS[key] = (prob, make_step(prob, parallel, tol, chunk, damping, line_search, hessian, model_dtype, inertia))
    step = _STEPS[key][1]
    history = []
    converged = False
    status, stuck, near = 'max_iter', 0, 0
    it = 0
    for it in range(1, max_iter + 1):
        st, info = step(st)
        info = {k: float(v) for k, v in info.items()}
        history.append(dict(it=it, wall=time.time() - t0, **info))
        if verbose and (it % 10 == 0 or it < 4):
            print(
                f'{it:4d} obj {10 * info["obj"]:.5f} stat {info["stat"]:.1e} feas {info["feas"]:.1e} comp {info["comp"]:.1e} '
                f'mu {info["mu"]:.0e} a {info["alpha"]:.2e} delta {info["delta"]:.0e} nf {int(info["nonfinite"])} t {history[-1]["wall"]:.1f}',
                flush=True,
            )
        if info['stat'] < tol and info['feas'] < tol and info['comp'] < tol:
            converged, status = True, 'converged'
            break
        near = near + 1 if info['stat'] < ACCEPT_TOL and info['feas'] < tol and info['comp'] < ACCEPT_TOL else 0
        if near >= ACCEPT_ITER:
            converged, status = True, 'acceptable'
            break
        if not np.isfinite(info['obj']):
            status = 'nonfinite'
            break
        # the iterate unmoved (alpha 0: no finite trial point; the fixture spent 586 of 600 iterations so, 2026-11-17).
        # Not rejected steps: their 1e-3 fallback creeps on, and stopping after 30 of them failed 2-gate co-design at 401
        stuck = stuck + 1 if info['alpha'] == 0.0 else 0
        if stuck >= STALL:
            status = 'stalled'
            break
    X, U = np.asarray(st['X']), np.asarray(st['U'])
    res = IPMResult(X, U, prob.evaluate(X, U, False)['obj'], it, converged, time.time() - t0, history)
    # the start rows' multipliers: d(obj*)/d(x0) = -lam0 (envelope theorem), on the fixed start components
    res.lam0 = np.asarray(st['lam'][0])
    res.state = st
    res.status = status
    return res


__all__ = ['initial_state', 'make_step', 'solve']
