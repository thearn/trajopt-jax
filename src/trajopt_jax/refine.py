"""Mesh refinement for trajopt_jax.stages problems, in passes: estimate each stage's error between its collocation
points, split the stages above tolerance in half (h-refinement), carry the solution onto the new mesh as the next
pass's start, solve again.

The error: the stage's collocation polynomial (through its start and three Radau points) is differentiated at the
midpoints between its nodes and compared with the model's dynamics there, in scaled states, per unit of the stage's
normalised time; collocation forces the two to agree only at the Radau points, so the gap between them measures how
well the stage resolves the motion. Where the flight is steady it is small on long stages; at a manoeuvre (a roll-in,
a control switch) it is large until the stages are short (the wing's recon, 2026-11-22: a 0.6 s short period needed
~0.3 s stages there and only there).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np

from trajopt_jax import ipm
from trajopt_jax._jax import Array, jax, jnp
from trajopt_jax.events import Leg
from trajopt_jax.stages import Stages


def _nodes(prob: Stages) -> np.ndarray:
    return np.concatenate([[0.0], np.asarray(prob._c)])


def _lagrange(nodes: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """(value weights, derivative weights) of the cubic through ``nodes`` at ``tau``."""
    V = np.vander(nodes, 4, increasing=True)
    inv = np.linalg.inv(V)
    pw = np.array([1.0, tau, tau * tau, tau**3])
    dw = np.array([0.0, 1.0, 2 * tau, 3 * tau * tau])
    return pw @ inv, dw @ inv


def stage_errors(prob: Stages, X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Each stage's largest scaled gap between the polynomial's slope and the dynamics at the midpoints of its nodes."""
    ns, nc, im = prob.ns, prob.nc, prob._im
    ref = jnp.asarray(prob.ref)
    nodes = _nodes(prob)
    mids = 0.5 * (nodes[:-1] + nodes[1:])
    W = [_lagrange(nodes, float(m)) for m in mids]
    PW, DW = jnp.asarray(np.stack([w[0] for w in W])), jnp.asarray(np.stack([w[1] for w in W]))

    def one(x: Array, u: Array, m: Array) -> Array:
        s0, k0 = prob._stage_start(x, m, 2)
        k1, h = u[:nc], x[-1] * m[im]
        Y = jnp.concatenate([s0[None], u[nc + 1 :].reshape(3, ns)])
        rate = (k1 - k0) / h
        ys, dys = PW @ Y, DW @ Y
        ctrl = (1.0 - jnp.asarray(mids)[:, None]) * k0[None] + jnp.asarray(mids)[:, None] * k1[None]
        f = jax.vmap(lambda y, k: prob._node(y * ref, k, rate, m[im + 1 :], x)[0])(ys, ctrl)
        return jnp.max(jnp.abs(dys - h * f / ref))

    return np.asarray(jax.vmap(one)(jnp.asarray(X[:-1]), jnp.asarray(U), jnp.asarray(prob.masks)))


def refined_legs(prob: Stages, errors: np.ndarray, tol: float, min_frac: float = 1e-3) -> tuple[tuple[Leg, ...], int]:
    """The legs with each stage above ``tol`` split in half (never below ``min_frac`` of its leg); how many split."""
    legs, n_split = [], 0
    for p, leg in enumerate(prob.legs):
        a, b = prob.leg_start[p], prob.leg_start[p + 1]
        mesh = []
        for k in range(a, b):
            f = float(prob.frac[k])
            if errors[k] > tol and f / 2 >= min_frac:
                mesh += [f / 2, f / 2]
                n_split += 1
            else:
                mesh.append(f)
        legs.append(dataclasses.replace(leg, mesh=tuple(mesh), segments=None))
    return tuple(legs), n_split


def _evaluator(prob: Stages, X: np.ndarray, U: np.ndarray) -> Callable[[int, float], tuple[np.ndarray, np.ndarray]]:
    """(scaled state, control) of the solution at fraction ``w`` of leg ``p`` (the stage's cubic; controls linear)."""
    ns, nc = prob.ns, prob.nc
    nodes = _nodes(prob)

    def at(p: int, w: float) -> tuple[np.ndarray, np.ndarray]:
        a, b = prob.leg_start[p], prob.leg_start[p + 1]
        cum = np.concatenate([[0.0], np.cumsum(prob.frac[a:b])])
        j = int(np.clip(np.searchsorted(cum, w, side='right') - 1, 0, b - a - 1))
        k = a + j
        tau = (w - cum[j]) / prob.frac[k]
        Y = np.concatenate([X[k, :ns][None], U[k, nc + 1 :].reshape(3, ns)])
        pw, _ = _lagrange(nodes, float(np.clip(tau, 0.0, 1.0)))
        k0, k1 = X[k, ns : ns + nc], U[k, :nc]
        return pw @ Y, (1.0 - tau) * k0 + tau * k1

    return at


def transfer(old: Stages, X: np.ndarray, U: np.ndarray, new: Stages) -> tuple[np.ndarray, np.ndarray]:
    """The old solution on the new mesh (same legs, refined): states and Radau points from the old stages' cubics,
    controls linear, leg times and design variables kept."""
    ns, nc = new.ns, new.nc
    at = _evaluator(old, X, U)
    nodes = _nodes(new)
    Xn, Un = np.zeros((new.N + 1, new.nx)), np.zeros((new.N, new.nu))
    for k in range(new.N):
        p = int(new.stage_leg[k])
        a = new.leg_start[p]
        w0, fk = float(new.frac[a:k].sum()), float(new.frac[k])
        pts = [at(p, min(w0 + c * fk, 1.0)) for c in nodes]
        tau_p = X[old.leg_start[p], -1]
        last = k == new.leg_start[p + 1] - 1
        tau_next = X[old.leg_start[min(p + 1, len(old.legs) - 1)], -1]
        Xn[k] = np.concatenate([pts[0][0], pts[0][1], X[old.leg_start[p], ns + nc : -1], [tau_p]])
        Un[k] = np.concatenate([pts[-1][1], [tau_next - tau_p if last else 0.0], *[q[0] for q in pts[1:]]])
    Xn[new.N] = X[old.N].copy()
    Xn[new.N, :ns] = Un[-1][nc + 1 + 2 * ns :]
    for k in new.junctions:  # (a junction stage's state: the last mission's end, its start reset from the data)
        Xn[k, :ns] = Un[k - 1][nc + 1 + 2 * ns :]
    return Xn, Un


def solve_refined(
    build: Callable[[tuple[Leg, ...]], Stages],
    legs: tuple[Leg, ...],
    X: np.ndarray,
    U: np.ndarray,
    passes: int = 3,
    tol: float = 1e-3,
    **solve_kw: Any,
) -> tuple[Stages, Any, list[dict]]:
    """Solve, estimate, split, carry over, solve again, until no stage is above ``tol`` or ``passes`` are spent.
    ``build(legs)`` makes the problem for a mesh. Returns the last problem, its result and each pass's summary."""
    prob = build(legs)
    record = []
    res = None
    for i in range(passes + 1):
        res = ipm.solve(prob, X, U, prob.x0(X[0, -1]), **solve_kw)
        err = stage_errors(prob, res.X, res.U)
        record.append(dict(pass_=i, stages=prob.N, status=res.status, obj=float(res.obj), err_max=float(err.max()), above=int((err > tol).sum())))
        if i == passes or not (err > tol).any():
            break
        new_legs, _ = refined_legs(prob, err, tol)
        new = build(new_legs)
        X, U = transfer(prob, res.X, res.U, new)
        prob = new
    return prob, res, record
