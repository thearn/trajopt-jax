"""The sensitivity of a solved stage problem's optimum to its model data, d(obj*)/d(data), from that solve alone
(envelope theorem): the Lagrangian's partial derivative in the data at the solution and its multipliers. Model data
enters only through the collocation defects (multipliers ``nuh``, the data in ``masks``) and the path inequalities
(``v``, the data in ``ends``); the dynamics rows copy variables and the objective is the cost state.

A design enters as data (the wing's aerodynamic coefficients, rebuilt from the planform outside the solve): one compiled
problem serves every design, and the design gradient is this sensitivity chained through the data's own map, a
vector-Jacobian product. Design variables held as constant states instead put that map inside every node's second-order
pass: 5.5 GB a 128-stage block on the wing's 784-stage recon chain, 2026-11-24.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trajopt_jax._jax import jax, jnp
from trajopt_jax.stages import Stages


def with_data(
    prob: Stages, row: np.ndarray, base: tuple[np.ndarray, np.ndarray] | None = None, columns: slice | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(masks, ends) with every stage's model data in ``columns`` (all of it by default) set to ``row``, the rest as in
    ``base`` (the problem's own by default): ``ipm.solve(..., data=...)``'s argument."""
    cols = columns or slice(0, prob.model.data_width)
    masks, ends = (prob.masks.copy(), prob.ends.copy()) if base is None else (base[0].copy(), base[1].copy())
    im, dc = prob._im + 1, prob._dcol
    masks[:, im + cols.start : im + cols.stop] = row
    ends[:, dc + cols.start : dc + cols.stop] = row
    return masks, ends


def data_sensitivity(prob: Stages, X: np.ndarray, U: np.ndarray, state: dict, chunk: int = 128, columns: slice | None = None) -> np.ndarray:
    """d(obj*)/d(data) for the model data in ``columns`` (all of it by default), the same at every stage (``with_data``),
    at the solution ``X, U`` with the solver state ``state`` (IPMResult.state: the multipliers ``v``, ``nuh`` and the stage
    data ``masks``, ``ends``); data outside ``columns`` (a stage's own: the whoop's wind on one course) is held. The
    stages in blocks of ``chunk``."""
    cols = columns or slice(0, prob.model.data_width)
    nd, N = cols.stop - cols.start, prob.N
    if not nd:
        raise ValueError('data_sensitivity needs data columns (a model with data_width > 0)')
    ineq, defects = prob.raw['ineq'], prob.raw['defects']
    im, dc = prob._im + 1 + cols.start, prob._dcol + cols.start
    Uz = np.concatenate([U, np.zeros((1, U.shape[1]))])
    v, nuh = np.asarray(state['v']), np.asarray(state['nuh'])
    masks, ends = np.asarray(state['masks']), np.asarray(state['ends'])
    row = jnp.asarray(ends[0, dc : dc + nd])

    def g_block(d: Any, xs: Any, us: Any, es: Any, vs: Any) -> Any:
        return jnp.sum(vs * jax.vmap(ineq)(xs, us, es.at[:, dc : dc + nd].set(d)))

    def h_block(d: Any, xs: Any, us: Any, ms: Any, ns: Any) -> Any:
        return jnp.sum(ns * jax.vmap(defects)(xs, us, ms.at[:, im : im + nd].set(d)))

    grad_g, grad_h = jax.jit(jax.grad(g_block)), jax.jit(jax.grad(h_block))
    total = jnp.zeros(nd)
    for a in range(0, N + 1, chunk):  # (the inequalities on every stage, the terminal one's with no control leaving)
        b = min(N + 1, a + chunk)
        total = total + grad_g(row, jnp.asarray(X[a:b]), jnp.asarray(Uz[a:b]), jnp.asarray(ends[a:b]), jnp.asarray(v[a:b]))
    if nuh.size:  # (the defects on the N stages only: the terminal stage's padded points give NaN, and 0 x NaN is NaN)
        for a in range(0, N, chunk):
            b = min(N, a + chunk)
            total = total + grad_h(row, jnp.asarray(X[a:b]), jnp.asarray(U[a:b]), jnp.asarray(masks[a:b]), jnp.asarray(nuh[a:b]))
    return np.asarray(total)
