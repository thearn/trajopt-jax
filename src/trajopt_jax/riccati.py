"""The interior-point step's KKT system as an LQR in JAX, sequential and parallel in time (doc/PLAN-interop.md, "a
structured solver": the GPU form of whoop.ocp_solvers.riccati_kkt).

Problem (as whoop.ocp_solvers.riccati_kkt): minimise sum_k 1/2 [x_k; u_k]' Q_k [x_k; u_k] + q_k' [x_k; u_k] + 1/2 x_N' QN x_N
+ qN' x_N subject to x_{k+1} = A_k x_k + B_k u_k + e_k, x_0 fixed except one component (the first leg's duration).
Returns the step (X, U), the defects' multipliers L (row k: x_{k+1} - A x_k - B u_k) and the start's, in whoop.ocp's
convention (stationarity Q z + q + J' lam = 0).

``riccati_scan``: the backward Riccati recursion as a lax.scan (N sequential stages, each a small dense solve).
``riccati_parallel``: S. Sarkka and A. F. Garcia-Fernandez, "Temporal parallelization of dynamic programming and linear
quadratic control", IEEE TAC 68 (2023): each stage an element (A, b, C, eta, J) of an associative operation whose
suffix combinations are the value functions V_k(x) = 1/2 x' J x - eta' x, by lax.associative_scan in O(log N) depth;
the forward pass an associative scan of affine maps; the multipliers the value functions' gradients. The cross and
linear control terms are removed first (u = v - Quu^-1 (Qux x + qu)), so each stage's Quu must be positive definite,
a stronger condition than the sequential recursion's (Quu + B' P B).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from trajopt_jax._jax import Array, jax, jnp


class LQR(NamedTuple):
    """An LQR's data: stage costs 1/2 z' Q z + q' z on z = [x; u] (Q (N, n, n), q (N, n)), the terminal's QN and qN,
    and the dynamics x' = A x + B u + e. What riccati_scan, riccati_parallel and whoop.ocp_solvers.riccati_kkt solve."""

    Q: Array
    QN: Array
    q: Array
    qN: Array
    A: Array
    B: Array
    e: Array


def _solve_spd(M: Array, R: Array) -> Array:
    c = jnp.linalg.cholesky(M)
    return jax.scipy.linalg.cho_solve((c, True), R)


def _start(S0: Array, v0: Array, x0: Array, free: Any) -> Array:
    """x_0: fixed except the components ``free`` (an index or indices), which minimise 1/2 x' S0 x - v0' x (their
    block's Cholesky: not positive definite gives NaN, which the caller's inertia correction catches)."""
    f = np.atleast_1d(free)
    x = jnp.asarray(x0).at[f].set(0.0)
    return x.at[f].set(_solve_spd(S0[np.ix_(f, f)], v0[f] - S0[f] @ x))


def riccati_scan(lqr: LQR, x0: Array, free: Any) -> tuple:
    """The backward Riccati recursion and the forward rollout as lax.scans (see the module docstring); ``x0`` the
    start, fixed but its components ``free`` (an index or indices)."""
    Q, QN, q, qN, A, B, e = lqr
    nx = A.shape[1]

    def back(carry: tuple, k: tuple) -> tuple:
        P, p = carry
        Qk, qk, Ak, Bk, ek = k
        pe = P @ ek + p
        Qxx = Qk[:nx, :nx] + Ak.T @ P @ Ak
        Quu = Qk[nx:, nx:] + Bk.T @ P @ Bk
        Qux = Qk[nx:, :nx] + Bk.T @ P @ Ak
        qx = qk[:nx] + Ak.T @ pe
        qu = qk[nx:] + Bk.T @ pe
        Kk = -_solve_spd(Quu, Qux)
        kk = -_solve_spd(Quu, qu)
        Pn = Qxx + Qux.T @ Kk
        return (0.5 * (Pn + Pn.T), qx + Qux.T @ kk), (Kk, kk, P, p)

    (P0, p0), (K, kff, Ps, ps) = jax.lax.scan(back, (QN, qN), (Q, q, A, B, e), reverse=True)
    x0s = _start(P0, -p0, x0, free)

    def fwd(x: Any, k: tuple) -> tuple:
        Kk, kk, Ak, Bk, ek = k
        u = Kk @ x + kk
        return Ak @ x + Bk @ u + ek, (x, u)

    xN, (Xs, U) = jax.lax.scan(fwd, x0s, (K, kff, A, B, e))
    X = jnp.concatenate([Xs, xN[None]])
    # Ps[k], ps[k]: the value function at stage k + 1 (V = 1/2 x' P x + p' x): the multipliers are its negative gradient
    L = -(jnp.einsum('kij,kj->ki', Ps, X[1:]) + ps)
    lam0 = -(P0 @ x0s + p0)
    return X, U, L, lam0


# the parallel form's control elimination: 'lu' (any nonsingular Quu; definiteness checked on Quu + B'SB, the stage
# Hessian with the cost-to-go, as the sequential recursion does) or 'spd' (Cholesky of each stage's own Quu: stricter
# than the inertia test; the hard start 374 iterations against 57, 2-gate co-design 181 against 75, 2026-11-18)
REDUCE = 'lu'
COMBINE = 'solve'  # the scan's combine: 'solve' (one LU of I + C_i J_j for both products) or 'inv' (explicit inverses)


def _combine(ei: tuple, ej: tuple) -> tuple:
    """Sarkka and Garcia-Fernandez's operation: element i (earlier) then element j (later), batched over a leading axis.
    Both products need (I + C_i J_j)^-1: (I + J_j C_i)^-1 is its transpose (C and J symmetric), so one LU serves both."""
    Ai, bi, Ci, ei_, Ji = ei
    Aj, bj, Cj, ej_, Jj = ej
    nx = Ai.shape[-1]
    T = lambda m: jnp.swapaxes(m, -1, -2)  # noqa: E731 - batched transpose, a one-line shorthand
    mv = lambda m, v: jnp.einsum('...ij,...j->...i', m, v)  # noqa: E731 - batched matrix-vector, a one-line shorthand
    K = jnp.eye(nx, dtype=Ci.dtype) + Ci @ Jj
    if COMBINE == 'inv':
        M = jnp.linalg.inv(K)

        def left(R: Any) -> Any:
            return M @ R

        def left_t(R: Any) -> Any:
            return T(M) @ R
    else:
        lu, piv = jax.vmap(jax.scipy.linalg.lu_factor)(K.reshape(-1, nx, nx)) if K.ndim > 2 else jax.scipy.linalg.lu_factor(K)
        lu, piv = lu.reshape(K.shape), piv.reshape(K.shape[:-1])
        solve = jnp.vectorize(lambda a, p, r: jax.scipy.linalg.lu_solve((a, p), r), signature='(n,n),(n),(n,k)->(n,k)')
        solve_t = jnp.vectorize(lambda a, p, r: jax.scipy.linalg.lu_solve((a, p), r, trans=1), signature='(n,n),(n),(n,k)->(n,k)')

        def left(R: Any) -> Any:
            return solve(lu, piv, R)

        def left_t(R: Any) -> Any:
            return solve_t(lu, piv, R)

    col = lambda v: v[..., None]  # noqa: E731 - a vector as a column, a one-line shorthand
    A = Aj @ left(Ai)
    b = mv(Aj, left(col(bi + mv(Ci, ej_)))[..., 0]) + bj
    C = Aj @ left(Ci @ T(Aj)) + Cj
    eta = mv(T(Ai), left_t(col(ej_ - mv(Jj, bi)))[..., 0]) + ei_
    J = T(Ai) @ left_t(Jj @ Ai) + Ji
    return A, b, 0.5 * (C + T(C)), eta, 0.5 * (J + T(J))


def riccati_parallel(lqr: LQR, x0: Array, free: Any) -> tuple:
    """The same solution by associative scans (see the module docstring)."""
    Q, QN, q, qN, A, B, e = lqr
    nx = A.shape[1]
    Qxx, Qxu, Quu = Q[:, :nx, :nx], Q[:, :nx, nx:], Q[:, nx:, nx:]
    qx, qu = q[:, :nx], q[:, nx:]
    # u = v - G x - g removes the cross and linear control terms
    red = _solve_spd if REDUCE == 'spd' else jnp.linalg.solve
    G = jax.vmap(red)(Quu, jnp.swapaxes(Qxu, 1, 2))
    g = jax.vmap(red)(Quu, qu[..., None])[..., 0]
    Qt = Qxx - Qxu @ G
    qt = qx - jnp.einsum('kji,kj->ki', G, qu)
    At = A - B @ G
    et = e - jnp.einsum('kij,kj->ki', B, g)
    Ruinv = jax.vmap(lambda M: red(M, jnp.eye(M.shape[0], dtype=M.dtype)))(Quu)
    Cs = B @ Ruinv @ jnp.swapaxes(B, 1, 2)
    elems = (
        jnp.concatenate([At, jnp.zeros((1, nx, nx), At.dtype)]),
        jnp.concatenate([et, jnp.zeros((1, nx), et.dtype)]),
        jnp.concatenate([Cs, jnp.zeros((1, nx, nx), Cs.dtype)]),
        jnp.concatenate([-qt, -qN[None]]),
        jnp.concatenate([Qt, QN[None]]),
    )
    # suffix combinations: V_k = e_k (+) e_{k+1} (+) ... (+) e_N (lax.associative_scan's reverse passes (later, earlier))
    suf = jax.lax.associative_scan(lambda later, earlier: _combine(earlier, later), elems, reverse=True)
    S, v = suf[4], suf[3]  # V_k(x) = 1/2 x' S_k x - v_k' x
    x0s = _start(S[0], v[0], x0, free)
    # the policy at each stage from the next stage's value function: v_ = -K x + k (in the transformed control)
    Sn, vn = S[1:], v[1:]
    H = Quu + jnp.swapaxes(B, 1, 2) @ Sn @ B
    rhs_x = jnp.swapaxes(B, 1, 2) @ Sn @ At
    rhs_c = jnp.einsum('kji,kj->ki', B, vn - jnp.einsum('kij,kj->ki', Sn, et))
    Kv = jax.vmap(_solve_spd)(H, rhs_x)
    kv = jax.vmap(_solve_spd)(H, rhs_c[..., None])[..., 0]
    # the closed loop x_{k+1} = (At - B Kv) x + (et + B kv): affine maps composed by an associative scan
    Mk = At - B @ Kv
    mk = et + jnp.einsum('kij,kj->ki', B, kv)

    def compose(f: tuple, h: tuple) -> tuple:  # f earlier, h later: x -> h(f(x))
        return h[0] @ f[0], jnp.einsum('...ij,...j->...i', h[0], f[1]) + h[1]

    Mc, mc = jax.lax.associative_scan(compose, (Mk, mk))
    X = jnp.concatenate([x0s[None], jnp.einsum('kij,j->ki', Mc, x0s) + mc])
    V = -jnp.einsum('kij,kj->ki', Kv, X[:-1]) + kv
    U = V - jnp.einsum('kij,kj->ki', G, X[:-1]) - g
    L = -(jnp.einsum('kij,kj->ki', S[1:], X[1:]) - v[1:])
    lam0 = -(S[0] @ x0s - v[0])
    return X, U, L, lam0


def eliminate_equalities(Q: Array, q: Array, A: Array, B: Array, e: Array, Hx: Array, Hu: Array, h: Array) -> tuple:
    """Stage equalities Hx_k x_k + Hu_k u_k + h_k = 0 (the Radau form's collocation defects, linearised) removed stage
    by stage: with Hu_k of full row rank, u_k = F_k x_k + f_k + Z_k w_k (Z_k an orthonormal basis of Hu_k's null space
    from its QR), leaving an LQR in (x, w) of the same shape for riccati_scan or riccati_parallel. Returns that LQR's
    (Q, q, A, B, e) and the maps (F, f, Z) for ``restore``."""
    nx, nu, nh = A.shape[1], B.shape[2], Hu.shape[1]
    # Hu' = [Y Z] R: Y spans Hu's row space, Z its null space (u = Y a + Z w; Hu Y = R1', so a = -R1'^-1 (Hx x + h))
    Qf, Rf = jnp.linalg.qr(jnp.swapaxes(Hu, 1, 2), mode='complete')
    Y, Z = Qf[:, :, :nh], Qf[:, :, nh:]
    R1t = jnp.swapaxes(Rf[:, :nh, :], 1, 2)
    F = -Y @ jnp.linalg.solve(R1t, Hx)
    f = -jnp.einsum('kij,kj->ki', Y, jnp.linalg.solve(R1t, h[..., None])[..., 0])
    # [x; u] = T [x; w] + t, T = [[I, 0], [F, Z]], t = [0; f]
    dt = A.dtype
    eye = jnp.broadcast_to(jnp.eye(nx, dtype=dt), (A.shape[0], nx, nx))
    T = jnp.concatenate([jnp.concatenate([eye, jnp.zeros((A.shape[0], nx, nu - nh), dt)], 2), jnp.concatenate([F, Z], 2)], 1)
    t = jnp.concatenate([jnp.zeros((A.shape[0], nx), dt), f], 1)
    Tt = jnp.swapaxes(T, 1, 2)
    Qw = Tt @ Q @ T
    qw = jnp.einsum('kji,kj->ki', T, jnp.einsum('kij,kj->ki', Q, t) + q)
    Aw = A + B @ F
    Bw = B @ Z
    ew = e + jnp.einsum('kij,kj->ki', B, f)
    return Qw, qw, Aw, Bw, ew, (F, f, Z)


def restore(X: Array, W: Array, L: Array, maps: tuple, Q: Array, q: Array, B: Array, Hu: Array) -> tuple:
    """The controls and the equalities' multipliers from the reduced LQR's solution: u = F x + f + Z w, and nu from
    the control rows' stationarity Q_u z + q_u - B' L + Hu' nu = 0 (whoop.ocp's sign: defect rows x_{k+1} - A x - B u,
    equality rows + Hu u), by least squares on Hu' (exact: the residual lies in Hu's row space at the optimum)."""
    F, f, Z = maps
    nx = F.shape[2]
    U = jnp.einsum('kij,kj->ki', F, X[:-1]) + f + jnp.einsum('kij,kj->ki', Z, W)
    z = jnp.concatenate([X[:-1], U], 1)
    r = jnp.einsum('kij,kj->ki', Q[:, nx:, :], z) + q[:, nx:] - jnp.einsum('kji,kj->ki', B, L)
    Hut = jnp.swapaxes(Hu, 1, 2)
    nu = -jnp.linalg.solve(Hu @ Hut, jnp.einsum('kij,kj->ki', Hu, r)[..., None])[..., 0]
    return U, nu


def refined(solver: Any, rounds: int = 1) -> Any:
    """``solver`` (riccati_parallel or riccati_scan) with ``rounds`` of iterative refinement: the LQR's KKT residuals
    (dynamics, stationarity, the start) as the right-hand side of the same LQR, the correction added. The parallel
    scan's stationarity residual on the 10-gate lap's stages (condition ~1e10-1e12): 2.5e-6 at 1,100 stages and 2.5e-5
    at 11,000, against ~1e-14 for the sequential recursion; refined once, 5e-10 and 1.4e-9 (2026-11-17)."""

    def solve(lqr: LQR, x0: Array, free: Any) -> tuple:
        Q, QN, q, qN, A, B, e = lqr
        nx = A.shape[1]
        fixed = np.setdiff1d(np.arange(nx), np.atleast_1d(free))

        def round_(_: int, sol: tuple) -> tuple:
            X, U, L, lam0 = sol
            rd = X[1:] - jnp.einsum('kij,kj->ki', A, X[:-1]) - jnp.einsum('kij,kj->ki', B, U) - e
            g = jnp.einsum('kij,kj->ki', Q, jnp.concatenate([X[:-1], U], 1)) + q
            rx = g[:, :nx].at[1:].add(L[:-1]) - jnp.einsum('kji,kj->ki', A, L)
            rx = rx.at[0, fixed].add(lam0[fixed])
            ru = g[:, nx:] - jnp.einsum('kji,kj->ki', B, L)
            rN = QN @ X[-1] + qN + L[-1]
            d0 = jnp.zeros(nx, X.dtype).at[fixed].set((X[0] - x0)[fixed])
            # the correction is the same LQR's solution for the residuals (signs as its data: stationarity -r,
            # dynamics +rd, start +d0)
            dX, dU, dL, dl0 = solver(lqr._replace(q=-jnp.concatenate([rx, ru], 1), qN=-rN, e=rd), d0, free)
            return X - dX, U - dU, L - dL, lam0 - dl0

        # (a fori_loop: the solver's program once however many rounds)
        X, U, L, lam0 = jax.lax.fori_loop(0, rounds, round_, solver(lqr, x0, free))
        return X, U, L, lam0

    return solve


__all__ = ['LQR', 'eliminate_equalities', 'refined', 'restore', 'riccati_parallel', 'riccati_scan']
