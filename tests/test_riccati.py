"""The Radau stage and the Riccati recursions (moved from projects/whoop tests/test_ocp.py, 2026-11-22)."""

from __future__ import annotations

import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np  # noqa: E402

from trajopt_jax.radau import radau_iia3  # noqa: E402
from trajopt_jax.riccati import LQR  # noqa: E402


def test_radau_matrix_is_exact_on_cubics() -> None:
    c, D = radau_iia3()
    pts = np.concatenate([[0.0], c])
    coef = np.array([0.3, -1.2, 2.5, 0.7])
    p = np.polynomial.polynomial.polyval(pts, coef)
    dp = np.polynomial.polynomial.polyval(c, np.polynomial.polynomial.polyder(coef))
    np.testing.assert_allclose(D @ p, dp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c[-1], 1.0)


def test_riccati_matches_the_dense_kkt() -> None:
    """The Riccati recursion gives the dense KKT solution's step and multipliers (a random LQR, one start component
    free)."""
    from trajopt_jax.reference import riccati_kkt

    rng = np.random.default_rng(0)
    N, nx, nu = 6, 4, 2
    nv = nx + nu
    Q = np.array([(lambda M: M @ M.T + 0.1 * np.eye(nv))(rng.standard_normal((nv, nv))) for _ in range(N)])
    QN = (lambda M: M @ M.T + 0.1 * np.eye(nx))(rng.standard_normal((nx, nx)))
    q, qN = rng.standard_normal((N, nv)), rng.standard_normal(nx)
    A, B, e = rng.standard_normal((N, nx, nx)) * 0.5, rng.standard_normal((N, nx, nu)), rng.standard_normal((N, nx))
    x0 = rng.standard_normal(nx)
    X, U, L, lam0 = riccati_kkt(LQR(Q, QN, q, qN, A, B, e), x0, free=nx - 1)
    # the dense KKT: variables (x_0..x_N, u_0..u_{N-1}); rows: the start (all but the free component), the defects
    nzx, nz = (N + 1) * nx, (N + 1) * nx + N * nu
    ix = np.arange(nzx).reshape(N + 1, nx)
    iu = nzx + np.arange(N * nu).reshape(N, nu)
    H, g = np.zeros((nz, nz)), np.zeros(nz)
    for k in range(N):
        idx = np.concatenate([ix[k], iu[k]])
        H[np.ix_(idx, idx)] += Q[k]
        g[idx] += q[k]
    H[np.ix_(ix[N], ix[N])] += QN
    g[ix[N]] += qN
    J = np.zeros((nx - 1 + N * nx, nz))
    r = np.zeros(nx - 1 + N * nx)
    J[np.arange(nx - 1), ix[0, : nx - 1]] = 1.0
    r[: nx - 1] = x0[: nx - 1]
    for k in range(N):
        rows = nx - 1 + k * nx + np.arange(nx)
        J[np.ix_(rows, ix[k + 1])] = np.eye(nx)
        J[np.ix_(rows, ix[k])] = -A[k]
        J[np.ix_(rows, iu[k])] = -B[k]
        r[rows] = e[k]
    K = np.block([[H, J.T], [J, np.zeros((len(r), len(r)))]])
    sol = np.linalg.solve(K, np.concatenate([-g, r]))
    np.testing.assert_allclose(X.ravel(), sol[:nzx], atol=1e-9)
    np.testing.assert_allclose(U.ravel(), sol[nzx:nz], atol=1e-9)
    np.testing.assert_allclose(L.ravel(), sol[nz + nx - 1 :], atol=1e-9)
    np.testing.assert_allclose(lam0[: nx - 1], sol[nz : nz + nx - 1], atol=1e-9)


def test_jax_riccati_matches_numpy() -> None:
    """trajopt_jax.riccati's lax.scan and parallel-in-time (associative scan) forms give whoop.ocp.riccati_kkt's solution."""
    import jax

    jax.config.update('jax_enable_x64', True)
    from trajopt_jax.reference import riccati_kkt
    from trajopt_jax.riccati import riccati_parallel, riccati_scan

    rng = np.random.default_rng(1)
    N, nx, nu = 9, 5, 3
    nv = nx + nu
    Q = np.array([(lambda M: M @ M.T + 0.5 * np.eye(nv))(rng.standard_normal((nv, nv))) for _ in range(N)])
    QN = (lambda M: M @ M.T + 0.5 * np.eye(nx))(rng.standard_normal((nx, nx)))
    q, qN = rng.standard_normal((N, nv)), rng.standard_normal(nx)
    A, B, e = rng.standard_normal((N, nx, nx)) * 0.4, rng.standard_normal((N, nx, nu)), rng.standard_normal((N, nx))
    x0 = rng.standard_normal(nx)
    ref = riccati_kkt(LQR(Q, QN, q, qN, A, B, e), x0, nx - 1)
    for fn in (riccati_scan, riccati_parallel):
        got = fn(LQR(Q, QN, q, qN, A, B, e), x0, nx - 1)
        for a, b in zip(got, ref, strict=True):
            np.testing.assert_allclose(np.asarray(a), b, atol=1e-8, err_msg=getattr(fn, '__name__', 'refined'))


def test_parallel_riccati_takes_an_indefinite_stage_control_hessian(monkeypatch: object) -> None:
    """A stage whose own control Hessian is indefinite while the cost-to-go makes its reduced Hessian positive definite
    (a far start's stages): the sequential recursion solves it, and so does the parallel form with the LU control
    elimination (REDUCE 'lu'); the Cholesky elimination gave NaN, and the inertia ladder then drove the shift to 1e9
    and the hard start to 374 iterations against 63 (2026-11-18)."""
    import jax

    jax.config.update('jax_enable_x64', True)
    import trajopt_jax.riccati as ric
    from trajopt_jax.reference import riccati_kkt

    rng = np.random.default_rng(2)
    N, nx, nu = 8, 4, 2
    nv = nx + nu
    Q = np.array([(lambda M: M @ M.T + 0.5 * np.eye(nv))(rng.standard_normal((nv, nv))) for _ in range(N)])
    Q[3, nx:, nx:] = -0.3 * np.eye(nu)  # indefinite on its own
    Q[3, :nx, nx:] = 0.0
    Q[3, nx:, :nx] = 0.0
    QN = 50.0 * np.eye(nx)
    q, qN = rng.standard_normal((N, nv)), rng.standard_normal(nx)
    A, B, e = rng.standard_normal((N, nx, nx)) * 0.4 + np.eye(nx), rng.standard_normal((N, nx, nu)), rng.standard_normal((N, nx))
    x0 = rng.standard_normal(nx)
    ref = riccati_kkt(LQR(Q, QN, q, qN, A, B, e), x0, nx - 1)
    for a, b in zip(ric.riccati_scan(LQR(Q, QN, q, qN, A, B, e), x0, nx - 1), ref, strict=True):
        np.testing.assert_allclose(np.asarray(a), b, atol=1e-8)
    monkeypatch.setattr(ric, 'REDUCE', 'spd')
    assert not np.all(np.isfinite(np.asarray(ric.riccati_parallel(LQR(Q, QN, q, qN, A, B, e), x0, nx - 1)[0])))
    monkeypatch.setattr(ric, 'REDUCE', 'lu')
    for a, b in zip(ric.riccati_parallel(LQR(Q, QN, q, qN, A, B, e), x0, nx - 1), ref, strict=True):
        np.testing.assert_allclose(np.asarray(a), b, atol=1e-7)


def test_riccati_with_stage_equalities_matches_the_dense_kkt() -> None:
    """Stage equalities Hx x + Hu u + h = 0 eliminated stage by stage (trajopt_jax.riccati.eliminate_equalities), solved by
    the parallel Riccati, restored: the dense KKT's step and every multiplier."""
    import jax

    jax.config.update('jax_enable_x64', True)
    from trajopt_jax.riccati import eliminate_equalities, restore, riccati_parallel

    rng = np.random.default_rng(2)
    N, nx, nu, nh = 5, 4, 5, 2
    nv = nx + nu
    Q = np.array([(lambda M: M @ M.T + 0.5 * np.eye(nv))(rng.standard_normal((nv, nv))) for _ in range(N)])
    QN = (lambda M: M @ M.T + 0.5 * np.eye(nx))(rng.standard_normal((nx, nx)))
    q, qN = rng.standard_normal((N, nv)), rng.standard_normal(nx)
    A, B, e = rng.standard_normal((N, nx, nx)) * 0.4, rng.standard_normal((N, nx, nu)), rng.standard_normal((N, nx))
    Hx, Hu, h = rng.standard_normal((N, nh, nx)), rng.standard_normal((N, nh, nu)), rng.standard_normal((N, nh))
    x0 = rng.standard_normal(nx)
    Qw, qw, Aw, Bw, ew, maps = eliminate_equalities(Q, q, A, B, e, Hx, Hu, h)
    X, W, L, lam0 = riccati_parallel(LQR(Qw, QN, qw, qN, Aw, Bw, ew), x0, nx - 1)
    U, nu_h = restore(X, W, L, maps, Q, q, B, Hu)
    # dense: variables (x_0..x_N, u_0..); rows: start (but the free component), defects, equalities
    nzx, nz = (N + 1) * nx, (N + 1) * nx + N * nu
    ix = np.arange(nzx).reshape(N + 1, nx)
    iu = nzx + np.arange(N * nu).reshape(N, nu)
    H, g = np.zeros((nz, nz)), np.zeros(nz)
    for k in range(N):
        idx = np.concatenate([ix[k], iu[k]])
        H[np.ix_(idx, idx)] += Q[k]
        g[idx] += q[k]
    H[np.ix_(ix[N], ix[N])] += QN
    g[ix[N]] += qN
    m = nx - 1 + N * nx + N * nh
    J, r = np.zeros((m, nz)), np.zeros(m)
    J[np.arange(nx - 1), ix[0, : nx - 1]] = 1.0
    r[: nx - 1] = x0[: nx - 1]
    for k in range(N):
        rows = nx - 1 + k * nx + np.arange(nx)
        J[np.ix_(rows, ix[k + 1])] = np.eye(nx)
        J[np.ix_(rows, ix[k])] = -A[k]
        J[np.ix_(rows, iu[k])] = -B[k]
        r[rows] = e[k]
        rows = nx - 1 + N * nx + k * nh + np.arange(nh)
        J[np.ix_(rows, ix[k])] = Hx[k]
        J[np.ix_(rows, iu[k])] = Hu[k]
        r[rows] = -h[k]
    K = np.block([[H, J.T], [J, np.zeros((m, m))]])
    sol = np.linalg.solve(K, np.concatenate([-g, r]))
    np.testing.assert_allclose(np.asarray(X).ravel(), sol[:nzx], atol=1e-8)
    np.testing.assert_allclose(np.asarray(U).ravel(), sol[nzx:nz], atol=1e-8)
    np.testing.assert_allclose(np.asarray(L).ravel(), sol[nz + nx - 1 : nz + nx - 1 + N * nx], atol=1e-8)
    np.testing.assert_allclose(np.asarray(nu_h).ravel(), sol[nz + nx - 1 + N * nx :], atol=1e-8)
    np.testing.assert_allclose(np.asarray(lam0)[: nx - 1], sol[nz : nz + nx - 1], atol=1e-8)


def test_riccati_with_several_free_start_components() -> None:
    """Co-design frees the design variables at the start beside the duration: numpy, lax.scan and parallel Riccati against the
    dense KKT with start rows only for the fixed components."""
    import jax

    jax.config.update('jax_enable_x64', True)
    from trajopt_jax.reference import riccati_kkt
    from trajopt_jax.riccati import refined, riccati_parallel, riccati_scan

    rng = np.random.default_rng(3)
    N, nx, nu = 6, 5, 2
    nv = nx + nu
    free = np.array([2, 3, nx - 1])
    fixed = np.setdiff1d(np.arange(nx), free)
    Q = np.array([(lambda M: M @ M.T + 0.5 * np.eye(nv))(rng.standard_normal((nv, nv))) for _ in range(N)])
    QN = (lambda M: M @ M.T + 0.5 * np.eye(nx))(rng.standard_normal((nx, nx)))
    q, qN = rng.standard_normal((N, nv)), rng.standard_normal(nx)
    A, B, e = rng.standard_normal((N, nx, nx)) * 0.4, rng.standard_normal((N, nx, nu)), rng.standard_normal((N, nx))
    x0 = rng.standard_normal(nx)
    nzx, nz = (N + 1) * nx, (N + 1) * nx + N * nu
    ix = np.arange(nzx).reshape(N + 1, nx)
    iu = nzx + np.arange(N * nu).reshape(N, nu)
    H, g = np.zeros((nz, nz)), np.zeros(nz)
    for k in range(N):
        idx = np.concatenate([ix[k], iu[k]])
        H[np.ix_(idx, idx)] += Q[k]
        g[idx] += q[k]
    H[np.ix_(ix[N], ix[N])] += QN
    g[ix[N]] += qN
    m = len(fixed) + N * nx
    J, r = np.zeros((m, nz)), np.zeros(m)
    J[np.arange(len(fixed)), ix[0, fixed]] = 1.0
    r[: len(fixed)] = x0[fixed]
    for k in range(N):
        rows = len(fixed) + k * nx + np.arange(nx)
        J[np.ix_(rows, ix[k + 1])] = np.eye(nx)
        J[np.ix_(rows, ix[k])] = -A[k]
        J[np.ix_(rows, iu[k])] = -B[k]
        r[rows] = e[k]
    sol = np.linalg.solve(np.block([[H, J.T], [J, np.zeros((m, m))]]), np.concatenate([-g, r]))
    for fn in (riccati_kkt, riccati_scan, riccati_parallel, refined(riccati_parallel)):
        X, U, L, lam0 = (np.asarray(o) for o in fn(LQR(Q, QN, q, qN, A, B, e), x0, free))
        np.testing.assert_allclose(X.ravel(), sol[:nzx], atol=1e-8, err_msg=getattr(fn, '__name__', 'refined'))
        np.testing.assert_allclose(U.ravel(), sol[nzx:nz], atol=1e-8, err_msg=getattr(fn, '__name__', 'refined'))
        np.testing.assert_allclose(L.ravel(), sol[nz + len(fixed) :], atol=1e-8, err_msg=getattr(fn, '__name__', 'refined'))
        np.testing.assert_allclose(lam0[fixed], sol[nz : nz + len(fixed)], atol=1e-8, err_msg=getattr(fn, '__name__', 'refined'))


def _count_equations(jaxpr) -> int:
    """Equations in a jaxpr, those nested in loops, conditionals and custom-derivative rules included."""
    n = 0
    for eqn in jaxpr.eqns:
        n += 1
        for p in eqn.params.values():
            for sub in p if isinstance(p, (list, tuple)) else [p]:
                inner = getattr(sub, 'jaxpr', None) or (sub if hasattr(sub, 'eqns') else None)
                if inner is not None and hasattr(inner, 'eqns'):
                    n += _count_equations(inner)
    return n


def test_the_guarded_parallel_riccati_falls_back_to_the_scan_when_inexact() -> None:
    """``guarded``: the relative KKT residual of an exact solve is near zero and of a perturbed one is not; with the
    tolerance unreachable the scan's solution comes back, with it generous the refined parallel one, both exact."""
    import jax

    jax.config.update('jax_enable_x64', True)
    from trajopt_jax.riccati import guarded, relative_residual, riccati_scan

    rng = np.random.default_rng(7)
    N, nx, nu = 40, 5, 3
    nv = nx + nu
    Q = np.array([(lambda M: M @ M.T + 0.5 * np.eye(nv))(rng.standard_normal((nv, nv))) for _ in range(N)])
    QN = (lambda M: M @ M.T + 0.5 * np.eye(nx))(rng.standard_normal((nx, nx)))
    q, qN = rng.standard_normal((N, nv)), rng.standard_normal(nx)
    A, B, e = rng.standard_normal((N, nx, nx)) * 0.4, rng.standard_normal((N, nx, nu)), rng.standard_normal((N, nx))
    lqr, x0 = LQR(Q, QN, q, qN, A, B, e), rng.standard_normal(nx)
    scan = riccati_scan(lqr, x0, nx - 1)
    assert float(relative_residual(lqr, x0, nx - 1, scan)) < 1e-10
    off = (scan[0] + 1e-3, *scan[1:])
    assert float(relative_residual(lqr, x0, nx - 1, off)) > 1e-5
    for tol in (0.0, 1.0):  # (0: never accepted, the scan; 1: always, the refined parallel solve)
        got = guarded(tol=tol)(lqr, x0, nx - 1)
        for a, b in zip(got, scan, strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-8)
