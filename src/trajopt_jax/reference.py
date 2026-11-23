"""The CPU references on whoop.ocp.StageProblem: the Riccati recursion in numpy (``riccati_kkt``, what the device's
whoop.riccati is checked against) and IPOPT on the same stages (``solve_ipopt``). The numpy interior-point SQP that was
here (RK4 and BFGS only) was removed on 2026-11-22: whoop.ipm runs on the CPU too, the same code as on the GPU.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from trajopt_jax.problem import StageProblem
from trajopt_jax.result import IPMResult
from trajopt_jax.riccati import LQR


def riccati_kkt(lqr: LQR, x0: np.ndarray, free: Any) -> tuple:
    """The interior-point step's KKT system as an LQR, by a backward Riccati recursion: minimise
    sum_k 1/2 [x_k; u_k]' Q_k [x_k; u_k] + q_k' [x_k; u_k] + 1/2 x_N' QN x_N + qN' x_N subject to x_{k+1} = A_k x_k +
    B_k u_k + e_k and x_0 = ``x0`` except its components ``free`` (an index or indices: the first leg's duration, and the
    design variables when co-designing). O(N) in the stages against
    the sparse factorisation's fill-in; Q_k's control block plus B' P B must be positive definite (BFGS blocks and the
    barrier's G' D G are). Returns X (N + 1, nx), U (N, nu) and the defects' multipliers L (N, nx) in the convention
    of ``solve``'s KKT (stationarity Q z + q + J' lam = 0, J's defect rows x_{k+1} - A x_k - B u_k), and the start's."""
    Q, QN, q, qN, A, B, e = (np.asarray(a) for a in lqr)
    N, nx = A.shape[0], A.shape[1]
    P, p = QN.copy(), qN.copy()
    Ks, ks = [], []
    for k in range(N - 1, -1, -1):
        Ak, Bk = A[k], B[k]
        pe = P @ e[k] + p
        Qxx = Q[k][:nx, :nx] + Ak.T @ P @ Ak
        Quu = Q[k][nx:, nx:] + Bk.T @ P @ Bk
        Qux = Q[k][nx:, :nx] + Bk.T @ P @ Ak
        qx = q[k][:nx] + Ak.T @ pe
        qu = q[k][nx:] + Bk.T @ pe
        cho = np.linalg.cholesky(Quu)
        K = -np.linalg.solve(cho.T, np.linalg.solve(cho, Qux))
        kff = -np.linalg.solve(cho.T, np.linalg.solve(cho, qu))
        P = Qxx + Qux.T @ K
        P = 0.5 * (P + P.T)
        p = qx + Qux.T @ kff
        Ks.append(K)
        ks.append(kff)
    Ks, ks = Ks[::-1], ks[::-1]
    X = np.zeros((N + 1, nx))
    X[0] = x0
    f = np.atleast_1d(free)
    X[0, f] = 0.0
    # (the free block's curvature must be positive definite: Cholesky raises otherwise and the caller regularises)
    cf = np.linalg.cholesky(P[np.ix_(f, f)])
    X[0, f] = -np.linalg.solve(cf.T, np.linalg.solve(cf, p[f] + P[f] @ X[0]))
    U = np.zeros((N, B.shape[2]))
    for k in range(N):
        U[k] = Ks[k] @ X[k] + ks[k]
        X[k + 1] = A[k] @ X[k] + B[k] @ U[k] + e[k]
    # the multipliers backwards from the state columns' stationarity: lam_k (defect k's row) and the start's
    L = np.zeros((N, nx))
    L[N - 1] = -(QN @ X[N] + qN)
    for k in range(N - 1, 0, -1):
        rx = Q[k][:nx, :nx] @ X[k] + Q[k][:nx, nx:] @ U[k] + q[k][:nx]
        L[k - 1] = -rx + A[k].T @ L[k]
    rx0 = Q[0][:nx, :nx] @ X[0] + Q[0][:nx, nx:] @ U[0] + q[0][:nx]
    lam0 = -rx0 + A[0].T @ L[0]
    return X, U, L, lam0


def solve_ipopt(
    prob: StageProblem, X: np.ndarray, U: np.ndarray, x0: np.ndarray, max_iter: int = 1000, tol: float = 1e-6, print_file: str | None = None
) -> IPMResult:
    """IPOPT on the stage problem, its own limited-memory quasi-Newton Hessian: the reference the device solver's gates
    are checked against on the same stages. Needs cyipopt with a recent libstdc++ preloaded (scripts/jaxtx_solve.py).
    On the 2-gate course, Radau stages, it converged where per-stage damped BFGS blocks stalled (20 stages: 1.9288 s in
    67 iterations, 75 s, against the blocks' 2.0601 s at 1000; 10 stages 1.9312 s in 122, 69 s, against 872 and 444 s);
    a partitioned form (Griewank and Toint: element blocks for the path inequalities and each collocation point)
    wandered to slow feasible points. Both removed on 2026-11-22; the record is doc/PLAN-interop.md."""
    import cyipopt

    nlp_problem = _IpoptStages(prob, x0)
    nlp = cyipopt.Problem(
        n=nlp_problem.nz, m=len(nlp_problem.cl), problem_obj=nlp_problem, lb=np.full(nlp_problem.nz, -1e20), ub=np.full(nlp_problem.nz, 1e20),
        cl=nlp_problem.cl, cu=np.zeros(len(nlp_problem.cl)),
    )  # fmt: skip
    # acceptable: the last digits of dual infeasibility stall (10 stages: 1.9413 s by iteration ~600, then 1400 more)
    opts = dict(max_iter=max_iter, tol=tol, mu_strategy='adaptive', print_level=5, linear_solver='mumps', acceptable_tol=1e-4, acceptable_iter=15)
    opts |= dict(hessian_approximation='limited-memory', limited_memory_max_history=6)
    for k, v in opts.items():
        nlp.add_option(k, v)
    if print_file:
        nlp.add_option('output_file', print_file)
    z0 = np.concatenate([X.ravel(), U.ravel()])
    nj = len(nlp_problem.jacobian(z0))
    if nj != len(nlp_problem.jr):
        raise ValueError(f'pattern sizes: jacobian {nj} values for {len(nlp_problem.jr)} indices')
    t0 = time.time()
    z, info = nlp.solve(z0)
    Xs, Us = nlp_problem.unpack(z)
    lam0 = np.zeros(prob.nx)
    lam0[nlp_problem.fixed] = info['mult_g'][: len(nlp_problem.fixed)]  # (as the IPM's: d obj*/d x0 = -lam0)
    st = nlp_problem.state
    return IPMResult(Xs, Us, float(info['obj_val']), st['iters'], info['status'] == 0, time.time() - t0, st['history'], lam0)


class _IpoptStages:
    """cyipopt's problem object for a StageProblem: the variables (X, U) flat; the constraints the fixed start rows
    (the ODE's states and the first stick knot, at hover, as whoop.ipm; the leg duration free; co-designing with
    design_vars_free=False, the design variables too), the defects, the inequalities and the Radau stage equalities; the Jacobian's
    fixed pattern (start, defects: x_{k+1} identity, x_k and u_k dense; inequalities and equalities: x_k and u_k dense)."""

    def __init__(self, prob: StageProblem, x0: np.ndarray) -> None:
        self.prob, self.x0 = prob, x0
        N, nx, nu, ng, nh = prob.N, prob.nx, prob.nu, prob.ng, prob.nh
        self.nzx = (N + 1) * nx
        self.nz = self.nzx + N * nu
        ix = np.arange(self.nzx).reshape(N + 1, nx)
        iu = self.nzx + np.arange(N * nu).reshape(N, nu)
        self.ix = ix
        stage_idx = [np.concatenate([ix[k], iu[k]]) for k in range(N)] + [ix[N]]
        self.fixed = np.setdiff1d(np.arange(nx), prob.free)
        ncs, ncd = len(self.fixed), N * nx
        self.ncs = ncs
        jr: list = list(range(ncs))
        jc: list = [int(ix[0, i]) for i in self.fixed]
        blocks = [(ncs + k * nx + np.arange(nx), ix[k + 1], True) for k in range(N)]  # (x_{k+1}: identity)
        blocks += [(ncs + k * nx + np.arange(nx), stage_idx[k], False) for k in range(N)]
        blocks += [(ncs + ncd + k * ng + np.arange(ng), stage_idx[k], False) for k in range(N + 1)]
        nch = ncs + ncd + (N + 1) * ng
        blocks += [(nch + k * nh + np.arange(nh), stage_idx[k], False) for k in range(N if nh else 0)]
        for r, cols, ident in self._ordered(blocks, N):
            if ident:
                jr += list(r)
                jc += list(cols)
            else:
                a, b = np.meshgrid(r, cols, indexing='ij')
                jr += list(a.ravel())
                jc += list(b.ravel())
        self.jr, self.jc = np.array(jr), np.array(jc)
        self.cl = np.concatenate([np.zeros(ncs + ncd), np.full((N + 1) * ng, -1e20), np.zeros(N * nh)])
        self.state: dict[str, Any] = {'z': None, 'ev': None, 'iters': 0, 'history': []}

    @staticmethod
    def _ordered(blocks: list, N: int) -> list:
        """The pattern's blocks in the Jacobian values' order: each defect's identity then its dense (x_k, u_k), then
        the inequalities, then the equalities."""
        ident, dense = blocks[:N], blocks[N : 2 * N]
        return [b for pair in zip(ident, dense, strict=True) for b in pair] + blocks[2 * N :]

    def unpack(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N, nx, nu = self.prob.N, self.prob.nx, self.prob.nu
        return z[: self.nzx].reshape(N + 1, nx), z[self.nzx :].reshape(N, nu)

    def at(self, z: np.ndarray) -> dict:
        """The problem's evaluation (with Jacobians) at ``z``, kept for the next call at the same point."""
        if self.state['z'] is None or not np.array_equal(z, self.state['z']):
            Xz, Uz = self.unpack(z)
            self.state['ev'] = self.prob.evaluate(Xz, Uz, True)
            self.state['z'] = z.copy()
        return self.state['ev']

    def objective(self, z: np.ndarray) -> float:
        p = self.prob
        return float(z[self.ix[p.N, p.ci]] * p.ref[p.ci] / 10.0)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        p = self.prob
        g = np.zeros(self.nz)
        g[self.ix[p.N, p.ci]] = p.ref[p.ci] / 10.0
        return g

    def constraints(self, z: np.ndarray) -> np.ndarray:
        ev = self.at(z)
        Xz, _ = self.unpack(z)
        parts = [(Xz[0] - self.x0)[self.fixed], ev['c'].ravel(), ev['g'].ravel()]
        return np.concatenate(parts + ([ev['h'].ravel()] if self.prob.nh else []))

    def jacobianstructure(self) -> tuple:
        return self.jr, self.jc

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        ev = self.at(z)
        N, nx = self.prob.N, self.prob.nx
        dyn = np.hstack([np.ones((N, nx)), -np.concatenate([ev['A'], ev['B']], axis=2).reshape(N, -1)])
        ineq = np.concatenate([ev['Gx'][:N], ev['Gu']], axis=2).reshape(-1)
        parts = [np.ones(self.ncs), dyn.ravel(), ineq, ev['Gx'][N].ravel()]
        if self.prob.nh:
            parts.append(np.concatenate([ev['Hx'], ev['Hu']], axis=2).ravel())
        return np.concatenate(parts)

    def intermediate(self, alg_mod: int, iter_count: int, obj_value: float, inf_pr: float, inf_du: float, mu: float, *args: Any) -> bool:
        self.state['iters'] = iter_count
        self.state['history'].append(dict(it=iter_count, obj=obj_value, inf_pr=inf_pr, inf_du=inf_du, mu=mu))
        return True


__all__ = ['riccati_kkt', 'solve_ipopt']
