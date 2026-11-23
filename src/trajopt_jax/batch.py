"""Multi-start on the device solver (whoop.ipm): several starts of one problem solved at once, the step vmapped over
slots, finished members' slots refilled (iteration-level, as jaxipm), optional restarts from the leader and racing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from trajopt_jax._jax import jax, jnp
from trajopt_jax.ipm import ACCEPT_ITER, ACCEPT_TOL, HESSIAN_BUDGET, STALL, initial_state, make_step
from trajopt_jax.problem import StageProblem


@dataclass
class BatchResult:
    """One device solve of several starts: each member's (X, U), lap objective, iterations, convergence and failure."""

    X: np.ndarray  # (B, N + 1, nx)
    U: np.ndarray  # (B, N, nu)
    obj: np.ndarray  # (B,)
    iterations: np.ndarray  # (B,) at convergence or the cap
    converged: np.ndarray  # (B,) bool
    failed: np.ndarray  # (B,) bool: a non-finite iterate
    wall_s: float
    history: list
    raced: np.ndarray | None = None  # (B,) bool: stopped by racing (clearly slower than the leader, feasible)
    restart: np.ndarray | None = None  # (B,) bool: a start made from the leader, not given
    status: list | None = None  # each member's: converged, max_iter, stalled, nonfinite or raced (as solve's)


def solve_batch(
    prob: StageProblem,
    starts: list[tuple[np.ndarray, np.ndarray]],
    x0: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    mu0: float = 0.1,
    parallel: bool | None = None,
    chunk: int | None = None,
    slots: int | None = None,
    verbose: bool = True,
    restarts: int = 0,
    perturb: Any = None,
    race: float | None = None,
    race_age: int = 30,
    seed: int = 0,
) -> BatchResult:
    """The device iteration on several starts of one problem at once (multi-start): the step vmapped over ``slots``
    members (default: every start at once), its device loops running until every member is done. A member that
    converges (stationarity, feasibility and complementarity below ``tol``), fails (a non-finite iterate) or reaches
    ``max_iter`` is recorded and its slot takes the next start at the next iteration (iteration-level refill, as
    jaxipm: without it the batch computes every member until the slowest finishes); slots with nothing left are
    frozen. One compile for the batch.

    Coordination (each optional, measured against plain multi-start): ``restarts`` more starts made by ``perturb(X, U,
    rng)`` from the best converged member (the leader) once the given starts are taken; ``race``: a member at least
    ``race_age`` iterations old, feasible to 1e-3 and slower than the leader by more than this fraction is stopped
    and its slot reused (feasible only: an infeasible iterate can look faster than it is)."""
    if parallel is None:
        parallel = jax.devices()[0].platform == 'gpu'
    cfg = _BatchConfig(max_iter, tol, mu0, restarts, perturb, race, race_age, verbose)
    return _Batch(prob, list(starts), x0, cfg, slots, seed).run(parallel, chunk)


@dataclass(frozen=True)
class _BatchConfig:
    """solve_batch's run options: the iteration cap, tolerance, barrier start, restarts (made by ``perturb``), the
    race (fraction behind the leader, minimum age) and printing."""

    max_iter: int
    tol: float
    mu0: float
    restarts: int
    perturb: Any
    race: float | None
    race_age: int
    verbose: bool


class _Batch:
    """solve_batch's bookkeeping: the slots and which start each works on, the queue of given starts and the restarts
    made from the leader, each finished member's record; the device step vmapped over the slots, frozen where idle."""

    def __init__(self, prob: StageProblem, starts: list, x0: np.ndarray, cfg: _BatchConfig, slots: int | None, seed: int) -> None:
        self.prob, self.starts, self.x0, self.cfg = prob, starts, x0, cfg
        self.K0 = len(starts)
        K = self.K0 + cfg.restarts
        self.B = B = min(slots or self.K0, self.K0)
        self.rng = np.random.default_rng(seed)
        self.queue = list(range(self.K0))  # the given starts (restarts are made later, from the leader)
        self.owner = [self.queue.pop(0) for _ in range(B)]  # the start each slot is working on (-1: idle)
        self.X = np.zeros((K, prob.N + 1, prob.nx))
        self.U = np.zeros((K, prob.N, prob.nu))
        self.iters, self.conv, self.fail, self.raced = np.zeros(K, int), np.zeros(K, bool), np.zeros(K, bool), np.zeros(K, bool)
        self.lap_at = np.full(K, np.inf)  # each converged member's objective
        self.status = [''] * K
        self.stuck = np.zeros(B, int)  # each slot's iterations running with its iterate unmoved
        self.near = np.zeros(B, int)  # each slot's iterations running at the acceptable level
        self.age = np.zeros(B, int)
        self.made = 0  # restarts made

    def _fresh(self, k: int) -> dict:
        X, U = self.starts[k]
        return initial_state(self.prob, X, U, self.x0, self.cfg.mu0)

    def next_start(self) -> int:
        """The next given start, else a restart from the leader (perturbed), else none."""
        if self.queue:
            return self.queue.pop(0)
        if self.made < self.cfg.restarts and np.isfinite(self.lap_at).any():
            lead = int(np.argmin(self.lap_at))
            self.starts.append(self.cfg.perturb(self.X[lead], self.U[lead], self.rng))
            self.made += 1
            return len(self.starts) - 1
        return -1

    def _more(self) -> bool:
        leader = np.isfinite(self.lap_at).any()
        return bool(any(o >= 0 for o in self.owner) or (self.made < self.cfg.restarts and leader and any(o < 0 for o in self.owner)))

    def _batched(self, parallel: bool, chunk: int | None) -> Callable[..., Any]:
        """The step vmapped over the slots, frozen slots left as they were; the stage passes chunked to the memory
        budget across the batch (the node's own derivatives: the Hessian pass measured 1.6x, 3.75x co-designing,
        lighter; a floor of 8 ran 16 co-design starts out of 16 GB)."""
        prob = self.prob
        if chunk is None:
            per_stage = (prob.nx + prob.nu) ** 2
            if prob.node_second_order:
                per_stage = int(per_stage / (3.0 if prob.nl else 1.5))
            if self.B * (prob.N + 1) * per_stage > HESSIAN_BUDGET:
                chunk = max(1, HESSIAN_BUDGET // (2 * self.B * per_stage))
        step = make_step(prob, parallel, self.cfg.tol, chunk, jit=False)

        @jax.jit
        def batched(st: dict, frozen: Any) -> tuple:
            new, info = jax.vmap(step)(st)
            keep = lambda n, o: jnp.where(frozen.reshape((-1,) + (1,) * (n.ndim - 1)), o, n)  # noqa: E731 - a one-line leaf for tree.map
            return jax.tree.map(keep, new, st), info

        return batched

    def _verdict(self, b: int, info: dict) -> str | None:
        """Slot ``b``'s member's end, or None while it runs: converged (or acceptable: IPOPT's rule), non-finite, stalled,
        raced (a member at least race_age iterations old, feasible to 1e-3 and slower than the leader by more than the
        race fraction: feasible only, as an infeasible iterate can look faster than it is), or at max_iter."""
        tol, cfg = self.cfg.tol, self.cfg
        full = info['stat'][b] < tol and info['feas'][b] < tol and info['comp'][b] < tol
        self.near[b] = self.near[b] + 1 if info['stat'][b] < ACCEPT_TOL and info['feas'][b] < tol and info['comp'][b] < ACCEPT_TOL else 0
        self.stuck[b] = self.stuck[b] + 1 if info['alpha'][b] == 0.0 else 0
        lead = self.lap_at.min()
        lose = cfg.race is not None and np.isfinite(lead) and self.age[b] >= cfg.race_age and info['feas'][b] < 1e-3
        lose = lose and info['obj'][b] > lead * (1 + cfg.race)
        if full or self.near[b] >= ACCEPT_ITER:
            return 'converged' if full else 'acceptable'
        if not np.isfinite(info['obj'][b]):
            return 'nonfinite'
        if self.stuck[b] >= STALL:
            return 'stalled'
        if lose:
            return 'raced'
        return 'max_iter' if self.age[b] >= cfg.max_iter else None

    def _finish(self, b: int, verdict: str, st: dict, info: dict) -> dict:
        """Record slot ``b``'s member and give the slot the next start (or idle it); the batch state returned."""
        k = self.owner[b]
        ok = verdict in ('converged', 'acceptable')
        self.status[k] = verdict
        self.stuck[b] = self.near[b] = 0
        self.X[k], self.U[k] = np.asarray(st['X'][b]), np.asarray(st['U'][b])
        self.iters[k], self.conv[k], self.fail[k], self.raced[k] = self.age[b], ok, verdict in ('nonfinite', 'stalled'), verdict == 'raced'
        if ok:
            self.lap_at[k] = info['obj'][b]
        self.owner[b] = self.next_start()
        self.age[b] = 0
        if self.owner[b] >= 0:  # the slot takes the next start
            st = jax.tree.map(lambda a, n, b=b: a.at[b].set(n), st, self._fresh(self.owner[b]))
        return st

    def run(self, parallel: bool, chunk: int | None) -> BatchResult:
        t0 = time.time()
        st = jax.tree.map(lambda *xs: jnp.stack(xs), *[self._fresh(k) for k in self.owner])
        batched = self._batched(parallel, chunk)
        history, it = [], 0
        while self._more():
            for b in range(self.B):  # idle slots revived once a leader exists
                if self.owner[b] < 0:
                    self.owner[b] = self.next_start()
                    if self.owner[b] >= 0:
                        st = jax.tree.map(lambda a, n, b=b: a.at[b].set(n), st, self._fresh(self.owner[b]))
                        self.age[b] = 0
            it += 1
            st, info = batched(st, jnp.asarray([o < 0 for o in self.owner]))
            info = {k: np.asarray(v) for k, v in info.items()}
            self.age += 1
            for b in range(self.B):
                verdict = self._verdict(b, info) if self.owner[b] >= 0 else None
                if verdict is not None:
                    st = self._finish(b, verdict, st, info)
            finished = int(len(self.starts) - len(self.queue) - sum(o >= 0 for o in self.owner))
            history.append(dict(it=it, wall=time.time() - t0, finished=finished, best=float(self.lap_at.min())))
            if self.cfg.verbose and (it % 20 == 0 or it < 3):
                print(f'{it:4d} finished {finished}/{len(self.lap_at)} converged {self.conv.sum()} t {history[-1]["wall"]:.1f}', flush=True)
        n = len(self.starts)
        obj = np.array([self.prob.evaluate(self.X[k], self.U[k], False)['obj'] for k in range(n)])
        is_restart = np.arange(n) >= self.K0
        sl = slice(0, n)
        return BatchResult(
            self.X[sl],
            self.U[sl],
            obj,
            self.iters[sl],
            self.conv[sl],
            self.fail[sl],
            time.time() - t0,
            history,
            self.raced[sl],
            is_restart,
            self.status[sl],
        )


__all__ = ['BatchResult', 'solve_batch']
