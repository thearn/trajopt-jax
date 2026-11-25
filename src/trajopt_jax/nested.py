"""Nested co-design: an outer loop over a few design variables whose inner solve is the stage problem at a fixed design,
the design entering as model data (one compiled problem for every design) and the design gradient the optimum's
sensitivity to that data (trajopt_jax.envelope) chained through the data's own map by a vector-Jacobian product.

``Nested(prob, X, U, stage_data)`` solves at a design and returns (objective, gradient); each solve starts from the
last converged one, its multipliers kept and its slacks on the central path at ``warm_mu``, and a design far from it is
reached in steps of at most ``max_step`` (relative), each warm. ``trust_region`` is the outer loop: a box-constrained
trust-region BFGS for a few design variables.

Why (the wing, 2026-11-23/24): all-at-once co-design (the design as constant states) stalled at two designs both beaten
by a fixed one, and with the design's map inside every node it took 5.5 GB a 128-stage block; restarted at mu 0.1 a
design re-converged into another local optimum ~1 % away, which a design gradient cannot see; SLSQP's first step moved
the span 33 % (314 iterations, 20 min) where 5 % steps take ~20 each; scipy's trust-constr held every step to 1-2 %
(its barrier as large as the objective's gradient).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from trajopt_jax import ipm
from trajopt_jax._jax import jax, jnp
from trajopt_jax.envelope import data_sensitivity, with_data
from trajopt_jax.stages import Stages

WARM_MU = 1e-5
MAX_STEP = 0.05
TR_RADIUS0, TR_RADIUS_MIN, TR_RADIUS_MAX = 0.1, 1e-3, 0.4  # (the trust region's radii: max-norm steps in the outer variables)
TR_ACCEPT, TR_SHRINK, TR_GROW = 0.1, 0.25, 0.75  # (its ratio thresholds, actual over predicted reduction)
TR_GTOL = 1e-4  # (the projected gradient's max at which it stops)


class Nested:
    """The problem solved at a design: ``stage_data(p) -> row`` (JAX) puts the design into the model data's ``columns``
    of every stage (the rest from ``base``, each stage's own); ``start(p) -> (scaled start states, start controls)``, if
    given, makes the fixed start depend on the design too (a start trimmed for it), its share of the gradient through the
    start rows' multipliers. ``report(prob, res) -> dict`` adds fields to each solve's log line. ``scale``: each design
    variable's typical size, the continuation's steps at most MAX_STEP of it (default the last converged value's size, 1
    where that is 0: the whoop's stock prop pitch, whose first 1e-7 step went in 2400 steps). Each design's (objective,
    gradient) cached: optimisers ask for the function and the gradient at one point separately."""

    def __init__(
        self,
        prob: Stages,
        X: np.ndarray,
        U: np.ndarray,
        stage_data: Callable[[Any], Any],
        *,
        start: Callable[[Any], Any] | None = None,
        base: tuple[np.ndarray, np.ndarray] | None = None,
        columns: slice | None = None,
        report: Callable[[Stages, Any], dict] | None = None,
        scale: np.ndarray | None = None,
        max_iter: int = 500,
        verbose: bool = False,
    ) -> None:
        self.prob, self.X, self.U = prob, X, U
        self.stage_data, self.start, self.base, self.columns = stage_data, start, base, columns
        self.report, self.max_iter, self.verbose = report, max_iter, verbose
        self.scale = None if scale is None else np.asarray(scale, float)
        self.cache: dict[tuple, tuple[float, np.ndarray]] = {}
        self.state: dict | None = None
        self.at: np.ndarray | None = None  # (the design of the last converged solve, which the next one starts from)
        self.best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None  # (objective, design, X, U)
        self.log: list[dict] = []

    def __call__(self, design: np.ndarray) -> tuple[float, np.ndarray]:
        design = np.asarray(design, float)
        key = tuple(np.round(design, 9))
        if key in self.cache:
            return self.cache[key]
        if self.at is not None:  # (far from the last converged design: there in steps of at most MAX_STEP, each warm)
            size = self.scale if self.scale is not None else np.where(np.abs(self.at) > 1e-9, np.abs(self.at), 1.0)
            rel = np.abs(design - self.at) / size
            n, at = int(np.ceil(np.max(rel) / MAX_STEP)), self.at.copy()
            for i in range(1, n):
                self.solve(at + (design - at) * i / n, step=True)
        return self.solve(design)

    def _x0(self, p: Any) -> tuple[np.ndarray, Callable | None, int]:
        """The fixed start for ``p`` (the problem's own without ``start``), its VJP and its length."""
        if self.start is None:
            x0 = self.prob.x0(self.X[0, -1])
            return x0, None, 0
        head, to_head = jax.vjp(self.start, p)
        return np.concatenate([np.asarray(head), np.asarray(self.prob.x0(self.X[0, -1]))[len(head) :]]), to_head, len(head)

    def solve(self, design: np.ndarray, step: bool = False) -> tuple[float, np.ndarray]:
        """One warm solve at ``design``: (objective, gradient); a step's result is not cached."""
        prob, p, t = self.prob, jnp.asarray(design), time.time()
        row, to_design = jax.vjp(self.stage_data, p)
        data = with_data(prob, np.asarray(row), self.base, self.columns)
        x0, to_head, n_head = self._x0(p)
        warm = dict(warm=self.state, mu0=WARM_MU) if self.state is not None else {}
        res = ipm.solve(prob, self.X, self.U, x0, max_iter=self.max_iter, verbose=self.verbose, parallel=False, data=data, **warm)
        grad = np.asarray(to_design(jnp.asarray(data_sensitivity(prob, res.X, res.U, res.state, columns=self.columns)))[0])
        if to_head is not None:
            if res.lam0 is None:
                raise ValueError("a design-dependent start needs the start rows' multipliers (IPMResult.lam0)")
            grad = grad + np.asarray(to_head(-jnp.asarray(res.lam0[:n_head]))[0])
        if not np.all(np.isfinite(grad)):  # (a NaN gradient sent the outer loop to an all-NaN design: the whoop, 2026-11-25)
            raise FloatingPointError(f'design gradient not finite at {design} ({res.status} after {res.iterations} iterations): {grad}')
        if res.converged:  # (the next solve starts from this one: warm only from a converged point)
            self.X, self.U, self.state, self.at = res.X, res.U, res.state, design.copy()
            best = self.best
            if best is None or res.obj < best[0]:
                self.best = (float(res.obj), design.copy(), res.X, res.U)
        line = dict(design=design.tolist(), obj=float(res.obj), grad=grad.tolist(), status=res.status, iterations=int(res.iterations),
                    seconds=round(time.time() - t, 1), step=step)  # fmt: skip
        line |= self.report(prob, res) if self.report else {}
        self.log.append(line)
        print(f'{"step" if step else "eval"} {len(self.log)}: design {np.array2string(design, precision=4)} obj {res.obj:.6f} grad '
              f'{np.array2string(grad, precision=4)} {res.status} it {res.iterations} {line["seconds"]} s', flush=True)  # fmt: skip
        out = (float(res.obj), grad)
        if not step:
            self.cache[tuple(np.round(design, 9))] = out
        return out


def trust_region(
    fg: Callable[[np.ndarray], tuple[float, np.ndarray]],
    z: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    feasible: Callable[[np.ndarray], bool] | None = None,
    maxiter: int = 30,
) -> dict:
    """A box-constrained trust-region BFGS for a few outer variables (scaled to ~1): the BFGS step clipped to the radius
    (max-norm) and the box, shortened while ``feasible`` fails; accepted when the reduction is at least TR_ACCEPT of the
    model's, the radius halved below TR_SHRINK and doubled above TR_GROW at the radius. Stops on the projected gradient
    (TR_GTOL), the radius (TR_RADIUS_MIN) or ``maxiter``."""
    f, gz = fg(z)
    B = np.eye(len(z)) * max(np.linalg.norm(gz), 1e-12) / TR_RADIUS0  # (the first step: the radius, along -g)
    radius, message, it = TR_RADIUS0, 'max_iter', 0
    for it in range(1, maxiter + 1):
        free = ~(((z <= lo) & (gz > 0)) | ((z >= hi) & (gz < 0)))
        if np.max(np.abs(gz[free]), initial=0.0) < TR_GTOL:
            message = 'gradient'
            break
        p = np.zeros_like(z)
        p[free] = -np.linalg.solve(B[np.ix_(free, free)], gz[free])
        p *= min(1.0, radius / max(np.max(np.abs(p)), 1e-12))
        p = np.clip(z + p, lo, hi) - z
        while feasible is not None and not feasible(z + p) and np.max(np.abs(p)) > 1e-6:
            p *= 0.5
        pred = -(gz @ p + 0.5 * p @ B @ p)
        f1, g1 = fg(z + p)
        ratio = (f - f1) / pred if pred > 0 else -1.0
        y = g1 - gz
        if p @ y > 1e-12:  # (BFGS on every trial: its gradients are exact either way)
            Bp = B @ p
            B = B - np.outer(Bp, Bp) / (p @ Bp) + np.outer(y, y) / (p @ y)
        step = np.max(np.abs(p))
        print(f'outer {it}: f {f:.6f} -> {f1:.6f} ratio {ratio:.2f} step {step:.3f} radius {radius:.3f}', flush=True)
        if ratio >= TR_ACCEPT:
            z, f, gz = z + p, f1, g1
        radius = radius * 0.5 if ratio < TR_SHRINK else min(TR_RADIUS_MAX, radius * 2.0) if ratio > TR_GROW and step >= 0.9 * radius else radius
        if radius < TR_RADIUS_MIN:
            message = 'radius'
            break
    return dict(z=z, f=f, iterations=it, message=message)
