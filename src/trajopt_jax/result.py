"""A solve's result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IPMResult:
    X: np.ndarray
    U: np.ndarray
    obj: float
    iterations: int
    converged: bool
    wall_s: float
    history: list
    lam0: np.ndarray | None = None  # the start rows' multipliers (d obj*/d x0 = -lam0 on the fixed components)
    state: dict | None = None
    status: str = ''  # whoop.ipm: converged, max_iter, stalled (no accepted step for STALL iterations), nonfinite
