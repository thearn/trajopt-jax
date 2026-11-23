"""3-point Radau IIA collocation on a unit segment."""

from __future__ import annotations

import math

import numpy as np


def radau_iia3() -> tuple[np.ndarray, np.ndarray]:
    """3-point Radau IIA on a unit segment: the points c (c3 = 1) and D (3, 4), the derivative at c of the cubic
    through the values at (0, c1, c2, 1)."""
    cs = np.array([0.0, (4.0 - math.sqrt(6.0)) / 10.0, (4.0 + math.sqrt(6.0)) / 10.0, 1.0])
    V = np.vander(cs, 4, increasing=True)  # the cubic's coefficients from its values: V^-1
    dV = np.array([[0.0, 1.0, 2.0 * c, 3.0 * c * c] for c in cs[1:]])
    return cs[1:], dV @ np.linalg.inv(V)
