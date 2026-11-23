"""The rendezvous example's Hohmann check: with a strong engine and no launch speed, the least-propellant transfer
between circular orbits approaches the impulsive Hohmann transfer (its speed change, and the first burn's centre and
phase angle against the Hohmann departure's)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import rendezvous as rz  # noqa: E402 - the example beside the tests


def test_the_strong_engine_rendezvous_is_the_hohmann_transfer() -> None:
    out, h = rz.solve(rz.HOHMANN_CASE), rz.hohmann(rz.HOHMANN_CASE)
    assert out['converged']
    assert out['dv_kms'] == pytest.approx(h['dv_kms'], rel=0.005)  # (finite 10-day burns: gravity losses ~0.2 %)
    assert out['burn_centre_to_arrival_yr'] == pytest.approx(h['flight_yr'], abs=0.01)  # (about 4 days)
    assert out['lead_at_burn_centre_deg'] == pytest.approx(h['lead_deg'], abs=1.5)


def test_the_low_thrust_default_converges_and_uses_its_launch() -> None:
    out = rz.solve(rz.Case())
    assert out['converged']
    assert out['vinf_kms'] == pytest.approx(3.0, abs=0.01)
    assert 100.0 < out['propellant_kg'] < 300.0
