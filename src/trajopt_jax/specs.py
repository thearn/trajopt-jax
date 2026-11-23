"""A vehicle model's declarations (doc: the vehicle interface): states with their scales, controls with their box and
rate limits, path constraints on the model's outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateSpec:
    name: str
    unit: str | None
    ref: float  # the scale the transcription divides by


@dataclass(frozen=True)
class ControlSpec:
    name: str
    lo: float
    hi: float
    rate_max: float | None = None  # 1/s in the control's units


@dataclass(frozen=True)
class PathConstraint:
    output: str
    lo: float | None
    hi: float | None
