"""Every function and method in ``src/trajopt_jax`` carries parameter and return type annotations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gate  # style checks: run at git land (pytest -m gate), not on branches

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 0
EXEMPT: tuple[str, ...] = ()


def missing() -> list[dict]:
    cmd = [
        sys.executable,
        '-m',
        'ruff',
        'check',
        'src/trajopt_jax',
        '--select',
        'ANN',
        '--ignore',
        'ANN002,ANN003,ANN401',
        '--output-format',
        'json',
        '--no-cache',
    ]
    if EXEMPT:
        cmd.extend(['--exclude', ','.join(EXEMPT)])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return json.loads(r.stdout or '[]')


def test_missing_type_hints_only_fall() -> None:
    found = missing()
    assert len(found) <= BASELINE, '\n'.join(f'{Path(x["filename"]).relative_to(ROOT)}:{x["location"]["row"]} {x["message"]}' for x in found[:40])
