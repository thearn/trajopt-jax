"""No dead code in ``src/trajopt_jax``."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gate  # style checks: run at git land (pytest -m gate), not on branches

ROOT = Path(__file__).resolve().parents[1]


def test_no_unused_code() -> None:
    if not shutil.which('vulture') and subprocess.run([sys.executable, '-m', 'vulture', '--version'], capture_output=True).returncode:
        pytest.skip('vulture not installed (uv sync --extra test)')
    r = subprocess.run(
        [sys.executable, '-m', 'vulture', 'src/trajopt_jax', 'tests/vulture_whitelist.py', '--min-confidence', '60'],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, 'unused code:\n' + r.stdout
