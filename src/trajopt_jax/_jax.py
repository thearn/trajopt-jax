"""JAX in float64 (interior-point steps need it). A caller that configures JAX itself (designloop's jax_env: the
compile cache) can import that first; this only turns float64 on."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

Array = jax.Array

__all__ = ['Array', 'jax', 'jnp']
