"""
Patch for TensorFlow Probability + JAX 0.7+ compatibility.

JAX 0.7+ removed `pytype_aval_mappings` from `jax.interpreters.xla` (it lives in
`jax.core`). TFP still writes to both; this module aliases the xla attribute to
the core dict so TFP imports work. Import this before any TFP JAX substrate:

    import tfp_jax_patch  # noqa: F401
    from tensorflow_probability.substrates.jax import bijectors
"""

import jax.interpreters.xla as xla
import jax.core

if not hasattr(xla, "pytype_aval_mappings"):
    xla.pytype_aval_mappings = jax.core.pytype_aval_mappings
