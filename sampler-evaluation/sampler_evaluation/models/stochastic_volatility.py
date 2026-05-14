import sys
#sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym")
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp
import numpy as np
import pickle
import os
module_dir = os.path.dirname(os.path.abspath(__file__))


def _load_expectations_stats(model_name: str) -> dict:
    """Load ``e_x2`` / ``e_x4`` (and optional keys) from disk.

    Legacy ``*_expectations.pkl`` files may contain pickled JAX arrays whose
    serialized ``aval`` metadata includes ``named_shape``. That field was
    removed from ``ShapedArray`` in newer JAX, so unpickling fails unless we
    strip it. Prefer the NumPy ``*_expectations.npz`` artifact when present.
    """
    base = f"{module_dir}/data/{model_name}_expectations"
    npz_path = f"{base}.npz"
    if os.path.isfile(npz_path):
        z = np.load(npz_path)
        return {k: z[k] for k in z.files}

    pkl_path = f"{base}.pkl"
    import jax._src.core as _core

    _orig_update = _core.ShapedArray.update

    def _update_no_named_shape(self, shape=None, dtype=None, weak_type=None, **kwargs):
        kwargs.pop("named_shape", None)
        return _orig_update(self, shape, dtype, weak_type, **kwargs)

    _core.ShapedArray.update = _update_no_named_shape
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    finally:
        _core.ShapedArray.update = _orig_update


def stochastic_volatility():

    stochastic_volatility = gym.targets.VectorModel(
        gym.targets.VectorizedStochasticVolatilityLogSP500(),
        flatten_sample_transformations=True,
    )

    stats = _load_expectations_stats(stochastic_volatility.name)

    e_x2 = jnp.asarray(stats["e_x2"])
    e_x4 = jnp.asarray(stats["e_x4"])
    var_x2 = e_x4 - e_x2**2

    stochastic_volatility.sample_transformations["square"] = (
        model.Model.SampleTransformation(
            fn=lambda params: stochastic_volatility.sample_transformations["identity"](params)**2,
            pretty_name="Square",
            ground_truth_mean=e_x2,
            ground_truth_standard_deviation=jnp.sqrt(var_x2),
        )
    )

    stochastic_volatility.ndims = 2519

    return stochastic_volatility
