"""Catalog of hierarchical problems: targets, hierarchical/latent split, references."""
from typing import Callable, NamedTuple

import numpy as np

from models import hierarchical
from models.hierarchical import HIERARCHICAL_COMPONENTS

FUNNELS = {
    "funnel_d10": lambda: hierarchical.neal_funnel(10),
    "funnel_d50": lambda: hierarchical.neal_funnel(50),
    "funnel_data_d20": lambda: hierarchical.funnel_with_data(20, noise_std=1.0, seed=1),
    "funnel_data_d100": lambda: hierarchical.funnel_with_data(100, noise_std=0.5, seed=2, num_observed=50),
}


class HierarchicalTarget(NamedTuple):
    """A sampling problem with its hierarchical/latent split and reference moments of ``x**2``."""

    name: str
    dim: int
    logdensity: Callable
    constrain: Callable
    sample_init: Callable
    hier_idx: np.ndarray
    latent_idx: np.ndarray
    ref_mean: np.ndarray
    ref_var: np.ndarray
    reference_source: str


def available_names():
    return list(HIERARCHICAL_COMPONENTS) + list(FUNNELS)


def build_hierarchical_target(name):
    """Build a catalog target (Inference Gym model or analytic funnel) with split and reference."""
    if name in FUNNELS:
        t = FUNNELS[name]()
        return HierarchicalTarget(name, t.dim, t.logdensity, t.constrain, t.sample_init, t.hier_idx,
                                  t.latent_idx, t.ref_mean_x2, t.ref_var_x2, t.details["reference"])
    from models.inference_gym_models import build_target
    from models.references import load_reference
    t = build_target(name)
    hier, latent = hierarchical.hierarchical_indices(name, t.gym_model, t.dim)
    mean, var, source = load_reference(t)
    return HierarchicalTarget(name, t.dim, t.logdensity, t.constrain, t.sample_init, hier, latent, mean, var, str(source))
