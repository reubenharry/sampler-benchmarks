"""Reference moments E[x_j^2] and Var[x_j^2] of the scored observable for catalog targets."""
import json
import os

import numpy as np

from models import analytic_references

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "references")


def load_reference(target):
    """Return ``(e_x2, var_x2, source)`` for a catalog target.

    Analytic where available, otherwise the validated NUTS product stored in
    ``references/<name>.npz`` (diagnostics in ``references/<name>.json``).
    """
    name = target.name
    if name == "banana":
        return (*analytic_references.banana_reference(target.dim), "analytic")
    if name == "ill_conditioned_gaussian":
        return (*analytic_references.gaussian_reference(target.gym_model), "analytic")
    path = os.path.join(REFERENCE_DIR, name + ".npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no validated reference for '{name}' in {REFERENCE_DIR}")
    data = np.load(path)
    with open(os.path.join(REFERENCE_DIR, name + ".json")) as handle:
        source = json.load(handle)["summary"]
    return data["e_x2"], data["var_x2"], source
