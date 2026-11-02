"""Stand-in for ``tensorflow_datasets`` serving the two datasets Inference Gym needs.

The arrays in ``data/`` were exported once from TensorFlow Datasets: German
credit numeric (UCI Statlog) and the radon dataset (Gelman and Hill; rebuilt
with the tensorflow_datasets 4.9.10 builder from a GitHub mirror of
``srrs2.dat`` and ``cty.dat`` because the original host blocks scripted
downloads; Indiana and Minnesota rows only).
"""
import os

import numpy as np

OVERRIDE = {}  # name -> arrays supplied by the caller instead of files under ``data/``
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load(name, split=None, batch_size=None):
    if name.startswith("german_credit_numeric"):
        z = np.load(os.path.join(DATA_DIR, "german_credit_numeric.npz"))
        return {"train": [{"features": f, "label": l} for f, l in zip(z["features"], z["label"])]}
    if name.startswith("radon"):
        z = OVERRIDE["radon"] if "radon" in OVERRIDE else np.load(os.path.join(DATA_DIR, "radon_in_mn.npz"))
        return {"features": {"state": z["state"], "floor": z["floor"], "county": z["county"], "Uppm": z["uppm"]},
                "activity": z["activity"]}
    raise ValueError(name)


def as_numpy(dataset):
    return dataset
