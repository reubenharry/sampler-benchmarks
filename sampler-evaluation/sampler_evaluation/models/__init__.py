import sys
import os
# list directories visible
print(os.listdir())
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.standardgaussian import Gaussian


models = {
    "Banana": banana(),
    "Gaussian_10D": Gaussian(ndims=10),
    # "Brownian_Motion": brownian_motion(),
}
