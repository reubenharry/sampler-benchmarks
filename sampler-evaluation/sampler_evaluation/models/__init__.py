import sys
import os

# list directories visible
print(os.listdir())
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.standardgaussian import Gaussian
from sampler_evaluation.models.german_credit import german_credit
from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
from sampler_evaluation.models.item_response import item_response


models = {
    "Banana": banana(),
    "Gaussian_100D": Gaussian(ndims=100),
    "Brownian_Motion": brownian_motion(),
    "German_Credit": german_credit(),
    # "Stochastic_Volatility": stochastic_volatility(),
    # "Item_Response": item_response(),
}
