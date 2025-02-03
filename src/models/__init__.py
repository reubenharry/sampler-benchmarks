from src.models.banana import banana
from src.models.brownian import brownian_motion
from src.models.standardgaussian import Gaussian


models = {
    "Banana": banana(),
    "Gaussian_10D": Gaussian(ndims=10),
    "Brownian_Motion": brownian_motion(),
}
