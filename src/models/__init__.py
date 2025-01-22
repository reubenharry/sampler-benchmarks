from src.models.banana import banana
from src.models.standardgaussian import Gaussian


models = {
    "Banana": banana(),
    "Gaussian_10D": Gaussian(ndims=10),
}
