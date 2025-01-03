from src.models.banana import banana
from src.models.standardgaussian import Gaussian


models = {
    "Gaussian_10D": Gaussian(ndims=10),
    "Banana": banana(),
}
