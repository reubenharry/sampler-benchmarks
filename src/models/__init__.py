from src.models.banana import banana
from src.models.standardgaussian import Gaussian


models = {
    "Gaussian" : Gaussian(ndims=100),
    "Banana" : banana(),
    }