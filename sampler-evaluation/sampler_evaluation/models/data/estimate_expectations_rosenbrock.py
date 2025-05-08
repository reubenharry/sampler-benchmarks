# from sampler_evaluation.models import brownian_motion
from sampler_evaluation.models.data.estimate_expectations import estimate_ground_truth
import sampler_evaluation

model = sampler_evaluation.models.rosenbrock.Rosenbrock(D=18)

estimate_ground_truth(model, num_samples=5000000, annealing=False)