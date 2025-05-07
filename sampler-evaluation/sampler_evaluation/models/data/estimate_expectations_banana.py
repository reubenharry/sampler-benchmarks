from sampler_evaluation.models.data.estimate_expectations import estimate_ground_truth
import sampler_evaluation

model = sampler_evaluation.models.banana()

estimate_ground_truth(model, num_samples=1000000, annealing=False)