from sampler_evaluation.models.data.estimate_expectations import estimate_ground_truth
import sampler_evaluation

model = sampler_evaluation.models.german_credit()

estimate_ground_truth(model, num_samples=2000000, annealing=False)