from sampler_evaluation.models.data.estimate_expectations import estimate_ground_truth

from sampler_evaluation.models.item_response import item_response


model = item_response()
# raise Exception

estimate_ground_truth(model, num_samples=400000, annealing=False)