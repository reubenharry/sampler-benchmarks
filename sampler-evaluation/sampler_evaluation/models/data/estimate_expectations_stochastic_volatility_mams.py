from sampler_evaluation.models.data.estimate_expectations import estimate_ground_truth

from sampler_evaluation.models.item_response import item_response


from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
# raise Exception

model = stochastic_volatility_mams_paper

estimate_ground_truth(model, num_samples=400000, annealing=False)