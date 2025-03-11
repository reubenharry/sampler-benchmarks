from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.german_credit import german_credit
from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
from sampler_evaluation.models.item_response import item_response
from sampler_evaluation.models.rosenbrock import Rosenbrock_36D
from sampler_evaluation.models.neals_funnel import neals_funnel


models = {
    "Banana": banana(),
    "Brownian_Motion": brownian_motion(),
    "German_Credit": german_credit(),
    "Rosenbrock_36D": Rosenbrock_36D(),
    "Neals_Funnel": neals_funnel(),
    "Stochastic_Volatility": stochastic_volatility(),
    "Item_Response": item_response(),
}
