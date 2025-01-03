from src.samplers.hamiltonianmontecarlo.nuts import nuts
from src.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc


samplers = {
    "nuts": nuts,
    "unadjusted_microcanonical": unadjusted_mclmc,
}