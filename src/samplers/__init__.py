from src.samplers.hamiltonianmontecarlo.nuts import nuts
from src.samplers.microcanonicalmontecarlo.adjusted import adjusted_mclmc
from src.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc


samplers = {
    "adjusted_microcanonical": adjusted_mclmc,
    "unadjusted_microcanonical": unadjusted_mclmc,
    "nuts": nuts,
}
