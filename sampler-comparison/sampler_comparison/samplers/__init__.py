from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import adjusted_mclmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc,
)


samplers = {
    "adjusted_microcanonical": adjusted_mclmc,
    "unadjusted_microcanonical": unadjusted_mclmc,
    "nuts": nuts,
}
