import os
import jax
jax.config.update("jax_enable_x64", True)

#from jax.lib import xla_bridge
#print(xla_bridge.get_backend().platform)
#print(jax.extend.backend.get_backend)

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()
print(num_cores)


import os, sys
sys.path.append('../blackjax/')
sys.path.append('sampler-evaluation/')
sys.path.append('sampler-comparison/')
sys.path.append('../probability/spinoffs/inference_gym')
sys.path.append('../probability/spinoffs/fun_mc')

#from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
# from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.item_response import item_response
from sampler_evaluation.models.german_credit import german_credit

from sampler_comparison.samplers.parallel.microcanonicalmontecarlo.laps import parallel_microcanonical
from sampler_comparison.samplers.parallel.hamiltonianmontecarlo.meads import meads_with_adam
from sampler_comparison.samplers.parallel.microcanonicalmontecarlo.laps import plot_trace
from sampler_comparison.samplers.general import initialize_model

batch_size = 4096
mesh = jax.sharding.Mesh(jax.devices()[:1], 'chains')

print('Number of devices: ', len(jax.devices()))

m = [(banana_mams_paper, 100, 50, 500),
     #(Gaussian(ndims=100, eigenvalues='Gamma', numpy_seed= rng_inference_gym_icg), 500, 500])
     (german_credit(), 500, 400, 2000),      
     (brownian_motion(), 500, 500, 2000),
     (item_response(), 500, 500, 2000), #500],
     (stochastic_volatility_mams_paper, 800, 1500, 5000)] # change to 3000 for M dependence plot
    

model, n1, n2, n_meads = m[0]
# samples, info, settings_info = parallel_microcanonical(num_steps1= n1, 
#                                                        num_steps2= n2, 
#                                                        num_chains= batch_size, mesh= mesh, superchain_size= 1)(model=model)


meads_results = meads_with_adam(model.logdensity_fn, model.ndims, n_meads, batch_size)

print(meads_results)


#plot_trace(info, model, settings_info, 'papers/LAPS/img/trace/')


#shifter --image=jrobnik/sampling:1.0 python3 -m papers.LAPS.main