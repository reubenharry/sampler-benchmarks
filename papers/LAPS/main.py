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
# sys.path.append(current_path + '/src/inference-gym/')

#from sampler_evaluation.models.german_credit import german_credit
#from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
# from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_comparison.samplers.parallel.microcanonicalmontecarlo.laps import parallel_microcanonical
from sampler_comparison.samplers.parallel.microcanonicalmontecarlo.laps import plot_trace
# from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_comparison.samplers.general import initialize_model
# from sampler_evaluation.models.banana import banana

batch_size = 4096
mesh = jax.sharding.Mesh(jax.devices()[:1], 'chains')

print('Number of devices: ', len(jax.devices()))

model = stochastic_volatility_mams_paper # IllConditionedGaussian(ndims=2, condition_number=1)

samples, info, settings_info = parallel_microcanonical(num_steps1= 800, 
                                                       num_steps2= 1500, 
                                                       num_chains= batch_size, mesh= mesh, superchain_size= 64)(model=model)

plot_trace(info, model, settings_info, 'papers/LAPS/img/trace/')


#shifter --image=reubenharry/cosmo:1.0 python3 -m papers.LAPS.main