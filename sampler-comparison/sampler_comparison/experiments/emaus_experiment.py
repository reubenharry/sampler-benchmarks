import os
import jax
#jax.config.update("jax_enable_x64", True)

#from jax.lib import xla_bridge
#print(xla_bridge.get_backend().platform)
#print(jax.extend.backend.get_backend)

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()
print(num_cores)


import os, sys
current_path = os.getcwd()
sys.path.append(current_path + '/../../blackjax/')
sys.path.append(current_path + '/../sampler-evaluation/')
# sys.path.append(current_path + '/src/inference-gym/')



from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
# from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
# from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_comparison.samplers.parallel.microcanonicalmontecarlo.emaus import parallel_microcanonical
# from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_comparison.samplers.general import initialize_model
# from sampler_evaluation.models.banana import banana
import time

batch_size = 4096
mesh = jax.sharding.Mesh(jax.devices()[:1], 'chains')

print('Number of devices: ', len(jax.devices()))

model = banana_mams_paper # IllConditionedGaussian(ndims=2, condition_number=1)

samples = parallel_microcanonical(num_steps1=800, num_steps2=1500, num_chains=batch_size, mesh=mesh)(
                model=model, num_steps=None, initial_position=None, key=jax.random.key(0)
                )

