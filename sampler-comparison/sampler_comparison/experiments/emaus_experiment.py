import os
import jax
jax.config.update("jax_enable_x64", True)

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
#print(jax.extend.backend.get_backend)

batch_size = 256
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
#num_cores = jax.local_device_count()

import os, sys
current_path = os.getcwd()
sys.path.append(current_path + '/../../blackjax/')
sys.path.append(current_path + '/../sampler-evaluation/')



from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_comparison.samplers.parallel.microcanonicalmontecarlo.emaus import parallel_microcanonical



mesh = jax.sharding.Mesh(jax.devices(), 'chains')
print('Number of devices: ', len(jax.devices()))

parallel_microcanonical(num_steps1=100, num_steps2=400, num_chains=batch_size, mesh=mesh)(
                model=banana_mams_paper, num_steps=None, initial_position=None, key=jax.random.key(0)
                )

