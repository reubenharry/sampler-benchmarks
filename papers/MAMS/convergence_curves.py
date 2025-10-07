
import os, sys
sys.path.append('../blackjax/')
sys.path.append('sampler-evaluation/')
sys.path.append('sampler-comparison/')
sys.path.append('../probability/spinoffs/inference_gym')
sys.path.append('../probability/spinoffs/fun_mc')

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import numpy as np
#import sampler_evaluation
#from sampler_comparison.samplers import samplers

num_cores = jax.local_device_count()
print(num_cores)
# import inference_gym.using_jax as gym


from functools import partial
from sampler_evaluation.models.banana import banana
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
#from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.item_response import item_response
from sampler_evaluation.models.german_credit import german_credit
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import adjusted_mclmc



def get_biases(model, niter):
    def sampler(key):
        init_key, sample_key = jax.random.split(key)
        x0 = jax.random.normal(init_key, shape= (model.ndims, ))
        nuts_output, _ = nuts(num_tuning_steps=500, integrator_type='velocity_verlet',diagonal_preconditioning=True, target_acc_rate=0.8)(
                model=model, 
                num_steps=niter,
                initial_position= x0, 
                key=sample_key)
                
        mams_output, _ = adjusted_mclmc(L_proposal_factor=jnp.inf, random_trajectory_length=False, alba_factor=0.23, target_acc_rate=0.9, num_tuning_steps=500,diagonal_preconditioning=True, integrator_type='velocity_verlet')(
                model=model, 
                num_steps=niter,
                initial_position= x0, 
                key=sample_key)
        return nuts_output, mams_output
    return sampler

#model=IllConditionedGaussian(2,1)
models= [(brownian_motion(), 5000),
         (german_credit(), 5000),
         (item_response(), 5000),
         (stochastic_volatility_mams_paper, 5000)   
]
    



for model in models:
    print(model[0].name)
    bias_nuts, bias_mams = jax.pmap(get_biases(*model))(jax.random.split(jax.random.key(0), 128))
    np.savez('papers/MAMS/img/' + model[0].name + '.npz', 
             nuts= bias_nuts['square']['max'].mean(axis=0), 
             mams= bias_mams['square']['max'].mean(axis=0), 
             nuts_avg= bias_nuts['square']['avg'].mean(axis=0), 
             mams_avg= bias_mams['square']['avg'].mean(axis=0), 
             )

