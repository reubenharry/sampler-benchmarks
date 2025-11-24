import os
import jax
jax.config.update("jax_enable_x64", True)

#from jax.lib import xla_bridge
#print(xla_bridge.get_backend().platform)
#print(jax.extend.backend.get_backend)

#batch_size= 128
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)

import numpy as np
import os, sys
sys.path.append('../blackjax/')
sys.path.append('sampler-evaluation/')
sys.path.append('sampler-comparison/')
sys.path.append('../probability/spinoffs/inference_gym')
sys.path.append('../probability/spinoffs/fun_mc')

#from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian, rng_inference_gym_icg
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
#from sampler_evaluation.models.brownian2 import Brownian
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.item_response import item_response
from sampler_evaluation.models.german_credit import german_credit

from sampler_comparison.samplers.parallel.microcanonicalmontecarlo.laps import parallel_microcanonical, get_trace, plot_trace
from sampler_comparison.samplers.parallel.hamiltonianmontecarlo.meads import meads_with_adam
from sampler_comparison.samplers.parallel.microcanonicalmontecarlo.laps import get_n
from sampler_comparison.samplers.general import initialize_model
from papers.LAPS.grid import do_grid, mylogspace


batch_size = 4096
mesh = jax.sharding.Mesh(jax.devices()[:1], 'chains')

print('Number of devices: ', len(jax.devices()))

imodel, itask = int(sys.argv[1]), int(sys.argv[2])

m = [(banana_mams_paper, 100, 50, 500),
     (IllConditionedGaussian(ndims=100, eigenvalues='gamma', numpy_seed= rng_inference_gym_icg), 500, 500, 2000),
     (german_credit(), 500, 400, 2000),      
     #(brownian_motion(), 500, 500, 2000),
     (item_response(), 500, 500, 2000),
     (stochastic_volatility_mams_paper, 800, 1500, 5000)][imodel] # change to 3000 for M dependence plot
    

def _main(folder,
          C= 0.1,
          alpha= 1.9,
          bias_type= 3,
          steps_per_sample=15,
          acc_prob= None,
          integrator_coefficients= None,
          ):

     results = {}


     model, n1, n2, n_meads = m
     _, info, settings_info = parallel_microcanonical(num_steps1= n1, 
                                                            num_steps2= n2, 
                                                            num_chains= batch_size, mesh= mesh, superchain_size= 1,
                                                            C= C, alpha= alpha, bias_type= bias_type, steps_per_sample= steps_per_sample, acc_prob= acc_prob, integrator_coefficients = integrator_coefficients,
                                                            )(model=model)

     res = meads_with_adam(n_meads, batch_size)(model= model)
     print(res)
     
     return
     #folder = 'papers/LAPS/img/trace/'
     n = plot_trace(info, model, settings_info, folder)
     #n = get_n(info, settings_info)
     
     results[model.name] = n
     #print(results)

     del model
     del info
     del settings_info


     return results



def _main2():

     model, _, _, _ = m

     _, info, settings_info = parallel_microcanonical(
          num_chains= batch_size, mesh= mesh, superchain_size= 1,
          num_steps1= 500, num_steps2= 100, early_stop = False)(model=model)

     dir = 'papers/LAPS/img/trace/fixed_stepsize/GermanCredit_stepsize=0.75.npy'
     get_trace(info, settings_info, dir)

     del model
     del info
     del settings_info

     
grid = lambda param_grid, fixed_params= None, verbose= True, extra_word= '': do_grid(_main, param_grid, fixed_params=fixed_params, verbose= verbose, extra_word= m[0].name + extra_word)

#grid = lambda param_name, param_vals, fixed_params= None, verbose= True, extra_word= '': do_single(_main, param_name, param_vals, which, fixed_params=fixed_params, verbose= verbose, extra_word= m[0].name + extra_word)

_main('papers/LAPS/img/trace/')

#shifter --image=reubenharry/cosmo:1.0 python3 -m papers.LAPS.main 2 0


# if itask == 0:
#      grid({'C': mylogspace(0.001, 3, 10)})


# elif itask == 1:
#      grid({'steps_per_sample': np.logspace(np.log10(5), np.log10(30), 10).astype(int)})

# elif itask == 2:
#      grid({'alpha': mylogspace(1, 4., 6)})

# elif itask == 3:
#      from blackjax.mcmc.integrators import mclachlan_coefficients, omelyan_coefficients

#      grid({'acc_prob': np.linspace(0.6, 0.9, 6)}, fixed_params= {'integrator_coefficients': mclachlan_coefficients}, extra_word = 'MN2')
#      grid({'acc_prob': np.linspace(0.75, 0.98, 6)}, fixed_params= {'integrator_coefficients': omelyan_coefficients}, extra_word = 'MN4')

# elif itask == 4:
#      grid({'bias_type': [2, 3]})

# elif itask == 5:
#      grid({'': np.linspace(0.6, 0.9, 6)}, fixed_params= {'integrator_coefficients': mclachlan_coefficients}, extra_word = 'MN2')





#grid({'chains': [2**k for k in range(6, 13)]}, extra_word= str(rng_key_int))


# for mm in m:


#      model, n1, n2, n_meads = mm
#      n = n1 + n2

#      def step(state):
#           jax.vmap(model.log_)
#           return state, None

#      jax.lax.scan(step, )

#meads_results = meads_with_adam(n_meads, batch_size)(model= model)

#print(meads_results)

#print(results)









