import functools
import itertools

import chex
import jax
# jax.config.update("jax_traceback_filtering", "off")
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import optax
from absl.testing import absltest, parameterized

import blackjax
import blackjax.diagnostics as diagnostics
import blackjax.mcmc.random_walk
from blackjax.adaptation.base import get_filter_adapt_info_fn, return_all_adapt_info
from blackjax.adaptation.ensemble_mclmc import emaus
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from blackjax.mcmc.integrators import isokinetic_mclachlan
from blackjax.util import run_inference_algorithm

from sampler_comparison.samplers.general import (
    make_log_density_fn,
)
from sampler_evaluation.evaluation.ess import samples_to_low_error

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
from blackjax.adaptation.ensemble_mclmc import emaus

# from sampler_evaluation.models.banana_mams_paper import banana_mams_paper



# "L": info["phase_2"][0]["L"],
# "step_size": info["phase_2"][0]["step_size"],
# "num_grads_per_proposal": info["phase_2"][0]["steps_per_sample"],
# "acc_rate": info["phase_2"][0]["acc_prob"],
# "num_tuning_grads": jnp.array([jnp.nan]),
            

def plot_trace(info, model, settings_info, dir):
            
    info1 = info['phase_1']
    info2 = info['phase_2'][0]

    print(info1)

    print('-----------------')
    print(info2)

    n1 = info["phase_1"]["steps_done"] # info1['step_size'].shape[0]
    
    steps1 = jnp.arange(1, n1+1)
    steps2 = jnp.cumsum(info2['steps_per_sample']) * settings_info['num_grads_per_proposal'] + n1
    #steps2 = jnp.cumsum(info['phase_2'][0]['steps_per_sample']) * grads_per_step + n1

    steps = np.concatenate((steps1, steps2))
    ntotal = steps[-1]

    bias = np.concatenate((info1['bias'], info2['bias']))
    #n = [find_crossing(steps, bias[:, i], 0.01) for i in range(2)]
        
    #steps_to_low_error = [jnp.ceil(samples_to_low_error(bias[:,i], low_error=0.01)).astype(int) for i in range(2)]


    plt.figure(figsize= (15, 5))

    def end_stage1():
        ax = plt.gca()
        ylim = ax.get_ylim()
        lw = ax.spines['bottom'].get_linewidth()
        color = ax.spines['bottom'].get_edgecolor()
        plt.plot((n1+1) * np.ones(2), ylim, color= color, lw= lw)
        plt.ylim(*ylim)
    
    ### bias ###
    plt.subplot(1, 2, 1)
    #plt.title('Bias')
    plt.plot([], [], '-', color= 'tab:red', label= 'max')
    plt.plot([], [], '-', color= 'tab:blue', label= 'average')
    
    # true
    plt.plot(steps, bias[:, 1], color = 'tab:blue')
    plt.plot(steps, bias[:, 0], lw= 3, color = 'tab:red')
    plt.plot([], [], color='tab:gray', label= r'Second moments $b_t^2[x_i^2]$')

    # equipartition
    plt.plot(steps1, info1['equi_diag'], '.', color = 'tab:blue', alpha= 0.4)
    plt.plot([], [], '.', color= 'tab:gray', alpha= 0.4, label = r'Equipartition $B_t^2$')
    #plt.plot(steps1, info1['equi_full'], '.', color = 'tab:green', alpha= 0.4, label = 'full rank equipartition')
    #plt.plot(steps2, info2['equi_diag'], '.', color = 'tab:blue', alpha= 0.3)
    #plt.plot(steps2, info2['equi_full'], '.-', color = 'tab:green', alpha= 0.3)
    
    
    # relative fluctuations
    plt.plot(steps1, info1['r_avg'], '--', color = 'tab:blue')
    plt.plot(steps1, info1['r_max'], '--', color = 'tab:red')
    plt.plot([], [], '--', color = 'tab:gray',label = r'Fluctuations $\delta_t^2[x_i^2]$')

    # pathfinder
    #pf= pd.read_csv('ensemble/submission/pathfinder_convergence.csv', sep= '\t')
    #pf_grads_all = np.array(pd.read_csv('ensemble/submission/pathfinder_cost.csv', sep= '\t')[model.name])
    #pf_grads = np.max(pf_grads_all) # in an ensemble setting we have to wait for the slowest chain

    # pf = pf[pf['name'] == model.name]
    # pf_bavg, pf_bmax = pf[['bavg', 'bmax']].to_numpy()[0]

    # if pf_bavg < 2 * np.max([np.max(bias), np.max(info1['equi_full']), np.max(info1['equi_diag'])]): # pathfinder has not converged
        
    #     plt.plot([pf_grads, ], [pf_bavg, ], '*', color= 'tab:blue')
    #     plt.plot([pf_grads, ], [pf_bmax, ], '*', color= 'tab:red')
    #     plt.plot([], [], '*', color= 'tab:gray', label= 'Pathfinder')
    
    # if annotations:
    #     plt.text(steps1[len(steps1)//2], 7e-4, 'Unadjusted', horizontalalignment= 'center')
    #     plt.text(steps2[len(steps2)//2], 7e-4, 'Adjusted', horizontalalignment= 'center')
        
  
    plt.ylim(2e-4, 2e2)
    
    plt.plot([0, ntotal], jnp.ones(2) * 1e-2, '-', color = 'black')
    plt.legend(fontsize= 12)
    plt.ylabel(r'$\mathrm{bias}^2$')
    plt.xlabel('# gradient evaluations')
    plt.xlim(0, ntotal)
    plt.yscale('log')
    end_stage1()
    
    ### stepsize tuning ###
    
    plt.subplot(3, 2, 2)
    #plt.title('Hyperparameters')
    plt.plot(steps1, info1['EEVPD'], '.', color='tab:orange')
    plt.plot([], [], '-', color= 'tab:orange', label= 'observed')
    plt.plot(steps1, info1['EEVPD_wanted'], '-', color='black', alpha = 0.5, label = 'targeted')

    plt.legend(loc=4, fontsize=10)
    plt.ylabel("EEVPD")
    plt.yscale('log')
    
    ylim = plt.gca().get_ylim()
    end_stage1()
    
    ax = plt.gca().twinx()  # instantiate a second axes that shares the same x-axis
    ax.spines['right'].set_visible(True)
    ax.plot(steps2, info2['acc_prob'], '.', color='tab:orange')
    ax.plot([steps1[-1], steps2[-1]], settings_info['acc_rate'] * np.ones(2), '-', alpha= 0.5, color='black')    
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('acc prob')
    ax.tick_params(axis='y')
        
    plt.subplot(3, 2, 4)
    plt.plot(steps1, info1['step_size'], '.', color='tab:orange')
    plt.plot(steps2, info2['step_size'], '.', color='tab:orange')
    plt.ylabel(r"step size")
    #plt.yscale('log')
    end_stage1()
    
    ### L tuning ###
    plt.subplot(3, 2, 6)
    L0 = jnp.sqrt(jnp.sum(model.E_x2))
    plt.plot(steps1, info1['L'], '.', color='tab:green')
    plt.plot(steps2, info2['L'], '.', color='tab:green')
    #plt.plot([0, ntotal], L0 * jnp.ones(2), '-', color='black')
    end_stage1()
    plt.ylabel("L")
    #plt.yscale('log')
    plt.xlabel('# gradient evaluations')
    
    
    plt.tight_layout()
    plt.savefig(dir + model.name + '.png')
    plt.close()



def parallel_microcanonical(num_steps1, num_steps2, num_chains, mesh, 
                            diagonal_preconditioning=True,
                            ):

    def s(model):

        logdensity_fn = make_log_density_fn(model)

        def contract(e_x):
            bsq = jnp.square(e_x - model.sample_transformations["square"].ground_truth_mean) / (model.sample_transformations["square"].ground_truth_standard_deviation**2)
            return jnp.array([jnp.max(bsq), jnp.average(bsq)])
        
        observables_for_bias = lambda position: jnp.square(
            model.default_event_space_bijector(jax.flatten_util.ravel_pytree(position)[0])
        )

        toc = time.time()
        info, grads_per_step, _acc_prob, final_state = emaus(
    
            logdensity_fn=logdensity_fn, 
            # sample_init=model.exact_sample, 
            sample_init=lambda k: jax.random.normal(k, (model.ndims,)), 
            ndims=model.ndims, 
            num_steps1=num_steps1, 
            num_steps2=num_steps2, 
            num_chains=num_chains, 
            mesh=mesh, 
            rng_key=jax.random.key(0), 
            early_stop=True,
            diagonal_preconditioning=diagonal_preconditioning, 
            integrator_coefficients= None, 
            steps_per_sample=15,
            ensemble_observables= lambda x: x,
            observables_for_bias=observables_for_bias,
            contract = contract,
            r_end=0.01,
            diagnostics= True
            ) 
        tic = time.time()
        #print("Time taken: ", tic-toc)

        settings_info = {'acc_rate': _acc_prob, 'num_grads_per_proposal': grads_per_step, 'num_chains': num_chains}
        print(info)
        return final_state.position, info, settings_info
    
    return s





