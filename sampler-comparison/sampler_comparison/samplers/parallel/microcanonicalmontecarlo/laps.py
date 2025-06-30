import jax
# jax.config.update("jax_traceback_filtering", "off")
import jax.numpy as jnp
import jax.scipy.stats as stats
jax.config.update("jax_enable_x64", True)

import blackjax.mcmc.random_walk
from blackjax.adaptation.ensemble_mclmc import laps
from sampler_comparison.samplers.general import make_log_density_fn
from sampler_evaluation.evaluation.ess import samples_to_low_error

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['font.size'] = 16
            

def plot_trace(info, model, settings_info, dir):
            
    info1 = info['phase_1']
    info2 = info['phase_2'][0]


    n1 = info["phase_1"]["steps_done"] # info1['step_size'].shape[0]
    
    steps1 = jnp.arange(1, n1+1)
    steps2 = jnp.cumsum(info2['steps_per_sample']) * settings_info['num_grads_per_proposal'] + n1
    #steps2 = jnp.cumsum(info['phase_2'][0]['steps_per_sample']) * grads_per_step + n1

    steps = np.concatenate((steps1, steps2))
    ntotal = steps[-1]
    bias0 = np.concatenate((info1['bias0'][:n1], info2['bias'][:, 0]))
    bias1 = np.concatenate((info1['bias1'][:n1], info2['bias'][:, 1]))

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
    plt.plot(steps, bias1, color = 'tab:blue')
    plt.plot(steps, bias0, lw= 3, color = 'tab:red')
    plt.plot([], [], color='tab:gray', label= r'Second moments $b_t^2[x_i^2]$')

    # equipartition
    plt.plot(steps1, info1['equi_diag'][:n1], '.', color = 'tab:blue', alpha= 0.4)
    plt.plot([], [], '.', color= 'tab:gray', alpha= 0.4, label = r'Equipartition $B_t^2$')
    #plt.plot(steps1, info1['equi_full'], '.', color = 'tab:green', alpha= 0.4, label = 'full rank equipartition')
    #plt.plot(steps2, info2['equi_diag'], '.', color = 'tab:blue', alpha= 0.3)
    #plt.plot(steps2, info2['equi_full'], '.-', color = 'tab:green', alpha= 0.3)
    
    
    # relative fluctuations
    plt.plot(steps1, info1['r_avg'][:n1], '--', color = 'tab:blue')
    plt.plot(steps1, info1['r_max'][:n1], '--', color = 'tab:red')
    plt.plot([], [], '--', color = 'tab:gray',label = r'Fluctuations $\delta_t^2[x_i^2]$')

    plt.plot(steps1, info1['R_avg'][:n1], ':', color = 'tab:blue')
    plt.plot(steps1, info1['R_max'][:n1], ':', color = 'tab:red')
    plt.plot(steps2, info2['R_avg'], ':', color = 'tab:blue')
    plt.plot(steps2, info2['R_max'], ':', color = 'tab:red')
    plt.plot([], [], '--', color = 'tab:gray',label = 'split R')

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
    plt.plot(steps1, info1['EEVPD'][:n1], '.', color='tab:orange')
    plt.plot([], [], '-', color= 'tab:orange', label= 'observed')
    plt.plot(steps1, info1['EEVPD_wanted'][:n1], '-', color='black', alpha = 0.5, label = 'targeted')

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
    plt.plot(steps1, info1['step_size'][:n1], '.', color='tab:orange')
    plt.plot(steps2, info2['step_size'], '.', color='tab:orange')
    plt.ylabel(r"step size")
    #plt.yscale('log')
    end_stage1()
    
    ### L tuning ###
    plt.subplot(3, 2, 6)
    # L0 = jnp.sqrt(jnp.sum(model.E_x2))
    plt.plot(steps1, info1['L'][:n1], '.', color='tab:green')
    plt.plot(steps2, info2['L'], '.', color='tab:green')
    #plt.plot([0, ntotal], L0 * jnp.ones(2), '-', color='black')
    end_stage1()
    plt.ylabel("L")
    #plt.yscale('log')
    plt.xlabel('# gradient evaluations')
    
    plt.tight_layout()
    plt.savefig(dir + model.name + '.png')
    plt.close()




def parallel_microcanonical(num_steps1, num_steps2, num_chains, mesh, early_stop=True,
                            diagonal_preconditioning=True, superchain_size= 1,
                            ):

    def s(model):

        logdensity_fn = make_log_density_fn(model)

        def contract(e_x):
            bsq = jnp.square(e_x - model.sample_transformations["square"].ground_truth_mean) / (model.sample_transformations["square"].ground_truth_standard_deviation**2)
            return jnp.array([jnp.max(bsq), jnp.average(bsq)])
        
        #model.sample_transformations["square"].fn(position)
        observables_for_bias = lambda position:jnp.square(model.default_event_space_bijector(jax.flatten_util.ravel_pytree(position)[0]))

        info, grads_per_step, _acc_prob, final_state = laps(
    
            logdensity_fn=logdensity_fn, 
            sample_init= model.sample_init,
            ndims=model.ndims, 
            num_steps1=num_steps1, 
            num_steps2=num_steps2, 
            num_chains=num_chains, 
            mesh=mesh, 
            rng_key=jax.random.key(0), 
            early_stop=early_stop,
            diagonal_preconditioning=diagonal_preconditioning, 
            integrator_coefficients= None, 
            steps_per_sample=15,
            ensemble_observables= lambda x: x,
            observables_for_bias=observables_for_bias,
            contract = contract,
            r_end=0.01,
            diagnostics= True,
            superchain_size= superchain_size
            ) 

        settings_info = {'acc_rate': _acc_prob, 'num_grads_per_proposal': grads_per_step, 'num_chains': num_chains}
        return final_state.position, info, settings_info
    
    return s





