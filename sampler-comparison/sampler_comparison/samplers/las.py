import sys
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")

from sampler_comparison.samplers.general import make_log_density_fn
import blackjax
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error

import numpy as np
import jax
import jax.numpy as jnp
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper

def las(num_adjusted_steps, num_chains, diagonal_preconditioning=True, target_acceptance_rate=0.8):

    def s(model, key):

        logdensity_fn = make_log_density_fn(model)

        # def contract(e_x):
        #     bsq = jnp.square(e_x - model.sample_transformations["square"].ground_truth_mean) / (model.sample_transformations["square"].ground_truth_standard_deviation**2)
        #     return jnp.array([jnp.max(bsq), jnp.average(bsq)])
        
        # #model.sample_transformations["square"].fn(position)
        # observables_for_bias = lambda position:jnp.square(model.default_event_space_bijector(jax.flatten_util.ravel_pytree(position)[0]))

        unadjusted_position, adjusted_position, infos, num_steps_unadjusted, step_size_adaptation_state = blackjax.adaptation.las.las(
            logdensity_fn=logdensity_fn,
            key=key,
            # sample_init=model.sample_init,
            ndims=model.ndims,
            num_adjusted_steps=num_adjusted_steps,
            num_chains=num_chains,
            diagonal_preconditioning=diagonal_preconditioning,
            target_acceptance_rate=target_acceptance_rate
        )
        return unadjusted_position, adjusted_position, infos, num_steps_unadjusted

        
    return s

if __name__ == "__main__":
    # run las on banana
    # model = IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log')
    model = stochastic_volatility_mams_paper
    # num_steps1 = 1000
    num_adjusted_steps = 1000
    num_chains = 1000
    diagonal_preconditioning = True
    print("running las")
    sampler = las(num_adjusted_steps, num_chains, diagonal_preconditioning)
    unadjusted_samples, adjusted_samples, infos, num_steps_unadjusted = sampler(model, key=jax.random.key(0))
    # print(samples)
    # print(samples.shape)
    
    unadjusted_error_at_each_step = jnp.nanmedian(get_standardized_squared_error(
    np.expand_dims(unadjusted_samples,0), 
    f=model.sample_transformations["square"].fn,
    E_f=model.sample_transformations["square"].ground_truth_mean,
    Var_f=model.sample_transformations["square"].ground_truth_standard_deviation**2
    ),axis=0)

    adjusted_error_at_each_step = ((((adjusted_samples**2).mean(axis=1) - model.sample_transformations["square"].ground_truth_mean[None, :])**2)/(model.sample_transformations["square"].ground_truth_standard_deviation[None, :]**2)).mean(axis=-1)


    adjusted_error_at_each_step_single = ((((unadjusted_samples**2).mean(axis=0) - model.sample_transformations["square"].ground_truth_mean)**2)/(model.sample_transformations["square"].ground_truth_standard_deviation**2))
    print("adjusted_error_at_each_step_single", adjusted_error_at_each_step_single)

    import matplotlib.pyplot as plt
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['font.size'] = 16
    # plot scatterplot of samples
    plt.scatter(adjusted_samples[-1, :,0], adjusted_samples[-1, :,1])
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title('LAS Scatterplot')
    plt.savefig('las_scatterplot.png')
    plt.close()

    # print plot of the bias
    print("Shapes")
    # print(error_at_each_step.shape, error_at_each_step)
    print(adjusted_samples.shape)
    print(unadjusted_samples.shape)
    # print(infos.shape)
    print(unadjusted_error_at_each_step.shape)
    print(adjusted_error_at_each_step.shape)
    # line plot of error at each step

    ### COUNT GRADIENT CALLS!!!! TODO TODO
    plt.plot(np.concatenate([unadjusted_error_at_each_step, adjusted_error_at_each_step]))
    # add a vertical line at the end of the unadjusted phase
    plt.axvline(x=num_steps_unadjusted, color='black', linestyle='--')
    # plt.plot(unadjusted_error_at_each_step)
    # save in sampler-comparison/sampler_comparison/experiments/results/figures
    plt.savefig(f'las_bias_{model.name}.png')
    plt.close()
    plt.plot(adjusted_error_at_each_step)
    plt.savefig(f'las_bias_adjusted_{model.name}.png')
    plt.close()
    # plt.plot(np.concatenate([unadjusted_error_at_each_step, adjusted_error_at_each_step]))




    # unadjusted samples

    # adjusted samples