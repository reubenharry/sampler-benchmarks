import sys
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")

from sampler_comparison.samplers.general import make_log_density_fn
import blackjax

import jax
import jax.numpy as jnp
from sampler_evaluation.models.banana import banana

def las(num_steps1, num_steps2, num_chains, diagonal_preconditioning=True):

    def s(model, key):

        logdensity_fn = make_log_density_fn(model)

        # def contract(e_x):
        #     bsq = jnp.square(e_x - model.sample_transformations["square"].ground_truth_mean) / (model.sample_transformations["square"].ground_truth_standard_deviation**2)
        #     return jnp.array([jnp.max(bsq), jnp.average(bsq)])
        
        # #model.sample_transformations["square"].fn(position)
        # observables_for_bias = lambda position:jnp.square(model.default_event_space_bijector(jax.flatten_util.ravel_pytree(position)[0]))

        position = blackjax.adaptation.las.las(
            logdensity_fn=logdensity_fn,
            key=key,
            # sample_init=model.sample_init,
            ndims=model.ndims,
            num_steps1=num_steps1,
            num_steps2=num_steps2,
            num_chains=num_chains,
            diagonal_preconditioning=diagonal_preconditioning
        )
        return position

        
    return s

if __name__ == "__main__":
    # run las on banana
    model = banana()
    num_steps1 = 1000
    num_steps2 = 1000
    num_chains = 100
    diagonal_preconditioning = True
    print("running las")
    sampler = las(num_steps1, num_steps2, num_chains, diagonal_preconditioning)
    samples = sampler(model, key=jax.random.key(0))
    # print(samples)
    print(samples.shape)

    import matplotlib.pyplot as plt
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['font.size'] = 16
    # plot scatterplot of samples
    plt.scatter(samples[:,0], samples[:,1])
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title('LAS Scatterplot')
    plt.savefig('las_scatterplot.png')
    plt.close()