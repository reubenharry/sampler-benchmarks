import jax.numpy as jnp
from sampler_evaluation.models.model import make_model
import pickle
import numpy as np
import jax
import os
module_dir = os.path.dirname(os.path.abspath(__file__))

def phi4(L,lam):

    ndims = L**2


    lam_str = str(lam)[:5]

    with open(
        f"{module_dir}/data/Phi4_L{L}_lam"+lam_str+"_expectations.pkl",
        "rb",
    ) as f:
        stats = pickle.load(f)

    e_x = stats["e_x"]
    e_x2 = stats["e_x2"]
    e_x4 = stats["e_x4"]
    var_x2 = e_x4 - e_x2**2

    jax.debug.print("e_x {x}", x=e_x)
    jax.debug.print("e_x2 {x}", x=e_x2)



        

    # sample_init = lambda key: jax.random.normal(key, shape = (ndims, ))
        
    
    def logdensity_fn(x):
        """action of the theory"""
        phi = x.reshape(L, L)
        action_density = lam*jnp.power(phi, 4) - phi*(jnp.roll(phi, -1, 0) + jnp.roll(phi, 1, 0) + jnp.roll(phi, -1, 1) + jnp.roll(phi, 1, 1))
        return -jnp.sum(action_density)

    def psd(phi):
        return jnp.square(jnp.abs(jnp.fft.fft2(phi.reshape(L, L)))) / L ** 2
    

    return make_model(
        logdensity_fn=logdensity_fn,
        ndims=ndims,
        transform=psd,

        # x_ground_truth_mean=jnp.zeros((L,L)),
        # x_ground_truth_std=jnp.sqrt((jnp.zeros((L,L)))),
        # x2_ground_truth_mean=jnp.zeros((L,L)),
        # x2_ground_truth_std=jnp.sqrt((jnp.zeros((L,L)))),

        x_ground_truth_mean=e_x,
        x_ground_truth_std=jnp.sqrt(e_x2 - e_x**2),
        x2_ground_truth_mean=e_x2,
        x2_ground_truth_std=jnp.sqrt(var_x2),
        exact_sample=None,
        name=f'Phi4_L{L}_lam'+lam_str,
    )


if __name__ == "__main__":
    print(phi4(4, 1).sample_transformations["identity"].ground_truth_mean)
    print(phi4(4, 1).sample_transformations["identity"].ground_truth_standard_deviation)

    reduced_lam = jnp.linspace(-2.5, 7.5, 16) #lambda range around the critical point (m^2 = -4 is fixed)


    def unreduce_lam(reduced_lam, side):
        """see Fig 3 in https://arxiv.org/pdf/2207.00283.pdf"""
        return 4.25 * (reduced_lam * np.power(side, -1.0) + 1.0)

    print(unreduce_lam(reduced_lam=reduced_lam,side=4))

    