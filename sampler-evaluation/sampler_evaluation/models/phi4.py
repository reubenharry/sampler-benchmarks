import jax.numpy as jnp
from sampler_evaluation.models.model import SampleTransformation, make_model
import pickle
import numpy as np
import os
module_dir = os.path.dirname(os.path.abspath(__file__))

def phi4(L,lam, load_from_file=True):

    ndims = L**2


    lam_str = str(lam)[:5]

    # try:
    if load_from_file:
        with open(
            f"{module_dir}/data/Phi4_L{L}_lam"+lam_str+"_expectations.pkl",
            "rb",
        ) as f:
            stats = pickle.load(f)

        # print(stats.keys(),stats['identity'].shape,stats['square'].shape, "stats")

        e_x = stats["identity"]
        e_x2 = stats["square"]
        # print(e_x2.shape, "e_x2")
        # raise Exception
        e_x4 = stats["quartic"]
        var_x2 = e_x4 - e_x2**2

    else:
        e_x = 0.0
        e_x2 = 0.0
        e_x4 = 0.0
        var_x2 = 0.0

    # jax.debug.print("e_x {x}", x=e_x)
    # jax.debug.print("e_x2 {x}", x=e_x2)



        

    # sample_init = lambda key: jax.random.normal(key, shape = (ndims, ))
        
    
    def logdensity_fn(x):
        """action of the theory"""
        phi = x.reshape(L, L)
        action_density = lam*jnp.power(phi, 4) - phi*(jnp.roll(phi, -1, 0) + jnp.roll(phi, 1, 0) + jnp.roll(phi, -1, 1) + jnp.roll(phi, 1, 1))
        return -jnp.sum(action_density)

    def psd(phi):
        return (jnp.square(jnp.abs(jnp.fft.fft2(phi.reshape(L, L)))) / L ** 2).reshape(ndims)
    

    return make_model(
        logdensity_fn=logdensity_fn,
        ndims=ndims,
        default_event_space_bijector=psd,
        sample_transformations={
            "identity": SampleTransformation(
                    fn=lambda x: x,
                    ground_truth_mean=e_x, ground_truth_standard_deviation=jnp.sqrt(e_x2 - e_x**2),),
            "square": SampleTransformation(
                    fn=lambda x: x**2,
                    ground_truth_mean=e_x2, ground_truth_standard_deviation=jnp.sqrt(var_x2),),
            "quartic": SampleTransformation(
                    fn=lambda x: x**4,
                    ground_truth_mean=e_x4, ground_truth_standard_deviation=jnp.nan,),
        },

        # x_ground_truth_mean=jnp.zeros((L,L)),
        # x_ground_truth_std=jnp.sqrt((jnp.zeros((L,L)))),
        # x2_ground_truth_mean=jnp.zeros((L,L)),
        # x2_ground_truth_std=jnp.sqrt((jnp.zeros((L,L)))),

        # x_ground_truth_mean=e_x,
        # x_ground_truth_std=jnp.sqrt(e_x2 - e_x**2),
        # x2_ground_truth_mean=e_x2,
        # x2_ground_truth_std=jnp.sqrt(var_x2),
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

    