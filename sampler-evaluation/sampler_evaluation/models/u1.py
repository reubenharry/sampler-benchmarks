import jax
import jax.numpy as jnp
from sampler_evaluation.models.model import make_model
import pickle
import os
from sampler_evaluation.models.model import SampleTransformation, make_model

module_dir = os.path.dirname(os.path.abspath(__file__))


def U1(Lt, Lx, beta= 1.):

    """Args:
            lattice size = (Lt, Lx)
            beta: inverse temperature
    """
    
    name = 'U1'

    try:
        with open(
            f"{module_dir}/data/U1_Lt{Lt}_Lx{Lx}_beta{beta}"+"_expectations.pkl",
            "rb",
        ) as f:
            stats = pickle.load(f)

        e_x = stats["polyakov"]
        e_x2 = stats["polyakov^2"]
        e_x4 = jnp.nan # stats["e_x4"]
        # var_x2 = e_x4 - e_x2**2

    except:
        e_x = 0
        e_x2 = 0
        e_x4 = 0
        var_x2 = 0

    jax.debug.print("e_x {x}", x=e_x)
    jax.debug.print("e_x^2 {x}", x=e_x2)
    jax.debug.print("std {x}", x=jnp.sqrt(e_x2 - e_x**2))
    # raise Exception


    ndims = 2 * Lt*Lx
    unflatten = lambda links_flattened: links_flattened.reshape(2, Lt, Lx)
    locs = jnp.array([[i//Lx, i%Lx] for i in range(Lt*Lx)]) #the set of all possible lattice sites

    sample_init = lambda key: 2 * jnp.pi * jax.random.uniform(key, shape = (ndims, ))


    # E_x = jnp.zeros(ndims)
    # Var_x2 = 2 * jnp.square(E_x2)


    def logdensity_fn(links):
        """Equation 27 in reference [1]"""
        action_density = jnp.cos(plaquete(unflatten(links)))
        return beta * jnp.sum(action_density)

    
    def plaquete(links):
        """Computers theta_{0 1} = Arg(P_{01}(x)) on the lattice. output shape: (L, L)"""

        #       theta_0(x) +    theta_1(x + e0)          - theta_0(x+e1)          - x_1(x)
        return (links[0] + jnp.roll(links[1], -1, 0) - jnp.roll(links[0], -1, 1) - links[1])


    def polyakov_autocorr(links_flattened):
        
        links = unflatten(links_flattened)
        
        polyakov_angle = jnp.sum(links[0], axis = 0)
        polyakov = jnp.cos(polyakov_angle) + 1j * jnp.sin(polyakov_angle)
        # the result is the same as using [jnp.real(jnp.average(polyakov * jnp.roll(jnp.conjugate(polyakov), -n))) for n in range(Lx)], but it is computed faster by the fft
        return jnp.real(jnp.fft.ifft(jnp.square(jnp.abs(jnp.fft.fft(polyakov))))[1:1+Lx//2]) / Lx # fft based autocorrelation, we only store 1:1+Lx//2 (as the autocorrelation is then periodic)
    

    return make_model(
        logdensity_fn=logdensity_fn,
        ndims=ndims,
        default_event_space_bijector=lambda x:x,
        sample_transformations = {
        'polyakov':SampleTransformation(
            fn=polyakov_autocorr,
            ground_truth_mean=e_x,
            ground_truth_standard_deviation=jnp.sqrt(e_x2 - e_x**2),
        ),
        'polyakov^2':SampleTransformation(
            fn=lambda x: polyakov_autocorr(x)**2,
            ground_truth_mean=e_x2,
            ground_truth_standard_deviation=jnp.sqrt(e_x4 - e_x2**2),
        ),
        },
        exact_sample=None,
        sample_init=sample_init,
        name=f'U1_Lt{Lt}_Lx{Lx}_beta{beta}',
    )
