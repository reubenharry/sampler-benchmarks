import sys
sys.path.append(".")  
sys.path.append("../../blackjax")
sys.path.append("../../sampler-benchmarks/sampler-comparison")
sys.path.append("../../sampler-benchmarks/sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
# print(os.listdir("../../src/inference-gym/spinoffs/inference_gym"))


import jax

import jax.numpy as jnp
from sampler_evaluation.models.model import make_model
import pickle
import os
from sampler_evaluation.models.model import SampleTransformation, make_model
import h5py 

module_dir = os.path.dirname(os.path.abspath(__file__))

class LFT:
    def __init__(self,Lt, Lx, beta= 1.,m=-0.188):

        self.Lt = Lt 
        self.Lx = Lx

        return
        
    def unflatten(self, links_flattened):
        
        return links_flattened.reshape(2, self.Lt, self.Lx)
    
    def plaquette(self,links):
        """Computers theta_{0 1} = Arg(P_{01}(x)) on the lattice. output shape: (L, L)"""

        #       theta_0(x) +    theta_1(x + e0)          - theta_0(x+e1)          - x_1(x)
        return (links[0] + jnp.roll(links[1], -1, 0) - jnp.roll(links[0], -1, 1) - links[1])


    def polyakov_autocorr(self,links_flattened):
        
        links = unflatten(links_flattened)
        
        polyakov_angle = jnp.sum(links[0], axis = 0)
        polyakov = jnp.cos(polyakov_angle) + 1j * jnp.sin(polyakov_angle)
        # the result is the same as using [jnp.real(jnp.average(polyakov * jnp.roll(jnp.conjugate(polyakov), -n))) for n in range(Lx)], but it is computed faster by the fft
        return jnp.real(jnp.fft.ifft(jnp.square(jnp.abs(jnp.fft.fft(polyakov))))[1:1+self.Lx//2]) / self.Lx # fft based autocorrelation, we only store 1:1+Lx//2 (as the autocorrelation is then periodic)
    

    def top_charge(self,links_flattened):
        links = unflatten(links_flattened)
        x = plaquette(links)
        charge = jnp.sum(jnp.sin(x)) / (2 * jnp.pi)
        return jnp.array([charge])