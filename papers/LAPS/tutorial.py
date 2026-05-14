import os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

#from jax.lib import xla_bridge
#print(xla_bridge.get_backend().platform)
#print(jax.extend.backend.get_backend)

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()
print(num_cores)

import os, sys
sys.path.append('../blackjax/')

from blackjax.adaptation.laps import laps as run_laps



def laps(logdensity_fn, ndims, 
         sample_init, rng_key, 
         num_steps1, num_steps2, 
         num_chains, mesh):

    info, grads_per_step, _acc_prob, final_state = run_laps(    
        logdensity_fn=logdensity_fn, 
        sample_init= sample_init,
        ndims= ndims, 
        num_steps1=num_steps1, 
        num_steps2=num_steps2, 
        num_chains=num_chains, 
        mesh=mesh, 
        rng_key= rng_key, 
        early_stop= False,
        diagonal_preconditioning= True, 
        steps_per_sample=15,
        r_end=0.01,
        diagnostics= False,
        superchain_size= 1
        )

    return final_state.position



ndims = 2

def logdensity_fn(x):
        mu2 = 0.03 * (x[0] ** 2 - 100)
        return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))


rng_key_sampling, rng_key_init = jax.random.split(jax.random.key(42))

# Function that takes a random seed and produces a vector of parameters. Each chain will be initialized by calling this function with a different random seed.
sample_init = lambda key: jax.random.normal(key, shape= ndims) 





# This script only works for a GPU backend.


# Initializes distributed JAX
jax.distributed.initialize()
num_devices = jax.process_count() # = 4 x number of nodes


# Creates local data (we will use different devices to do different realizations)
num_realizations = 64
assert (num_realizations % num_devices) == 0 # num_realizations should be divisible by the number of devices)

global_key = jax.random.key(42)
local_key = jax.random.split(global_key, num_devices)[jax.process_index()] # each device gets its own key
local_size = num_realizations//num_devices
local_keys = jax.random.split(local_key, local_size) # the remaining random keys 

# Put it on the global devices
mesh = jax.sharding.Mesh(jax.devices(), 'devices')
p = jax.sharding.PartitionSpec('devices')
global_keys = jax.make_array_from_single_device_arrays((num_realizations,),  jax.sharding.NamedSharding(mesh, p), [local_keys])


# Use the external setup() to determine what function(x, y, z, ..., key) do we want to evaluate for different values of the parameters x, y, z, ... and random keys.
# grid = (x, y, z, ...), where each parameter is a vector of different values. A full grid x \times y \times z ... will be computed.
grid, func, save_name = setup()

num_params = len(grid)
Grid = jnp.meshgrid(*grid, indexing = 'ij') # get the grid matrices


# we distribute over grid and the local_keys on a single gpu, using vmap (we could also do this part without ifs, using recursion, but not sure if neccessary)
# this following lines are general wrt num_params. For example for num_params = 2 they are equivalent to
# return jax.vmap(lambda key: jax.vmap(jax.vmap(func, in_specs), in_specs)(*Grid, key))
in_specs = (0, ) * num_params + (None, )    
f = func
for i in range(num_params):
    f = jax.vmap(f, in_specs)

func_vmap = jax.vmap(lambda key: f(*Grid, key))


# parallelize over different devices
parallel_execute = shard_map(func_vmap, 
                        mesh= mesh,
                        in_specs= p, 
                        out_specs= p
                        )

# execute calculation
results = parallel_execute(global_keys)

# save results
jnp.save(save_name, process_allgather(results))








num_chains = 256
num_steps1, num_steps2 = 100, 100

mesh = jax.sharding.Mesh(jax.devices()[:1], 'chains')

print('Number of devices: ', len(jax.devices()))

samples = laps(logdensity_fn, ndims, sample_init, rng_key_sampling, num_steps1, num_steps2, num_chains, mesh)


import matplotlib.pyplot as plt

x1 = jnp.linspace(-35, 35, 500)
x2 = jnp.linspace(-35, 35, 500)
X1, X2= jnp.meshgrid(x1, x2)
X = jnp.stack((X1, X2), axis=-1)
Z = jnp.exp(jax.vmap(jax.vmap(logdensity_fn))(X))

plt.plot(samples[:, 0], samples[:, 1], '.', color = 'teal')
plt.contourf(X1, X2, Z, cmap = 'Greys')
plt.axis('off')
plt.savefig("banana.png")
plt.close()
