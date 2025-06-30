import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from typing import NamedTuple
import os, sys
current_path = os.getcwd()
sys.path.append('../blackjax/')
sys.path.append('sampler-evaluation/')
sys.path.append('sampler-comparison/')

from blackjax.adaptation import ensemble_umclmc as umclmc
from sampler_comparison.samplers.general import make_log_density_fn

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import optimize

img_path = os.path.dirname(os.path.abspath(__file__)) + '/img/greedy/'


class AdaptationState(NamedTuple):
    L: float
    step_size: float
    inverse_mass_matrix: jnp.ndarray


def _grid1(loss, x, savefig = None):
    Z = jax.vmap(loss)(x)
    index = jnp.argmin(Z)

    if savefig != None:
        plt.plot(x, Z, 'o-')
        plt.savefig(img_path +  savefig + '.png')
        plt.close()

    return Z[index], x[index]


def grid1(eps_low, eps_high):

    def func(loss): 
        x = np.logspace(eps_low, eps_high, 50)
        return _grid1(loss, x)[1]
    
    return func

def _grid2(loss, x, y, savefig = None):
    X, Y = jnp.meshgrid(x, y)
    Z = jax.vmap(jax.vmap(loss))(X, Y)
    index = jnp.unravel_index(jnp.argmin(Z), Z.shape)

    if savefig != None:
        plt.contourf(X, Y, Z, levels=50)
        plt.plot(X[index], Y[index], 'ro', markersize=10)
        plt.savefig(img_path +  savefig + '.png')
        plt.close()

    return Z[index], X[index], Y[index]
        


class Problem:

    def __init__(self,
        model,
        num_chains,
        mesh,
        rng_key,
        num_steps_pow,
        L = 100.,
        beta = 1):

        self.num_steps = 2 ** num_steps_pow
        self.L = jnp.full(self.num_steps, L)
        self.sigma = jnp.ones(model.ndims)

        logdensity_fn = make_log_density_fn(model)

        key_init, key_kernel = jax.random.split(rng_key)
        keys_kernel = jax.random.split(key_kernel, (self.num_steps, num_chains))

        # initialize the chains
        initial_state = umclmc.initialize(
            key_init, logdensity_fn, lambda k: model.sample_init(k, beta), num_chains, mesh, superchain_size=1
        )

        self.kernel = jax.vmap(umclmc.build_kernel(logdensity_fn), (0, 0, None))

        def kernel_wrap(state, xs):
            return self.kernel(xs['key'], state, AdaptationState(L= xs['steps_per_trajectory'], step_size = xs['step_size'], inverse_mass_matrix= self.sigma))


        def propagator(step_size):
            return jax.lax.scan(kernel_wrap, init= initial_state, 
                                xs = {'key': keys_kernel, 
                                        'step_size': step_size,
                                        'steps_per_trajectory': self.L
                                        }
                                    )[0]

        def state_to_loss(state):
            transf = jax.vmap(lambda x: model.sample_transformations['square'].fn(model.default_event_space_bijector(jax.flatten_util.ravel_pytree(x)[0])))
            e_x = jnp.average(transf(state.position), axis = 0)

            bsq = jnp.square(e_x - model.sample_transformations["square"].ground_truth_mean) / (model.sample_transformations["square"].ground_truth_standard_deviation**2)
            
            return jnp.max(bsq)
 

        self.propagator = propagator
        self.initial_state = initial_state
        self.keys = keys_kernel
        self.state_to_loss = state_to_loss
        self.num_steps_pow = num_steps_pow


    def get_loss_history(self, stepsize):

        def step(state, xs):
            state_new = self.kernel(xs['key'], state, AdaptationState(L = xs['steps_per_trajectory'], step_size = xs['stepsize'], inverse_mass_matrix= self.sigma))[0]
            return state_new, self.state_to_loss(state_new)

        loss_history = jax.lax.scan(step, self.initial_state, xs= {'key': self.keys, 'stepsize': stepsize, 'steps_per_trajectory': self.L})[1]

        return np.array(loss_history)



    def optimize(self, optimizer, x_to_eps):


        def loss(x):
            step_size = x_to_eps(x, self.num_steps)
            final_state = self.propagator(step_size)
            return jnp.log(self.state_to_loss(final_state))

        x = optimizer(loss)
        
        stepsize = x_to_eps(x, self.num_steps)

        loss_history = self.get_loss_history(stepsize)

        return x, loss_history, stepsize


        


#####   Optimizers   #####

def refine_annealing(problem):


    def step(x_init, maxiter, grid_search= False):

        x_to_eps = lambda x: jnp.repeat(jnp.exp(-jnp.cumsum(x)), len(problem.keys) // len(x_init))
        
        
        def loss(x):
            step_size = x_to_eps(x)
            final_state = problem.propagator(step_size)
            return jnp.log(problem.state_to_loss(final_state))

        if not grid_search:
            bounds = [(-np.inf, np.inf), ] + [(0, np.inf) for i in range(len(x_init)-1)]

            opt = optimize.minimize(jax.value_and_grad(loss), jac= True, 
                        x0= x_init, 
                        method='L-BFGS-B', 
                        bounds = bounds,
                        options={'maxiter': maxiter})
            
            opt_x = opt.x

        else:
            if len(x_init) == 1:
                _, x0 = grid_search1(loss, 
                            x[0] + jnp.linspace(jnp.log(1e-2), jnp.log(1e2), 50),
                            savefig='grid1')
                opt_x = jnp.array([x0,])
            
            elif len(x_init) == 2:
                func = lambda param0, param1: loss(jnp.array([param0, param1]))
                _, x0, x1 = grid_search2(func, x[0] + jnp.linspace(jnp.log(0.1), jnp.log(10), 30), 
                            x[1] + jnp.linspace(jnp.log(0.1), jnp.log(10), 30), 
                            savefig='grid2')
                opt_x = jnp.array([x0, x1])

            else:
                raise ValueError('Grid search only implemented for 1 or 2 parameters')
            

        stepsize = x_to_eps(opt_x)

        loss_history = problem.get_loss_history(stepsize)

        x_init = jnp.concatenate(jnp.array([opt_x, jnp.ones(len(opt_x))]).T) # initial condition for the next iteration = [x[0], 0, x[1], 0, ...]
        
        return x_init, loss_history, stepsize
    


    loss_history, stepsize = [], []
    x = np.array([0., ])


    for i in range(problem.num_steps_pow+1):
        print('Iteration ' + str(i))
        x, h, s = step(x, maxiter= 20, grid_search= i < 2)
        loss_history.append(h)
        stepsize.append(s)


    return np.array(loss_history), np.array(stepsize)


def scipy_wrap(method, x0, bounds= None, maxiter= 100):

    def func(loss):
        opt = optimize.minimize(jax.value_and_grad(loss), jac= True, 
                    x0= x0, 
                    method=method, 
                    bounds= bounds,
                    options={'maxiter': maxiter})
        
        return opt.x

    return func

monotonic_repeat = lambda x, size: jnp.repeat(jnp.exp(-jnp.cumsum(x)), size // len(x))



def plott():

    plt.rcParams.update({'font.size': 23})
    plt.figure(figsize=(10, 10))
    
    colors = matplotlib.cm.inferno(np.linspace(0, 1, len(loss_full)))

    plt.subplot(2, 1, 1)
    for i in range(len(loss_full)):
        plt.plot(loss_full[i], 'o-', color = colors[i], label='Iteration ' + str(i))

    plt.legend()
    plt.yscale('log')
    plt.ylabel('Max squared bias')

    plt.subplot(2, 1, 2)
    edges = np.arange(num_steps + 1)
    for i in range(len(loss_full)):
        plt.stairs(eps_full[i], edges, color= colors[i])
    
    plt.ylabel('Stepsize')

    # plt.subplot(3, 1, 3)
    # plt.plot(np.ones(num_steps) * L_const, 'o-', color = 'black')
    # plt.plot(L_greedy, 'o-', color = 'tab:purple')

    #plt.ylabel('Steps to decoherence')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(img_path + model.name + '_annealing.png')
    plt.close()



from sampler_evaluation.models.banana_mams_paper import banana_mams_paper


batch_size = 2048
mesh = jax.sharding.Mesh(jax.devices()[:1], 'chains')
key = jax.random.key(0)

model = banana_mams_paper # IllConditionedGaussian(ndims=2, condition_number=1)
num_steps_pow = 4
num_steps = 2 ** num_steps_pow

problem = Problem(model, batch_size, mesh, key, num_steps_pow)


# x, b, eps = problem.optimize(grid1(0, 1.5), 
#                              x_to_eps = lambda x, size: jnp.full(size, x))

bounds_unconstrained = [(0., np.inf) for i in range(num_steps)]
bounds = [(0., np.log(3.))]

basinhopping = {'algorithm': lambda loss: optimize.basinhopping(jax.value_and_grad(loss), 
                                                  x0= jnp.ones(num_steps), 
                                                  minimizer_kwargs= {'bounds': bounds_unconstrained, 
                                                                     'method': 'L-BFGS-B', 
                                                                     'jac': True}).x,
                'name': 'basinhopping'}

diff_evol = {'algorithm': lambda loss: optimize.differential_evolution(loss, bounds).x, 'name': 'differential_evolution'}

shgo = {'algorithm': lambda loss: optimize.shgo(loss, bounds).x, 'name': 'shgo'}

dual_annealing = {'algorithm': lambda loss: optimize.dual_annealing(loss, bounds).x, 'name': 'dual_annealing'}

direct = {'algorithm': lambda loss: optimize.direct(loss, bounds).x, 'name': 'direct'}


algorithms = [basinhopping, diff_evol, shgo, dual_annealing, direct]

from time import time

for alg in algorithms:

    tic = time()
    x, b, eps = problem.optimize(alg['algorithm'], monotonic_repeat)


    plt.rcParams.update({'font.size': 23})
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(b, 'o-')

    plt.yscale('log')
    plt.ylabel('Max squared bias')

    plt.subplot(2, 1, 2)
    edges = np.arange(len(b) + 1)
    plt.stairs(eps, edges)

    plt.ylabel('Stepsize')

    plt.xlabel('# gradient calls')
    plt.tight_layout()
    plt.savefig(img_path + model.name + '_' + alg['name']+ '.png')
    plt.close()


    toc = time()

    diff = (toc - tic)/ 60

    print(f"Algorithm {alg['name']} took {diff:.2f} minutes.")


#shifter --image=reubenharry/cosmo:1.0 python3 -m papers.LAPS.opt
