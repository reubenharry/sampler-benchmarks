import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from typing import NamedTuple
import os, sys
current_path = os.getcwd()
sys.path.append(current_path + '/../../blackjax/')
sys.path.append(current_path + '/../sampler-evaluation/')

from blackjax.adaptation import ensemble_umclmc as umclmc
from sampler_comparison.samplers.general import make_log_density_fn

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

img_path = os.path.dirname(os.path.abspath(__file__)) + '/../img/laps/greedy/'


class AdaptationState(NamedTuple):
    L: float
    step_size: float


def get_loss_history(kernel, state_to_loss, initial_state, stepsize, L, keys):

    def step(state, xs):
        state_new = kernel(xs['key'], state, AdaptationState(L = xs['steps_per_trajectory'], step_size = xs['stepsize']))[0]
        return state_new, state_to_loss(state_new)

    loss_history = jax.lax.scan(step, initial_state, xs= {'key': keys, 'stepsize': stepsize, 'steps_per_trajectory': L})[1]

    return np.array(loss_history)


def const_stepsize(kernel, propagator, state_to_loss, initial_state, keys, L):

    LL = jnp.full(len(keys), L)

    def loss_const_stepsize(step_size):
        final_state = propagator(initial_state, jnp.full(len(keys), step_size), LL, keys)
        return state_to_loss(final_state)

    stepsize = jnp.logspace(-1, 1, 100)
    bsq = jax.vmap(loss_const_stepsize)(stepsize)

    # plt.plot(stepsize, bsq, '.-')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('Step size')
    # plt.ylabel('Max squared bias')
    # plt.savefig(img_path + model.name + 'const.png')
    # plt.close()

    stepsize = stepsize[np.argmin(bsq)]
    loss_history = get_loss_history(kernel, state_to_loss, initial_state, jnp.full(len(keys), stepsize), LL, keys)
    
    return loss_history, stepsize, LL


def grid_search(loss, x, y):
    X, Y = jnp.meshgrid(x, y)
    Z = jax.vmap(jax.vmap(loss))(X, Y)
    index = jnp.unravel_index(jnp.argmin(Z), Z.shape)

    plt.contourf(X, Y, Z, levels=50)
    plt.plot(X[index], Y[index], 'ro', markersize=10)
    plt.xlabel('Step size')
    plt.ylabel('Steps to decoherence')
    plt.savefig(img_path + 'grid_search.png')
    plt.close()

    return Z[index], X[index], Y[index]
        


def const_hyp(kernel, propagator, state_to_loss, initial_state, keys):


    def loss_const_stepsize(eps, L):
        final_state = propagator(initial_state, jnp.full(len(keys), eps), jnp.full(len(keys), L), keys)
        return state_to_loss(final_state)

    eps_arr = jnp.logspace(0, 1, 100)
    L_arr = jnp.logspace(0, 2, 100)

    _, eps, L = grid_search(loss_const_stepsize, eps_arr, L_arr)

    loss_history = get_loss_history(kernel, state_to_loss, initial_state, jnp.full(len(keys), eps), jnp.full(len(keys), L), keys)
    
    return loss_history, eps, L


def greedy_opt1(kernel, state_to_loss, initial_state, initial_stepsize, keys, L):

    """
    Greedily optimize the step size by minimizing the loss function.
    """
    state = initial_state
    stepsize = initial_stepsize
    stepsize_history = []
    loss_history = []

    for key in keys:

        one_step = lambda eps: kernel(key, state, AdaptationState(L = L, step_size= eps))[0]
        fact = np.log10(4.)
        eps = jnp.logspace(-fact, fact, 200) * stepsize
        loss= jax.vmap(lambda eps: state_to_loss(one_step(eps)))(eps)
        stepsize = eps[jnp.argmin(loss)]
        state = one_step(stepsize)
        stepsize_history.append(stepsize)
        loss_history.append(jnp.min(loss))

    return np.array(loss_history), np.array(stepsize_history)



def greedy_opt(kernel, state_to_loss, initial_state, initial_stepsize, initial_L, keys):

    """
    Greedily optimize the step size by minimizing the loss function.
    """
    state = initial_state
    stepsize = initial_stepsize
    L = initial_L
    stepsize_history, L_history, loss_history = [], [], []

    for key in keys:

        one_step = lambda eps, L: kernel(key, state, AdaptationState(L = L, step_size= eps))[0]
        loss = lambda eps, L: state_to_loss(one_step(eps, L))

        fact = np.log10(4.)
        eps_arr = jnp.logspace(-fact, fact, 100) * stepsize
        #L_arr = jnp.logspace(-fact, fact, 100) * L
        L_arr = jnp.logspace(0, 2, 100)

        ls, eps, L = grid_search(loss, eps_arr, L_arr)

        state = one_step(eps, L)
        stepsize_history.append(eps)
        L_history.append(L)
        loss_history.append(jnp.min(ls))

    return np.array(loss_history), np.array(stepsize_history), np.array(L_history)


def full_opt_reparam(kernel, propagator, state_to_loss, initial_state, stepsize_guess, keys, L):


    x_to_eps = lambda x: jnp.cumprod(x) * stepsize_guess[0]
    init = jnp.insert(stepsize_guess[1:] / stepsize_guess[:-1], 0, 1.)

    def loss(x):
        step_size = x_to_eps(x)
        final_state = propagator(initial_state, step_size, keys)
        return jnp.log(state_to_loss(final_state))

    opt = minimize(jax.value_and_grad(loss), jac= True, 
                   x0= init, 
                   method='L-BFGS-B', 
                   bounds= [(1./4., 4.),] * len(keys), 
                   options={'maxiter': 20}
                   )
    print(opt)

    stepsize = x_to_eps(opt.x)

    loss_history = get_loss_history(kernel, state_to_loss, initial_state, stepsize, keys, L)

    return loss_history, stepsize


def full_opt(kernel, propagator, state_to_loss, initial_state, stepsize_guess, keys, L):

    fix= 9

    x_to_eps = lambda x: jnp.concatenate((jnp.ones(fix), x)) * stepsize_guess
    init = jnp.ones(len(stepsize_guess) - fix)


    def loss(x):
        final_state = propagator(initial_state, x_to_eps(x), keys)
        return jnp.log(state_to_loss(final_state))


    opt = minimize(jax.value_and_grad(loss), jac= True, 
                   x0= init, 
                   method='L-BFGS-B', 
                   #bounds= jnp.array([stepsize_guess * 0.5, stepsize_guess * 2.]).T,
                   options={'maxiter': 20}
                   )
    print(opt)

    stepsize = x_to_eps(opt.x)

    loss_history = get_loss_history(kernel, state_to_loss, initial_state, stepsize, keys, L)

    return loss_history, stepsize



def mainn(
    model,
    num_chains,
    mesh,
    num_steps,
    rng_key
):

    logdensity_fn = make_log_density_fn(model)

    key_init, key_kernel = jax.random.split(rng_key)
    keys_kernel = jax.random.split(key_kernel, (num_steps, num_chains))

    # initialize the chains
    initial_state = umclmc.initialize(
        key_init, logdensity_fn, model.sample_init, num_chains, mesh
    )

    # burn-in with the unadjusted method
    kernel = jax.vmap(umclmc.build_kernel(logdensity_fn), (0, 0, None))

    def kernel_wrap(state, xs):
        return kernel(xs['key'], state, AdaptationState(L= xs['steps_per_trajectory'], step_size = xs['step_size']))


    def propagator(init, step_size, steps_per_trajectory, keys_kernel):
        return jax.lax.scan(kernel_wrap, init= init, 
                               xs = {'key': keys_kernel, 
                                     'step_size': step_size,
                                     'steps_per_trajectory': steps_per_trajectory
                                     }
                                )[0]

    def state_to_loss(state):
        transf = jax.vmap(lambda x: model.sample_transformations['square'].fn(model.default_event_space_bijector(jax.flatten_util.ravel_pytree(x)[0])))
        e_x = jnp.average(transf(state.position), axis = 0)

        bsq = jnp.square(e_x - model.sample_transformations["square"].ground_truth_mean) / (model.sample_transformations["square"].ground_truth_standard_deviation**2)
        
        return jnp.max(bsq)

    
    # constant step size
    loss_const, eps_const, L_const = const_hyp(kernel, propagator, state_to_loss, initial_state, keys_kernel)
    
    # greedy optimization
    loss_greedy, eps_greedy, L_greedy = greedy_opt(kernel, state_to_loss, initial_state, eps_const * 10, L_const * 5, keys_kernel)

    # proper optimization
    #loss_full, eps_full = full_opt(kernel, propagator, state_to_loss, initial_state, eps_greedy, keys_kernel, L)

    plt.rcParams.update({'font.size': 23})
    plt.figure(figsize=(10, 15))
    
    plt.subplot(3, 1, 1)
    plt.plot(loss_const, 'o-', color= 'black', label='Constant')
    plt.plot(loss_greedy, 'o-', color= 'tab:purple', label='Greedy optimization')
    plt.legend()
    plt.yscale('log')
    plt.ylabel('Max squared bias')

    plt.subplot(3, 1, 2)
    plt.plot(np.ones(num_steps) * eps_const, 'o-', color = 'black')
    plt.plot(eps_greedy, 'o-', color = 'tab:purple')
    plt.ylabel('Stepsize')

    plt.subplot(3, 1, 3)
    plt.plot(np.ones(num_steps) * L_const, 'o-', color = 'black')
    plt.plot(L_greedy, 'o-', color = 'tab:purple')

    plt.ylabel('Steps to decoherence')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(img_path + model.name + '.png')
    plt.close()



from sampler_evaluation.models.banana_mams_paper import banana_mams_paper


batch_size = 2048
mesh = jax.sharding.Mesh(jax.devices()[:1], 'chains')
key = jax.random.key(0)

model = banana_mams_paper # IllConditionedGaussian(ndims=2, condition_number=1)
num_steps = 15

mainn(model, batch_size, mesh, num_steps, key)


#shifter --image=reubenharry/cosmo:1.0 python3 -m sampler_comparison.experiments.greedy_ensemble


# Use with "autodiff" branch on blackjax