from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt
import time
import time
import matplotlib.pyplot as plt
# from sampling_algorithms import da_adaptation
import sys
sys.path.append("/global/u1/r/reubenh/blackjax")
sys.path.append("/global/u1/r/reubenh/sampler-benchmarks/sampler-comparison")
import jax
# import blackjax
import numpy as np
import jax.numpy as jnp
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc,
    unadjusted_mclmc_no_tuning,
)
import blackjax
import time as time_module

def histogram_area(values, bins=50, range=None):
    counts, edges = jnp.histogram(values, bins=bins, range=range, density=False)
    widths = jnp.diff(edges)
    area = jnp.sum(counts * widths)
    return area

def mod_index(arr, i):
    return arr[i % (arr.shape[0])]

def analytic_gaussian(l, K, Minv):
    return jax.scipy.stats.norm.pdf(loc=0, scale=jnp.sqrt(K @ Minv @ K), x=l)

def analytic(lam, i, K, Minv): 

    return (1/ (2*np.sqrt(2*np.pi))) *2*(Minv[:,i]@K)*((1/(K @ Minv @ K))**(3/2))*lam*np.exp( (-(lam**2)) / (2 * K @ Minv @ K) )


def make_histograms(filename, samples, weights, K, Minv, i):

    num_bins = 100
    samples = np.array(samples)
    l = np.linspace(jnp.min(samples), jnp.max(samples), num_bins)

    gaussian = analytic_gaussian(l, K=K, Minv=Minv)
    plt.plot(l, gaussian)
    plt.hist(samples, bins=num_bins, density=True)

    plt.savefig(filename + "hist.png")
    plt.clf()

def make_M_Minv_K(P, t, U, r, beta, hbar,m):
    tau_c = t - ((beta * hbar * im) / 2)

    alpha = (m*P*beta)/(4*(jnp.abs(tau_c)**2))
    gamma = (m*P*t)/(hbar * (jnp.abs(tau_c)**2)) 

    M = (jnp.diag(2*alpha  + (beta / (4*P))*jax.vmap(jax.grad(jax.grad(U)))(r[1:-1])) ) - alpha * jnp.diag(jnp.ones(P-2),k=1) - alpha * jnp.diag(jnp.ones(P-2),k=-1)

    Minv = jnp.linalg.inv(M)

    K = gamma * (2*r[1:-1] - r[:-2] - r[2:]) - (t * jax.vmap(jax.grad(U))(r[1:-1]))/(P*hbar)

    print(K.shape, "k shape")

    return M, Minv, K, alpha, gamma, r

def xi(s, r, U, t, P, hbar, gamma):
    term1 = gamma * ((r[2:-1] - r[1:-2]).dot(s[1:] - s[:-1])  + (r[1] - r[0])*s[0] - (r[-1] - r[-2])*s[-1] )
    term2 = -(t/(P*hbar))*jnp.sum(jnp.array([U(r[i-1]+(s[i-2]/2)) - U(r[i-1]-(s[i-2]/2)) for i in range (2, P+1)]))
    return term1 + term2




def sample_s_chi(U, r, t=1, i=1, beta=1, hbar=1, m =1, rng_key=jax.random.PRNGKey(0), sequential=False, sample_init=None, num_unadjusted_steps=100, num_adjusted_steps=100, num_chains=5000, initial_ss_and_params=None):


    P = r.shape[0] - 1
    print(P, "P")

    sqnorm = lambda x: x.dot(x)

    
    M, Minv, K, alpha, gamma, r = make_M_Minv_K(P, t, U, r, beta, hbar, m)

    @jax.jit
    def logdensity_fn(s):
        term1 = (alpha / 2) * (sqnorm(s[1:] - s[:-1]) + (  (s[0]**2) + (s[-1]**2) ))
        term2 = (beta / (2*P)) * jnp.sum(jax.vmap(U)(r[1:-1] + s/2) + jax.vmap(U)(r[1:-1] - s/2))
        return  -(term1 + term2)
    


    
    def transform(state, info):
        x = state.position
        return (xi(x,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma
                   
                   ),x[i])
        
    
    init_key, run_key = jax.random.split(rng_key)
    
   
    from collections import namedtuple
    Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector"])
    
    model = Model(
        ndims=(P-1),
        log_density_fn=logdensity_fn,
        default_event_space_bijector=lambda x: x,
    )



    if initial_ss_and_params is not None:

            raw_samples, metadata = jax.vmap(lambda key, pos: 
                unadjusted_mclmc_no_tuning(
                    return_samples=True, 
                    # return_only_final=True, 
                    integrator_type="mclachlan",
                    L=initial_ss_and_params['params']['L'], 
                    step_size=initial_ss_and_params['params']['step_size'], 
                    initial_state=blackjax.mclmc.init(
                        position=pos,
                        logdensity_fn=logdensity_fn,
                        random_generator_arg=jax.random.split(key)[0],
                    ), 
                    inverse_mass_matrix=initial_ss_and_params['params']['inverse_mass_matrix'],
                )(
                model=model, 
                num_steps=num_unadjusted_steps,
                initial_position=pos, 
                key=key))(jax.random.split(init_key, num_chains), initial_ss_and_params['ss'])

            # raw_samples = raw_samples.reshape(num_chains*num_unadjusted_steps, P-1)
            samples, weights = (jax.vmap(lambda x : (xi(x,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma), x[i]))(raw_samples.reshape(num_chains*num_unadjusted_steps, P-1)))
            return samples, weights, raw_samples, K, Minv

    else:
        raw_samples, metadata = jax.vmap(lambda key: unadjusted_mclmc(
            return_samples=True, 
            return_only_final=False,
            num_tuning_steps=1000,
            )(
                model=model, 
                num_steps=num_unadjusted_steps,
                
                initial_position=jax.random.normal(init_key, (P-1,)), 
                key=key))(jax.random.split(init_key, num_chains))
        
        L = metadata['L'].mean()
        step_size = metadata['step_size'].mean()
        inverse_mass_matrix = metadata['inverse_mass_matrix'].mean(axis=0)

        samples = raw_samples

        return samples, L, step_size, inverse_mass_matrix

 

def bin_centers(bin_edges):
  return (bin_edges[1:] + bin_edges[:-1]) / 2.0


def get_harmonic_density(pos, inv_temp, mass, omega):
  normalization = jnp.sqrt((mass * omega) / (4 * jnp.pi * jnp.tanh(inv_temp * omega / 2)))
  exp_constant = jnp.pi * normalization ** 2
  return normalization * jnp.exp(-exp_constant * pos ** 2)


def get_sigma_vals(inv_kt, bead_omega, mass, j, sigma_sqrds):
  # Only populate the first j entries, mirroring the original behavior
  k = jnp.arange(1, j + 1)
  vals = k / (inv_kt * (k + 1) * mass * bead_omega * bead_omega)
  sigma_sqrds = sigma_sqrds.at[:j].set(vals)
  return sigma_sqrds


def partial_chain_u_to_x(prims, num_beads, rand_bead, j_this, max_j, new_chain):
  # Iterate k = max_j, ..., 1 and only update when k <= j_this
  def body_fun(k, prims_acc):
    idx = rand_bead + k
    updated = new_chain[idx] + (k * prims_acc[idx + 1] / (k + 1)) + (prims_acc[rand_bead] / (k + 1))
    prims_candidate = prims_acc.at[idx].set(updated)
    do_update = k <= j_this
    return jnp.where(do_update, prims_candidate, prims_acc)

  # run k from max_j down to 1
  def rev_body(kk, acc):
    k = max_j + 1 - kk
    return body_fun(k, acc)

  prims = lax.fori_loop(1, max_j + 1, rev_body, prims)
  return prims


def do_intermediate_staging(rng, chain_pos, left_wall, sigma_sqrds, j_this, big_p, max_j):
  # Use fixed-size normals of length max_j; only first j_this are used
  rng, sub = jax.random.split(rng)
  stds_full = jnp.sqrt(sigma_sqrds[:max_j])
  normals_full = jax.random.normal(sub, shape=(max_j,)) * stds_full

  staged = jnp.zeros(big_p + 1)

  def write_body(k, staged_acc):
    do_write = k < j_this
    idx = left_wall + k + 1
    staged_candidate = staged_acc.at[idx].set(normals_full[k])
    return jnp.where(do_write, staged_candidate, staged_acc)

  staged = lax.fori_loop(0, max_j, write_body, staged)
  chain_pos = partial_chain_u_to_x(chain_pos, big_p, left_wall, j_this, max_j, staged)
  return rng, chain_pos


def endpoint_sampling(rng, chain_pos, beadval, real_mass, tau_c, inv_kt, big_p):
  # std = jnp.sqrt(inv_kt / (real_mass * big_p))
  # std = jnp.sqrt(jnp.abs(tau_c)**2 / (real_mass * big_p))
  std = jnp.sqrt(jnp.abs(tau_c)**2 / (real_mass * big_p * inv_kt))
  rng, sub0 = jax.random.split(rng)
  rng, sub1 = jax.random.split(rng)
  mean0 = chain_pos[1]
  mean1 = chain_pos[big_p - 1]
  sample0 = mean0 + std * jax.random.normal(sub0)
  sample1 = mean1 + std * jax.random.normal(sub1)

  def write_first(_):
    return chain_pos.at[0].set(sample0)

  def write_last(_):
    return chain_pos.at[big_p].set(sample1)

  chain_pos = lax.cond(beadval == 0, write_first, write_last, operand=None)
  return rng, chain_pos


def make_flat_step(pbeads_r, jval_r, real_mass, beta, beads_sigma_sqrds, 
                   lft_wall_choices, numchoices, alt_jval_r, t, tau_c, accept_move_fn, L, step_size, inverse_mass_matrix):
  
  
  def r_step(rng, chain_r, state_in_sweep):
    
    # rng, chain_r, ss, old_pot = carry
    phase, wall_index = state_in_sweep
    
    # Phase 0: Segment updates
    def do_segment_update():
      leftwall = lft_wall_choices[wall_index]
      j_this = jnp.where(wall_index == (numchoices - 1), alt_jval_r, jval_r)
      # jax.debug.print("0 {x}", x=(leftwall, wall_index, j_this))
      rng_new, chain_prop = do_intermediate_staging(rng, chain_r, leftwall, beads_sigma_sqrds, j_this, pbeads_r, jval_r)
      # _, chain_r_new, K_new, Minv_new, new_pot, acc_prob_new = accept_move_fn(rng_new, chain_prop, chain_r, ss, old_pot)
      
      # Check if we've completed all segment updates
      next_wall = wall_index + 1
      next_phase = jnp.where(next_wall >= (numchoices), 1, 0)  # Move to single-bead phase when done
      next_wall = jnp.where(next_wall >= (numchoices ), 0, next_wall)  # Reset wall index for next phase
      
      return (chain_prop), (next_phase, next_wall)
    
    # Phase 1: Single-bead updates (excluding first choice)
    def do_single_bead_update():
      leftwall = lft_wall_choices[wall_index+1] - 1
      # jax.debug.print("1 {x}", x=(leftwall, wall_index))
      rng_new, chain_prop = do_intermediate_staging(rng, chain_r, leftwall, beads_sigma_sqrds, jnp.array(1), pbeads_r, jval_r)
      # _, chain_r_new, K_new, Minv_new, new_pot, acc_prob_new = accept_move_fn(rng_new, chain_prop, chain_r, ss, old_pot)
      
      # Check if we've completed all single-bead updates
      next_wall = wall_index + 1
      next_phase = jnp.where(next_wall >= (numchoices - 1 ), 2, 1)  # Move to endpoint phase when done
      next_wall = jnp.where(next_wall >= (numchoices - 1 ), 0, next_wall)  # Reset wall index for next phase
      
      return (chain_prop), (next_phase, next_wall)
    
    # Phase 2: Endpoint updates
    def do_endpoint_update():
      rand_int = wall_index * pbeads_r
      rng_new, chain_prop = endpoint_sampling(rng, chain_r, rand_int, real_mass, tau_c, beta, pbeads_r)
      # _, chain_r_new, K_new, Minv_new, new_pot, acc_prob_new = accept_move_fn(rng_new, chain_prop, chain_r, ss, old_pot)

      # jax.debug.print("2 {x}", x=(rand_int, wall_index))
      # Check if we've completed all endpoint updates
      next_wall = wall_index + 1
      next_phase = jnp.where(next_wall >= 2, 0, 2)  # Reset to segment phase when done
      next_wall = jnp.where(next_wall >= 2, 0, next_wall)  # Reset wall index for next phase
      
      return (chain_prop), (next_phase, next_wall)
    
    # Use lax.switch to select the appropriate phase function
    updated_carry, updated_state = lax.switch(
      phase,
      [do_segment_update, do_single_bead_update, do_endpoint_update]
    )
    
    # jax.debug.print("updated_state {x}", x=updated_state)
    return updated_carry, updated_state
  

  def ss_step(ss, r, rng_key):

    ss_and_params = {'params': {'L': L, 'step_size': step_size, 'inverse_mass_matrix': inverse_mass_matrix}, 'ss': ss[:, -1, :]}
    _, _, new_ss, K, Minv = sample_s_chi(
        t=t,
        i=i,
        beta=beta,
        hbar=hbar,
        m=m,
        rng_key=rng_key,
        U = U,
        r=r,
        sequential=False,
        initial_ss_and_params=ss_and_params,
        num_unadjusted_steps=num_unadjusted_steps, # md = unadjusted. This could probably be fewer.
        num_adjusted_steps=0, # mc = adjusted. 
        num_chains=num_chains # this should be at large as possible while still fitting in memory.
        )

    return new_ss

  def flat_step(carry, rng):

    r_key, ss_key, accept_key = jax.random.split(rng, 3)

    chain_r, ss, old_pot, state_in_sweep = carry
    (proposed_chain_r), new_state_in_sweep = r_step(r_key, chain_r, state_in_sweep)

    proposed_ss = ss_step(ss, proposed_chain_r, ss_key)
    _, chain_r_new, new_ss, K_new, Minv_new, new_pot, acc_prob_new = accept_move_fn(
      rng=accept_key, chain_new=proposed_chain_r, chain_current=chain_r, new_ss=proposed_ss, old_ss=ss, old_pot=old_pot)
    # accept or reject

    return (chain_r_new, new_ss, new_pot, new_state_in_sweep), chain_r_new
    

  return flat_step


def do_mc_open_chain(rng, mc_steps, mc_equilibrate, chain_r, pbeads_r, jval_r, real_mass, num_unadjusted_steps, num_chains, t):
  tau_c = t - ((beta * hbar * im) / 2)
  omega_p = jnp.sqrt(pbeads_r) / (jnp.abs(tau_c))

  
  def get_Utilde(ss,r, beta):

    # return None
  
    M, Minv, K, alpha, gamma, r = make_M_Minv_K(P, t, U, r, beta, hbar,m)
    raw_samples = ss.reshape(num_chains*(num_unadjusted_steps - burn_in), ss.shape[2])
    chi_samples, weights = (jax.vmap(lambda s : (xi(s,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma), s[i]))(raw_samples))
    variance = jnp.mean(chi_samples**2)
    area = histogram_area(chi_samples)
    Utilde = variance / (2 *beta)
    return Utilde - jnp.log(area) / (beta)
  

  beads_sigma_sqrds = get_sigma_vals(beta, omega_p, real_mass, jval_r, jnp.zeros(jval_r))

  def pot_energy(r, Utilde):
    
    potential_term = (1 / (2 * P)) * (U(r[0]) + U(r[P])) 
    return potential_term + Utilde

  # todo: use masses?
  initial_ss, L, step_size, inverse_mass_matrix = sample_s_chi(
        t=t,
        i=i,
        beta=beta,
        hbar=hbar,
        m=m,
        U = U,
        r=chain_r,
        sequential=False,
        initial_ss_and_params=None,
        sample_init=None,
        num_unadjusted_steps=num_unadjusted_steps, # md = unadjusted. This could probably be fewer.
        num_adjusted_steps=0, # mc = adjusted. 
        num_chains=num_chains # this should be at large as possible while still fitting in memory.
        )
  
  Utilde = get_Utilde(initial_ss[:, burn_in:, :], chain_r, beta)
  old_pot_init = pot_energy(chain_r, Utilde)

  def accept_move(rng, chain_new, chain_current, new_ss, old_ss, old_pot, beta, pot_energy_fn):
    
    rng, sub = jax.random.split(rng)
    Utilde = get_Utilde(new_ss[:, burn_in:, :], chain_new, beta)
    new_pot = pot_energy_fn(chain_new, Utilde)
    # new_pot = pot_energy_fn(chain_new)
    exp_pot = jnp.exp(-beta * (new_pot - old_pot))
    rng, sub = jax.random.split(rng)
    randval = jax.random.uniform(sub)
    accept = randval <= jnp.minimum(1.0, exp_pot)
    updated_r = jnp.where(accept, chain_new, chain_current)
    updated_pot = jnp.where(accept, new_pot, old_pot)

    updated_ss = jnp.where(accept, new_ss, old_ss)

    new_M, new_Minv, new_K, new_alpha, new_gamma, new_r = make_M_Minv_K(P, t, U, updated_r, beta, hbar,m)

    return rng, updated_r, updated_ss, new_K, new_Minv, updated_pot, jnp.minimum(1.0, exp_pot)


  # Precompute integer choices
  numchoices = int(np.floor((pbeads_r - 1) / (jval_r + 1))) + 1
  print(f'numchoices {numchoices}')
  # raise Exception("Stop here")
  lft_wall_choices = jnp.arange(numchoices) * (jval_r + 1) + 1
  alt_jval_r = pbeads_r - lft_wall_choices[-1]
  lft_wall_choices = lft_wall_choices - 1  # zero-based

  
  accept_fn = partial(accept_move, beta=beta, pot_energy_fn=pot_energy)
 

  flat_step_fn = make_flat_step(pbeads_r, jval_r, real_mass, beta, beads_sigma_sqrds, 
                      lft_wall_choices, numchoices, alt_jval_r, t, tau_c, accept_fn, L, step_size, inverse_mass_matrix)


  
  # Initialize with sweep state (phase=0, wall_index=0)
  init_carry = (chain_r, initial_ss, old_pot_init, (0, 0))
  tic = time_module.time()
  keys = jax.random.split(rng, mc_steps)
  (chain_r_out, ss_out, old_pot_out, final_sweep_state), all_samples = lax.scan(flat_step_fn, init_carry, keys)
  print(time_module.time() - tic, "time of flat loop")
 
  
  

  samples_np = np.asarray(all_samples[mc_equilibrate:])
  return samples_np

if __name__ == "__main__":
  # Parameters
  numsteps = 500000
  equilibration = 0
  num_chains = 10000
  num_unadjusted_steps = 1
  burn_in = 0 # inner loop burn in

  P = 16
  j_r = 1
  m = 1.0
  omega = 1.0

  hbar = 1.0
  i=1
  U = lambda x : 0.5*m*(omega**2)*(x**2)

  
  kbt = 1.0  
  beta = 1 / kbt

  # time = 1.0
  im = 0 + 1j
  

  # Initial chain from uniform(0,1)
  rng = jax.random.PRNGKey(0)
  rng, sub = jax.random.split(rng)
  # load r_chain 

  tic = time.time()
  for time in [1.0]:
    if time < 4.0:
      P = 8
    elif time < 8.0:
      P = 16
    else:
      P = 32


  # for time in [3.0]:
    r_chain = jax.random.uniform(sub, shape=(P + 1,))
    # r_chains = np.load(f'/global/homes/r/reubenh/sampler-benchmarks/sampler-comparison/samples_np_{time}.npy')
    # r_chain= jnp.array(r_chains[-1, :])
    # print(r_chain.shape, "r_chain shape")
    samples_np = do_mc_open_chain(rng, numsteps, equilibration, r_chain, P, j_r, m, num_unadjusted_steps, num_chains, t=time)
    print(samples_np.shape)
    # save samples  
    np.save(f'samples_np_flat_{time}_{j_r}.npy', samples_np)
    # plt.plot(std_errs)
    # plt.savefig(f'std_errs_{time}_{j_r}.png')
    # plt.clf()
    # get running avg of acc_probs
    # running_avg_acc_probs = np.cumsum(acc_probs) / np.arange(1, len(acc_probs) + 1)
    # plt.plot(running_avg_acc_probs)
    # # plt.savefig(f'running_avg_acc_probs_{time}.png')
    # plt.savefig(f'acc_probs_{time}_{j_r}.png')
    # # plt.clf()


  # observable = lambda x : x[0] * x[-1]
  # observables = jax.vmap(observable)(samples_np)
  # plt.hist(observables, bins=50)
  # plt.savefig('the_density.png')
  # plt.clf()

  # histvals, hist_bin_edges = np.histogram(samples_np, bins=50)
  # # make histogram
  # plt.plot(hist_bin_edges[:-1], histvals, label='Monte Carlo Samples')
  # plt.legend(loc='upper right')
  # plt.xlabel(r'$u_{P+1}$')
  # plt.ylabel(r'$N(u_{P+1})$')
  # plt.title('Estimator for harmonic oscillator density matrix (JAX)')
  # plt.savefig('the_density.png')
  # plt.clf()


  # print(time.time() - tic, "time")
  # # Normalize histogram
  # sum1 = 0.0
  # for s in range(len(dist_hist)):
  #   sum1 += dist_hist[s] * (dist_bin_edges[1] - dist_bin_edges[0])
  # dist_hist = dist_hist / sum1

  # # Plot
  # ideal_x = np.arange(-7, 7, 0.05)
  # ideal_prediction_x = np.asarray(get_harmonic_density(jnp.array(ideal_x), 1 / kbt, m, w))
  # # plt.plot(ideal_x, ideal_prediction_x, linestyle='-', label='Exact', color='blue')
  # plt.plot(bin_centers(dist_bin_edges), dist_hist, 'o', label=r'$N(u_{P+1})$', color='red')
  # plt.legend(loc='upper right')
  # plt.xlabel(r'$u_{P+1}$')
  # plt.ylabel(r'$N(u_{P+1})$')
  # plt.title('Estimator for harmonic oscillator density matrix (JAX)')
  # plt.savefig('the_density.png')
  # plt.clf()


