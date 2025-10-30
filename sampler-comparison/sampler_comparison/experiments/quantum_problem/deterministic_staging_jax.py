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


from collections import namedtuple
Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector"])


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
    


    
    # def transform(state, info):
    #     x = state.position
    #     return (xi(x,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma
                   
    #                ),x[i])
        
    
    init_key, run_key = jax.random.split(rng_key)
    
   
    
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


def pot_energy_simple(r, ss, real_mass, omga, pbeads_r):
  """Simple potential energy calculation for harmonic oscillator."""
  first = 0.5 * 0.5 * real_mass * omga * omga * r[0] * r[0] / pbeads_r
  middle = 0.5 * real_mass * omga * omga * jnp.sum(r[1:pbeads_r] ** 2) / pbeads_r
  last = 0.5 * 0.5 * real_mass * omga * omga * r[pbeads_r] * r[pbeads_r] / pbeads_r

  new_ss = ss
  return first + middle + last, new_ss








def do_mc_open_chain(rng, mc_steps, mc_equilibrate, chain_r, pbeads_r, jval_r, real_mass, num_unadjusted_steps, num_chains, t):
  tau_c = t - ((beta * hbar * im) / 2)
  omega_p = jnp.sqrt(pbeads_r) / (jnp.abs(tau_c))

  def make_mc_step(pbeads_r, jval_r, real_mass, beta, beads_sigma_sqrds, 
              lft_wall_choices, numchoices, alt_jval_r, tau_c, accept_move_fn):
    
    def step(carry, step_idx):
      """
      Perform one Monte Carlo step including segment updates, single-bead updates, and endpoint updates.
      
      Args:
        carry: Tuple of (rng, chain_r, ss, old_pot)
        step_idx: Current step index
        pbeads_r: Number of beads
        jval_r: Maximum j value for staging
        real_mass: Mass parameter
        beta: Inverse temperature
        beads_sigma_sqrds: Precomputed sigma squared values for staging
        lft_wall_choices: Precomputed left wall positions
        numchoices: Number of segment choices
        alt_jval_r: Alternative j value for last segment
        accept_move_fn: Function to accept/reject moves
      
      Returns:
        Tuple of (updated_carry, sample_value)
      """
      rng, chain_r, ss, old_pot = carry
    #   jax.debug.print("step_idx {x}", x=step_idx)

      # Segment updates over all left walls
      def seg_body(i, inner_carry):
        # jax.debug.print("0 {x}", x=i)
        rng, chain_r, ss, _, _, old_pot, acc_prob = inner_carry
        leftwall = lft_wall_choices[i]
        j_this = jnp.where(i == (numchoices - 1), alt_jval_r, jval_r)
        # jax.debug.print("0 {x}", x=(leftwall, i, j_this))
        rng, chain_prop = do_intermediate_staging(rng, chain_r, leftwall, beads_sigma_sqrds, j_this, pbeads_r, jval_r)
        rng, chain_r_new, ss_new, K, Minv, new_pot, acc_prob = accept_move_fn(rng, chain_prop, chain_r, ss, old_pot)
        return (rng, chain_r_new, ss_new, K, Minv, new_pot, acc_prob)

      rng, chain_r, ss, K, Minv, old_pot, acc_prob = lax.fori_loop(0, numchoices, seg_body, (rng, chain_r, ss, jnp.zeros(P-1), jnp.zeros((P-1,P-1)), old_pot, 0))

      # Single-bead updates (excluding first choice)
      def single_body(i, inner_carry):
        rng, chain_r, ss, _, _, old_pot, acc_prob = inner_carry
        leftwall = lft_wall_choices[i] - 1
        # jax.debug.print("1 {x}", x=(leftwall, i))
        rng, chain_prop = do_intermediate_staging(rng, chain_r, leftwall, beads_sigma_sqrds, jnp.array(1), pbeads_r, jval_r)
        rng, chain_r_new, ss_new, K, Minv, old_pot_new, acc_prob = accept_move_fn(rng, chain_prop, chain_r, ss, old_pot)
        return (rng, chain_r_new, ss_new, K, Minv, old_pot_new, acc_prob)

      rng, chain_r, ss, K, Minv, old_pot, acc_prob = lax.fori_loop(1, numchoices, single_body, (rng, chain_r, ss, jnp.zeros(P-1), jnp.zeros((P-1,P-1)), old_pot, 0))

      # Endpoint updates for two ends
      def end_body(i, inner_carry):
        rng, chain_r, ss, _, _, old_pot, acc_prob = inner_carry
        rand_int = i * pbeads_r
        # jax.debug.print("2 {x}", x=(rand_int, i))
        rng, chain_prop = endpoint_sampling(rng, chain_r, rand_int, real_mass, tau_c, beta, pbeads_r)
        rng, chain_r_new, ss_new, K, Minv, old_pot_new, acc_prob = accept_move_fn(rng, chain_prop, chain_r, ss, old_pot)
        return (rng, chain_r_new, ss_new, K, Minv, old_pot_new, acc_prob)

      rng, chain_r, ss, K, Minv, old_pot, acc_prob = lax.fori_loop(0, 2, end_body, (rng, chain_r, ss, jnp.zeros(P-1), jnp.zeros((P-1,P-1)), old_pot, 0))


      # std = jnp.sqrt(K @ Minv @ K)
      # empirical_std = jnp.sqrt(2*beta*get_Utilde(ss[:, burn_in:, :], chain_r, beta))
      # std_err = std - empirical_std
      std_err = 0

      # jax.debug.print("std error {x}", x=(jnp.abs(std - empirical_std), std, empirical_std))
      # raw_samples = ss.reshape(num_chains*(num_unadjusted_steps - burn_in), ss.shape[2])
      # chi_samples, weights = (jax.vmap(lambda s : (xi(s,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma), s[i]))(raw_samples))
      # variance = jnp.mean(chi_samples**2)
      # Utilde = variance / (2 *beta)
      # todo

      
      
      # return (rng, chain_r, ss, old_pot), (chain_r, ss, K, Minv) # use this for plotting
      return (rng, chain_r, ss, old_pot), (chain_r, std_err, acc_prob)
    return step

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

  @jax.jit
  def pot_energy(r, Utilde):
    # Calculate the kinetic term: (mP / (2|τ_c|^2)) * Σ_{k=1}^{P} (r_{k+1} - r_k)^2
    # kinetic_term = (m * P) / (2 * jnp.abs(tau_c)**2) * jnp.sum((r[1:] - r[:-1])**2)
    # Calculate the potential term: (1 / (2P)) * (U(r_1) + U(r_{P+1}))
    potential_term = (1 / (2 * P)) * (U(r[0]) + U(r[P])) 

    # potential_term_2 = (1/P) * jnp.sum(U(r[1:-1]))

    # M, Minv, K, alpha, gamma, r = make_M_Minv_K(P, t, U, r, beta, hbar,m)

    # potential_term_3 = jnp.log(jax.scipy.linalg.det(M)) / (2 * beta)

    # potential_term_4 = (K @ Minv @ K) / (2 * beta)

    # return potential_term + potential_term_2 + potential_term_3 + potential_term_4




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

  # print(initial_ss.shape, "initial_ss shape")
  
  Utilde = get_Utilde(initial_ss[:, burn_in:, :], chain_r, beta)
  old_pot_init = pot_energy(chain_r, Utilde)
  # old_pot_init = pot_energy(chain_r)

  def accept_move(rng, chain_new, chain_current, ss, old_pot, beta, pot_energy_fn, L, step_size, inverse_mass_matrix):

    # return rng, chain_new, ss, jnp.zeros(P-1), jnp.zeros((P-1,P-1)), old_pot, jnp.minimum(1.0, 0.0)
    """
    Accept or reject a Monte Carlo move based on potential energy difference.
    
    Args:
      rng: JAX random key
      chain_new: Proposed new chain configuration
      chain_current: Current chain configuration
      ss: Current state for potential energy calculation
      old_pot: Current potential energy
      beta: Inverse temperature
      pot_energy_fn: Function to compute potential energy, signature (r, ss, key) -> (energy, new_ss)
    
    Returns:
      Tuple of (new_rng, updated_chain, new_ss, updated_potential)
    """
    rng, sub = jax.random.split(rng)
    ss_and_params = {'params': {'L': L, 'step_size': step_size, 'inverse_mass_matrix': inverse_mass_matrix}, 'ss': ss[:, -1, :]}
    _, _, new_ss, K, Minv = sample_s_chi(
        t=t,
        i=i,
        beta=beta,
        hbar=hbar,
        m=m,
        rng_key=sub,
        U = U,
        r=chain_new, #todo: should it be the new chain?
        sequential=False,
        initial_ss_and_params=ss_and_params,
        num_unadjusted_steps=num_unadjusted_steps, # md = unadjusted. This could probably be fewer.
        num_adjusted_steps=0, # mc = adjusted. 
        # filename=filename,
        num_chains=num_chains # this should be at large as possible while still fitting in memory.
        )
    # new_ss = ss

    Utilde = get_Utilde(new_ss[:, burn_in:, :], chain_new, beta)
    new_pot = pot_energy_fn(chain_new, Utilde)
    # new_pot = pot_energy_fn(chain_new)
    exp_pot = jnp.exp(-beta * (new_pot - old_pot))
    rng, sub = jax.random.split(rng)
    randval = jax.random.uniform(sub)
    accept = randval <= jnp.minimum(1.0, exp_pot)
    updated_r = jnp.where(accept, chain_new, chain_current)
    updated_pot = jnp.where(accept, new_pot, old_pot)

    new_M, new_Minv, new_K, new_alpha, new_gamma, new_r = make_M_Minv_K(P, t, U, updated_r, beta, hbar,m)

    updated_ss = jnp.where(accept, new_ss, ss)
    return rng, updated_r, updated_ss, new_K, new_Minv, updated_pot, jnp.minimum(1.0, exp_pot)


  # Precompute integer choices
  numchoices = int(np.floor((pbeads_r - 1) / (jval_r + 1))) + 1
  lft_wall_choices = jnp.arange(numchoices) * (jval_r + 1) + 1
  alt_jval_r = pbeads_r - lft_wall_choices[-1]
  lft_wall_choices = lft_wall_choices - 1  # zero-based

  accept_fn = partial(accept_move, beta=beta, pot_energy_fn=pot_energy, L=L, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix)
 
  step = make_mc_step(pbeads_r, jval_r, real_mass, beta, beads_sigma_sqrds, 
                      lft_wall_choices, numchoices, alt_jval_r, tau_c, accept_fn)

  init_carry = (rng, chain_r, initial_ss, old_pot_init)
  # (rng_out, chain_r_out, ss_out, old_pot_out), (all_samples, ss_hist, K, Minv) = lax.scan(step, init_carry, jnp.arange(mc_steps)) # use this for plotting
  import time as time_module
  tic = time_module.time()
  (rng_out, chain_r_out, ss_out, old_pot_out), (all_samples, std_errs, acc_probs) = lax.scan(step, init_carry, jnp.arange(mc_steps))
  print(time_module.time() - tic, "time of scan")
  plotting = False
  if plotting:

    index= 1

    print("\n ss out ", ss_hist.shape, K.shape, Minv.shape, all_samples.shape)
    flat_samples = ss_hist[index][:, burn_in:, :].reshape(num_chains*(num_unadjusted_steps-burn_in), ss_hist.shape[-1])
    gamma = (m*P*t)/(hbar * (jnp.abs(tau_c)**2)) 
    r_chain = all_samples[index]
    
    chi_samples, weights = (jax.vmap(lambda s : (xi(s,r=r_chain,U=U,t=t,P=P,hbar=hbar,gamma=gamma), s[i]))(flat_samples))

    std = jnp.sqrt(K[index] @ Minv[index] @ K[index])
    empirical_std = jnp.sqrt(jnp.mean(chi_samples**2))
    # jax.debug.print("std error final {x}", x=(jnp.abs(std - empirical_std), std, empirical_std))


    make_histograms('testing', samples=chi_samples, weights=None, K=K[index], Minv=Minv[index], i=i)
    raise Exception("Stop here")

  samples_np = np.asarray(all_samples[mc_equilibrate:])
  return samples_np, std_errs, acc_probs


if __name__ == "__main__":
  # Parameters
  numsteps = 100000
  equilibration = 0
  num_chains = 10000
  num_unadjusted_steps = 1
  burn_in = 0 # inner loop burn in

  P = 8
  j_r = 8
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
#   for time in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
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
    samples_np, std_errs, acc_probs = do_mc_open_chain(rng, numsteps, equilibration, r_chain, P, j_r, m, num_unadjusted_steps, num_chains, t=time)
    print(samples_np.shape)
    # save samples  
    np.save(f'samples_np_{time}_{j_r}.npy', samples_np)
    plt.plot(std_errs)
    plt.savefig(f'std_errs_{time}_{j_r}.png')
    plt.clf()
    # get running avg of acc_probs
    running_avg_acc_probs = np.cumsum(acc_probs) / np.arange(1, len(acc_probs) + 1)
    plt.plot(running_avg_acc_probs)
    # plt.savefig(f'running_avg_acc_probs_{time}.png')
    plt.savefig(f'acc_probs_{time}_{j_r}.png')
    # plt.clf()


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


