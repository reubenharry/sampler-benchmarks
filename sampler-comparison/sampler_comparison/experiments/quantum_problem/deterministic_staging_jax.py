import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt
import time
from quantum_problem import sample_s_chi, xi, make_M_Minv_K

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


def endpoint_sampling(rng, chain_pos, beadval, real_mass, inv_kt, big_p):
  std = jnp.sqrt(inv_kt / (real_mass * big_p))
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


def do_mc_open_chain(rng, mc_steps, mc_equilibrate, chain_r, pbeads_r, jval_r, real_mass, omga, kbt):
  beta = 1.0 / kbt
  omega_p = jnp.sqrt(pbeads_r) / beta
  

  beads_sigma_sqrds = get_sigma_vals(beta, omega_p, real_mass, jval_r, jnp.zeros(jval_r))

  def pot_energy(r, ss, key):
    # Calculate tau_c
    M, Minv, K, alpha, gamma, r = make_M_Minv_K(P, t, U, r, beta, hbar,m)
    
    # Calculate the kinetic term: (mP / (2|τ_c|^2)) * Σ_{k=1}^{P} (r_{k+1} - r_k)^2
    kinetic_term = (m * P) / (2 * jnp.abs(tau_c)**2) * jnp.sum((r[1:] - r[:-1])**2)
    
    # Calculate the potential term: (1 / (2P)) * (U(r_1) + U(r_{P+1}))
    potential_term = (1 / (2 * P)) * (U(r[0]) + U(r[P]))
    
    # Apply the formula: -β * [kinetic_term + potential_term]
    
    # Keep the existing sampling logic for ss
    ss_and_params = {'params': {'L': L, 'step_size': step_size, 'inverse_mass_matrix': inverse_mass_matrix}, 'ss': ss[:, -1, :]}
    _, _, new_ss, _, _ = sample_s_chi(
        t=1,
        i=1,
        beta=beta,
        hbar=hbar,
        m=m,
        rng_key=key,
        U = U,
        r=r,
        sequential=False,
        initial_ss_and_params=ss_and_params,
        num_unadjusted_steps=num_unadjusted_steps, # md = unadjusted. This could probably be fewer.
        num_adjusted_steps=0, # mc = adjusted. 
        # filename=filename,
        num_chains=num_chains # this should be at large as possible while still fitting in memory.
        )
    raw_samples = new_ss.reshape(num_chains*num_unadjusted_steps, new_ss.shape[2])
    samples, weights = (jax.vmap(lambda x : (xi(x,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma), x[i]))(raw_samples))

    print("samples.shape", samples.shape, weights.shape)
    # jax.debug.print("samples {x}", x=samples)

    variance = jnp.mean(samples**2)
    Utilde = jnp.exp(-0.5 * variance)
    pot_energy_value = -beta * (kinetic_term + potential_term + Utilde)
    
    return pot_energy_value, new_ss

  # Initial potential
  def pot_energy_simple(r, ss, key):
    first = 0.5 * 0.5 * real_mass * omga * omga * r[0] * r[0] / pbeads_r
    middle = 0.5 * real_mass * omga * omga * jnp.sum(r[1:pbeads_r] ** 2) / pbeads_r
    last = 0.5 * 0.5 * real_mass * omga * omga * r[pbeads_r] * r[pbeads_r] / pbeads_r

    new_ss = ss
    # print("new_ss shape", new_ss.shape)
    # print("samples shape", raw_samples.shape)
    return first + middle + last, new_ss

  r_length = chain_r.shape[0]

  
  initial_ss, L, step_size, inverse_mass_matrix = sample_s_chi(
        t=1,
        i=1,
        beta=beta,
        hbar=hbar,
        m=m,
        U = lambda x : 0.5*m*(omga**2)*(x**2),
        r=chain_r,
        sequential=False,
        initial_ss_and_params=None,
        sample_init=None,
        num_unadjusted_steps=num_unadjusted_steps, # md = unadjusted. This could probably be fewer.
        num_adjusted_steps=0, # mc = adjusted. 
        num_chains=num_chains # this should be at large as possible while still fitting in memory.
        )

  old_pot_init, new_ss = pot_energy(chain_r, initial_ss, jax.random.PRNGKey(0))

  # Precompute integer choices
  numchoices = int(np.floor((pbeads_r - 1) / (jval_r + 1))) + 1
  lft_wall_choices = jnp.arange(numchoices) * (jval_r + 1) + 1
  lft_wall_choices = lft_wall_choices - 1  # zero-based
  alt_jval_r = pbeads_r - lft_wall_choices[-1]

  def accept_move(rng, chain_new, chain_current, ss, old_pot):
    rng, sub = jax.random.split(rng)
    new_pot, new_ss = pot_energy(chain_new, ss, sub)
    exp_pot = jnp.exp(-beta * (new_pot - old_pot))
    rng, sub = jax.random.split(rng)
    randval = jax.random.uniform(sub)
    accept = randval <= jnp.minimum(1.0, exp_pot)
    updated_r = jnp.where(accept, chain_new, chain_current)
    updated_pot = jnp.where(accept, new_pot, old_pot)
    return rng, updated_r, new_ss, updated_pot

  def mc_step(carry, step_idx):
    rng, chain_r, ss, old_pot = carry
    jax.debug.print("step_idx {x}", x=step_idx)

    # Segment updates over all left walls
    def seg_body(i, inner_carry):
      rng, chain_r, ss, old_pot = inner_carry
      leftwall = lft_wall_choices[i]
      j_this = jnp.where(i == (numchoices - 1), alt_jval_r, jval_r)
      rng, chain_prop = do_intermediate_staging(rng, chain_r, leftwall, beads_sigma_sqrds, j_this, pbeads_r, jval_r)
      rng, chain_r_new, ss_new, old_pot_new = accept_move(rng, chain_prop, chain_r, ss, old_pot)
      return (rng, chain_r_new, ss_new, old_pot_new)

    rng, chain_r, ss, old_pot = lax.fori_loop(0, numchoices, seg_body, (rng, chain_r, ss, old_pot))

    # Single-bead updates (excluding first choice)
    def single_body(i, inner_carry):
      rng, chain_r, ss, old_pot = inner_carry
      leftwall = lft_wall_choices[i] - 1
      rng, chain_prop = do_intermediate_staging(rng, chain_r, leftwall, beads_sigma_sqrds, jnp.array(1), pbeads_r, jval_r)
      rng, chain_r_new, ss_new, old_pot_new = accept_move(rng, chain_prop, chain_r, ss, old_pot)
      return (rng, chain_r_new, ss_new, old_pot_new)

    rng, chain_r, ss, old_pot = lax.fori_loop(1, numchoices, single_body, (rng, chain_r, ss, old_pot))

    # Endpoint updates for two ends
    def end_body(i, inner_carry):
      rng, chain_r, ss, old_pot = inner_carry
      rand_int = i * pbeads_r
      rng, chain_prop = endpoint_sampling(rng, chain_r, rand_int, real_mass, beta, pbeads_r)
      rng, chain_r_new, ss_new, old_pot_new = accept_move(rng, chain_prop, chain_r, ss, old_pot)
      return (rng, chain_r_new, ss_new, old_pot_new)

    rng, chain_r, ss, old_pot = lax.fori_loop(0, 2, end_body, (rng, chain_r, ss, old_pot))

    # Output the sample value (difference between endpoints)
    sample_value = chain_r[0] - chain_r[pbeads_r]
    
    return (rng, chain_r, ss, old_pot), sample_value

  


  init_carry = (rng, chain_r, initial_ss, old_pot_init)
  jax.debug.print("num steps {x}", x=mc_steps)
  (rng_out, chain_r_out, ss_out, old_pot_out), all_samples = lax.scan(mc_step, init_carry, jnp.arange(mc_steps))

  # Extract only post-equilibration samples
  samples_np = np.asarray(all_samples[mc_equilibrate:])
  histvals, hist_bin_edges = np.histogram(samples_np, bins=50)
  return histvals, hist_bin_edges


if __name__ == "__main__":
  # Parameters
  numsteps = 3000
  equilibration = 1000
  num_chains = 10000
  num_unadjusted_steps = 100

  bigp = 32
  P = bigp
  j_r = 8
  # m = 1.0
  # harm_omga = 1.0
  beta_hbar_omega = 15.8
  m_omega_over_hbar = 0.03
  m = 1.0
  hbar = 1.0
  i=1
  U = lambda x : 0.5*m*(omega**2)*(x**2)

  
  omega = (m_omega_over_hbar * hbar) / m
  inverse_kbt = 1.0 / 10.0
  beta = inverse_kbt
  hbar = 1.0
  t = 1.0
  im = 0 + 1j
  # beta = (beta_hbar_omega / (hbar * omega))
  tau_c = t - ((beta * hbar * im) / 2)

  # Initial chain from uniform(0,1)
  rng = jax.random.PRNGKey(0)
  rng, sub = jax.random.split(rng)
  r_chain = jax.random.uniform(sub, shape=(bigp + 1,))

  m = 1.0
  w = 1.0

  tic = time.time()
  dist_hist, dist_bin_edges = do_mc_open_chain(rng, numsteps, equilibration, r_chain, bigp, j_r, m, omega, inverse_kbt)
  print(time.time() - tic, "time")
  # Normalize histogram
  sum1 = 0.0
  for s in range(len(dist_hist)):
    sum1 += dist_hist[s] * (dist_bin_edges[1] - dist_bin_edges[0])
  dist_hist = dist_hist / sum1

  # Plot
  ideal_x = np.arange(-7, 7, 0.05)
  ideal_prediction_x = np.asarray(get_harmonic_density(jnp.array(ideal_x), 1 / inverse_kbt, m, w))
  # plt.plot(ideal_x, ideal_prediction_x, linestyle='-', label='Exact', color='blue')
  plt.plot(bin_centers(dist_bin_edges), dist_hist, 'o', label=r'$N(u_{P+1})$', color='red')
  plt.legend(loc='upper right')
  plt.xlabel(r'$u_{P+1}$')
  plt.ylabel(r'$N(u_{P+1})$')
  plt.title('Estimator for harmonic oscillator density matrix (JAX)')
  plt.savefig('the_density.png')
  plt.clf()


