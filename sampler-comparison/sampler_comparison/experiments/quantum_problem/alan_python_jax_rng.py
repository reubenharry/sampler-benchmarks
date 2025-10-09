import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

def get_sigma_vals(inv_kt, bead_omega, mass, j, sigma_sqrds):
  for k in range(1,j+1):
    sigma_sqrds = sigma_sqrds.at[k-1].set(k / (inv_kt * (k+1) * mass * bead_omega * bead_omega))
  return sigma_sqrds

def partial_chain_u_to_x(prims, num_beads, rand_bead, j, new_chain):
  def update_prim(prims, k):
    idx = rand_bead + k
    new_val = new_chain[idx] + (k * prims[rand_bead + k + 1] / (k + 1)) + (prims[rand_bead] / (k + 1))
    return prims.at[idx].set(new_val)
  
  for k in range(j, 0, -1):
    prims = update_prim(prims, k)
  return prims 

def do_intermediate_staging(chain_pos, left_wall, staged, sigma_sqrds, j, big_p, key):
  def update_staged(staged, k):
    idx = left_wall + k + 1
    key_k = jr.fold_in(key, k)
    new_val = jr.normal(key_k, shape=()) * jnp.sqrt(sigma_sqrds[k])
    return staged.at[idx].set(new_val)
  
  for k in range(j):
    staged = update_staged(staged, k)
  
  chain_pos = partial_chain_u_to_x(chain_pos, big_p, left_wall, j, staged)
  return chain_pos, staged

def endpoint_sampling(chain_pos, beadval, real_mass, inv_kt, big_p, key):
  std = jnp.sqrt(inv_kt / (real_mass * big_p))
  
  def sample_first_bead(chain_pos, key):
    new_val = jr.normal(key, shape=()) * std + chain_pos[1]
    return chain_pos.at[0].set(new_val)
  
  def sample_last_bead(chain_pos, key):
    new_val = jr.normal(key, shape=()) * std + chain_pos[big_p - 1]
    return chain_pos.at[big_p].set(new_val)
  
  return jax.lax.cond(beadval == 0, 
                     lambda: sample_first_bead(chain_pos, key),
                     lambda: sample_last_bead(chain_pos, key))

def bin_centers(bin_edges):
    return (bin_edges[1:]+bin_edges[:-1])/2.

def get_harmonic_density(pos, inv_temp, mass, omega):
    normalization = jnp.sqrt((mass*omega) / (4*jnp.pi*jnp.tanh(inv_temp*omega/2)))
    exp_constant = jnp.pi * normalization**2
    return normalization * jnp.exp(-exp_constant * pos**2 )

def compute_potential(chain_r, pbeads_r, real_mass, omga):
  endpoint_contrib = 0.5 * 0.5 * real_mass * omga * omga * (chain_r[0]**2 + chain_r[pbeads_r]**2) / pbeads_r
  middle_contrib = 0.5 * real_mass * omga * omga * jnp.sum(chain_r[1:pbeads_r]**2) / pbeads_r
  return endpoint_contrib + middle_contrib

#======================================================================================
# Parameters
mc_steps = 300
mc_equilibrate = 100

pbeads_r = 300
jval_r = 60
real_mass = 1.0
omga = 1.0
kbt = 1.0/10.0
#======================================================================================

beta = 1 / kbt
# Use fixed JAX key for reproducible randomness
key = jr.PRNGKey(42)
key, subkey = jr.split(key)
chain_r = jr.normal(subkey, shape=(pbeads_r+1,)) * jnp.sqrt(kbt)

omega_p = jnp.sqrt(pbeads_r) / beta
beads_sigma_sqrds = jnp.zeros(jval_r)
beads_sigma_sqrds = get_sigma_vals(beta, omega_p, real_mass, jval_r, beads_sigma_sqrds)
staged_r = jnp.zeros(pbeads_r+1)
old_r = jnp.zeros(pbeads_r+1)

old_pot = compute_potential(chain_r, pbeads_r, real_mass, omga)

numchoices = int(jnp.floor( (pbeads_r-1) / (jval_r+1) ))
lft_wall_choices = jnp.zeros(numchoices, dtype=int)
for h in range(1,numchoices+1):
  lft_wall_choices = lft_wall_choices.at[h-1].set(h*(jval_r+1) + 1)
alt_jval_r = pbeads_r - lft_wall_choices[numchoices-1]
lft_wall_choices = lft_wall_choices - 1

def monte_carlo_step(chain_r, old_pot, key, lft_wall_choices, numchoices, jval_r, alt_jval_r, pbeads_r, beads_sigma_sqrds, staged_r, real_mass, omga, beta):
  """JAX implementation using the same random key as NumPy version"""
  
  moved = jnp.zeros(pbeads_r+1)
  current_staged = staged_r
  
  # First while loop - segment sampling
  def segment_cond(carry):
    chain_r, old_pot, moved, staged, key = carry
    moved_chek = jnp.sum(moved[1:pbeads_r])
    return moved_chek < (pbeads_r + 1 - 2 - numchoices)
  
  def segment_body(carry):
    chain_r, old_pot, moved, staged, key = carry
    key, subkey1, subkey2, subkey3 = jr.split(key, 4)
    old_r = chain_r
    
    leftwall = lft_wall_choices[jr.randint(subkey1, (), 0, numchoices)]
    direction = jr.randint(subkey2, (), 0, 2)
    leftwall = jax.lax.cond(direction == 0, 
                           lambda: leftwall - (jval_r + 1),
                           lambda: leftwall)
    
    def sample_alt_jval():
      moved_new = moved
      for h in range(alt_jval_r):
        moved_new = moved_new.at[leftwall + h + 1].set(1)
      chain_r_new, staged_new = do_intermediate_staging(chain_r, leftwall, staged, beads_sigma_sqrds, alt_jval_r, pbeads_r, subkey3)
      return chain_r_new, staged_new, moved_new
    
    def sample_jval():
      moved_new = moved
      for h in range(jval_r):
        moved_new = moved_new.at[leftwall + h + 1].set(1)
      chain_r_new, staged_new = do_intermediate_staging(chain_r, leftwall, staged, beads_sigma_sqrds, jval_r, pbeads_r, subkey3)
      return chain_r_new, staged_new, moved_new
    
    chain_r_new, staged_new, moved_new = jax.lax.cond(
      (leftwall == lft_wall_choices[numchoices-1]) & (direction == 1),
      sample_alt_jval,
      sample_jval
    )
    
    new_pot = compute_potential(chain_r_new, pbeads_r, real_mass, omga)
    
    key, subkey = jr.split(key)
    exp_pot = jnp.exp(-beta * (new_pot - old_pot))
    pacc = jnp.minimum(1.0, exp_pot)
    accept = jr.uniform(subkey) < pacc
    
    chain_r_final, old_pot_new = jax.lax.cond(
      accept,
      lambda: (chain_r_new, new_pot),
      lambda: (old_r, old_pot)
    )
    
    # The moved array tracks which beads have been ATTEMPTED (not just accepted)
    # So we use moved_new which was set before the acceptance decision
    moved_final = moved_new
    
    return chain_r_final, old_pot_new, moved_final, staged_new, key
  
  chain_r, old_pot, moved, staged, key = jax.lax.while_loop(
    segment_cond, segment_body, (chain_r, old_pot, moved, current_staged, key)
  )
  
  # Second while loop - single bead sampling
  def single_cond(carry):
    chain_r, old_pot, moved, staged, key = carry
    moved_chek = jnp.sum(moved[1:pbeads_r])
    return moved_chek < (pbeads_r - 1)
  
  def single_body(carry):
    chain_r, old_pot, moved, staged, key = carry
    key, subkey1, subkey2 = jr.split(key, 3)
    old_r = chain_r
    
    leftwall = lft_wall_choices[jr.randint(subkey1, (), 0, numchoices)] - 1
    chain_r_new, staged_new = do_intermediate_staging(chain_r, leftwall, staged, beads_sigma_sqrds, 1, pbeads_r, subkey2)
    
    new_pot = compute_potential(chain_r_new, pbeads_r, real_mass, omga)
    
    key, subkey = jr.split(key)
    exp_pot = jnp.exp(-beta * (new_pot - old_pot))
    pacc = jnp.minimum(1.0, exp_pot)
    accept = jr.uniform(subkey) < pacc
    
    chain_r_final, old_pot_new = jax.lax.cond(
      accept,
      lambda: (chain_r_new, new_pot),
      lambda: (old_r, old_pot)
    )
    
    moved_new = jax.lax.cond(accept, 
                            lambda: moved.at[leftwall + 1].set(1),
                            lambda: moved)
    
    return chain_r_final, old_pot_new, moved_new, staged_new, key
  
  chain_r, old_pot, moved, staged, key = jax.lax.while_loop(
    single_cond, single_body, (chain_r, old_pot, moved, staged, key)
  )
  
  # Third while loop - endpoint sampling
  def endpoint_cond(carry):
    chain_r, old_pot, moved, staged, key = carry
    moved_chek = jnp.sum(moved)
    return moved_chek < (pbeads_r + 1)
  
  def endpoint_body(carry):
    chain_r, old_pot, moved, staged, key = carry
    key, subkey1, subkey2 = jr.split(key, 3)
    old_r = chain_r
    
    rand_int = jr.randint(subkey1, (), 0, 2) * pbeads_r
    chain_r_new = endpoint_sampling(chain_r, rand_int, real_mass, beta, pbeads_r, subkey2)
    
    new_pot = compute_potential(chain_r_new, pbeads_r, real_mass, omga)
    
    key, subkey = jr.split(key)
    exp_pot = jnp.exp(-beta * (new_pot - old_pot))
    pacc = jnp.minimum(1.0, exp_pot)
    accept = jr.uniform(subkey) < pacc
    
    chain_r_final, old_pot_final = jax.lax.cond(
      accept,
      lambda: (chain_r_new, new_pot),
      lambda: (old_r, old_pot)
    )
    
    moved_new = jax.lax.cond(accept,
                            lambda: moved.at[rand_int].set(1),
                            lambda: moved)
    
    return chain_r_final, old_pot_final, moved_new, staged, key
  
  chain_r, old_pot, moved, staged, key = jax.lax.while_loop(
    endpoint_cond, endpoint_body, (chain_r, old_pot, moved, staged, key)
  )
  
  return chain_r, old_pot, key

if __name__ == "__main__":
    # Run Monte Carlo simulation
    samples = []
    for mcint in range(1, mc_steps+1):
        chain_r, old_pot, key = monte_carlo_step(chain_r, old_pot, key, lft_wall_choices, numchoices, jval_r, alt_jval_r, pbeads_r, beads_sigma_sqrds, staged_r, real_mass, omga, beta)
        
        if (mcint > mc_equilibrate):
            samples.append(chain_r[0] - chain_r[pbeads_r])

    m = 1
    w = 1
    # Make a histogram of the open path end-to-end distance
    samples_np = np.array(samples)
    dist_hist, dist_bin_edges = np.histogram(samples_np, bins=50, density=True)
    ideal_x = np.arange(-10, 10, .1)
    ideal_prediction_x = get_harmonic_density(ideal_x, 1/kbt, m, w)
    ideal_prediction_x_np = np.array(ideal_prediction_x)
    plt.plot(ideal_x, ideal_prediction_x_np, linestyle='-', label='Exact', color='blue')
    plt.plot(bin_centers(dist_bin_edges), dist_hist, linestyle='--', label=r'$N(u_{P+1})$', color='red')
    plt.legend(loc='upper right')
    plt.xlabel(r'$u_{P+1}$')
    plt.ylabel(r'$N(u_{P+1})$')
    plt.title('Estimator for harmonic oscillator density matrix (JAX with JAX RNG)')
    plt.savefig('h_density_jax_rng.png')
    plt.clf()
    print(f"saved histogram to h_density_jax_rng.png")

