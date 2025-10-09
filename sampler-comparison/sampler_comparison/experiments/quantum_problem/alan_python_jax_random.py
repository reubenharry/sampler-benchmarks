import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

def get_sigma_vals(inv_kt, bead_omega, mass, j, sigma_sqrds):
  """Calculate variances for staging variables according to Equation (82)
     Each u_l+k is sampled from Gaussian with variance 1/(β * m_k * ω_P²)
     where m_k = (k+1)m/k and ω_P = sqrt(P/(βħ))"""
  for k in range(1,j+1):
    # m_k = (k+1)m/k, so variance = 1/(β * (k+1)m/k * ω_P²)
    # = k/(β * (k+1) * m * ω_P²)
    sigma_sqrds = sigma_sqrds.at[k-1].set(k / (inv_kt * (k+1) * mass * bead_omega * bead_omega))
  return sigma_sqrds

def x_to_u_staging(prims, left_wall, j):
  """Forward staging transformation: Equation (81)
     u_l+k = x_l+k - (k * x_l+k+1 + x_l) / (k+1)
     Converts primitive coordinates to staging coordinates"""
  staged = jnp.zeros_like(prims)
  
  def update_staged(staged, k):
    idx = left_wall + k
    # Equation (81): u_l+k = x_l+k - (k * x_l+k+1 + x_l) / (k+1)
    new_val = prims[idx] - (k * prims[idx + 1] + prims[left_wall]) / (k + 1)
    return staged.at[idx].set(new_val)
  
  # Apply transformation for k=1 to k=j
  for k in range(1, int(j) + 1):
    staged = update_staged(staged, k)
  return staged

def partial_chain_u_to_x(prims, num_beads, rand_bead, j, new_chain):
  """Inverse staging transformation: Equation (83)
     x_l+k = u_l+k + (k / (k+1)) * x_l+k+1 + (1 / (k+1)) * x_l
     Converts staging coordinates back to primitive coordinates"""
  def update_prim(prims, k):
    idx = rand_bead + k
    # Equation (83): x_l+k = u_l+k + (k / (k+1)) * x_l+k+1 + (1 / (k+1)) * x_l
    new_val = new_chain[idx] + (k * prims[rand_bead + k + 1] / (k + 1)) + (prims[rand_bead] / (k + 1))
    return prims.at[idx].set(new_val)
  
  # Apply inverse transformation in reverse order (k=j down to k=1)
  for k in range(int(j), 0, -1):
    prims = update_prim(prims, k)
  return prims 

def do_intermediate_staging(chain_pos, left_wall, staged, sigma_sqrds, j, big_p, key):
  """Perform staging transformation for intermediate beads
     Part 1: Convert primitive to staging coordinates (Equation 81)
     Part 2: Sample new staging variables from Gaussian (Equation 82)
     Part 3: Convert back to primitive coordinates (Equation 83)"""
  
  # Part 1: Forward staging transformation (Equation 81)
  # Convert current primitive coordinates to staging coordinates
  current_staged = x_to_u_staging(chain_pos, left_wall, j)
  
  # Part 2: Sample new staging variables from Gaussian distribution
  # According to Equation (82), u_l+k ~ N(0, 1/(β * m_k * ω_P²))
  def update_staged(staged, k):
    idx = left_wall + k + 1
    key_k = jr.fold_in(key, k)
    new_val = jr.normal(key_k, shape=()) * jnp.sqrt(sigma_sqrds[k])
    return staged.at[idx].set(new_val)
  
  for k in range(int(j)):
    staged = update_staged(staged, k)
  
  # Part 3: Inverse staging transformation (Equation 83)
  # Convert new staging coordinates back to primitive coordinates
  chain_pos = partial_chain_u_to_x(chain_pos, big_p, left_wall, j, staged)
  return chain_pos, staged

def endpoint_sampling(chain_pos, beadval, real_mass, inv_kt, big_p, key):
  """Sample first or last bead using Gaussian distribution
     Algorithm Step 3: Move first and last beads separately
     First bead: sample from N(x_2, βħ²/mP) with x_2 fixed
     Last bead: sample from N(x_P, βħ²/mP) with x_P fixed"""
  std = jnp.sqrt(inv_kt / (real_mass * big_p))
  
  def sample_first_bead(chain_pos, key):
    # First bead: mean = x_2, variance = βħ²/mP
    new_val = jr.normal(key, shape=()) * std + chain_pos[1]
    return chain_pos.at[0].set(new_val)
  
  def sample_last_bead(chain_pos, key):
    # Last bead: mean = x_P, variance = βħ²/mP  
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
  """Compute potential energy part of density (Equation 80)
     U(x_1) + U(x_{P+1}) / (2P) + Σ_{k=2}^{P} U(x_k) / P
     For harmonic oscillator: U(x) = 0.5 * m * ω² * x²"""
  # Endpoint contribution: (U(x_1) + U(x_{P+1})) / (2P)
  endpoint_contrib = 0.5 * 0.5 * real_mass * omga * omga * (chain_r[0]**2 + chain_r[pbeads_r]**2) / pbeads_r
  # Middle beads contribution: Σ_{k=2}^{P} U(x_k) / P
  middle_contrib = 0.5 * real_mass * omga * omga * jnp.sum(chain_r[1:pbeads_r]**2) / pbeads_r
  return endpoint_contrib + middle_contrib

#======================================================================================
# Parameters
mc_steps = 50  # Reduced for memory constraints
mc_equilibrate = 20  # Reduced equilibration time
pbeads_r = 8  # Much smaller for testing - large while loops are slow to compile in JAX
jval_r = 2  # Reduced j value
real_mass = 1.0
omga = 1.0
kbt = 1.0/10.0
#======================================================================================

beta = 1 / kbt
# Use same random seed as NumPy version
key = jr.PRNGKey(42)
key, subkey = jr.split(key)
chain_r = jr.normal(subkey, shape=(pbeads_r+1,)) * jnp.sqrt(kbt)
# chain_r = jr.uniform(subkey, shape=(pbeads_r+1,), maxval=1.0, minval=0.0)

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
  """Monte Carlo algorithm for open chains implementing Equations (81)-(85)
     Algorithm Steps:
     1. Segment sampling: Sample j beads using staging transformation
     2. Single bead sampling: Sample individual beads from set {x_i}^A_{i=1}
     3. Endpoint sampling: Sample first and last beads with Gaussian distributions"""
  
  moved = jnp.zeros(pbeads_r+1)
  
  # Keep track of the current staged array
  current_staged = staged_r
  
  # Algorithm Step 1: Segment sampling using staging transformation
  # Sample j beads to the left or right of randomly chosen wall bead
  def segment_cond(carry):
    chain_r, old_pot, moved, staged, key = carry
    moved_chek = jnp.sum(moved[1:pbeads_r])
    return moved_chek < (pbeads_r + 1 - 2 - numchoices)
  
  def segment_body(carry):
    chain_r, old_pot, moved, staged, key = carry
    key, subkey1, subkey2, subkey3 = jr.split(key, 4)
    old_r = chain_r
    
    # Pick a random bead from the list of choices to act as a wall and sample j beads to the left or right 
    leftwall = lft_wall_choices[jr.randint(subkey1, (), 0, numchoices)]
    direction = jr.randint(subkey2, (), 0, 2)
    # If going to the left, leftwall is adjusted
    leftwall = jax.lax.cond(direction == 0, 
                           lambda: leftwall - (jval_r + 1),
                           lambda: leftwall)
    
    # Sample beads
    def sample_alt_jval():
      # Update moved array for alt_jval_r case
      moved_new = moved
      for h in range(alt_jval_r):
        moved_new = moved_new.at[leftwall + h + 1].set(1)
      
      chain_r_new, staged_new = do_intermediate_staging(chain_r, leftwall, staged, beads_sigma_sqrds, alt_jval_r, pbeads_r, subkey3)
      return chain_r_new, staged_new, moved_new
    
    def sample_jval():
      # Update moved array for jval_r case
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
    
    # Get potential from proposal
    new_pot = compute_potential(chain_r_new, pbeads_r, real_mass, omga)
    
    # Accept or reject using Equations (84) and (85)
    # Equation (84): P(x'|x) = min[1, exp(-βΔŨ)]
    # Equation (85): ΔŨ = (1/P) [Σ_{k=ℓ+1}^{ℓ+j} U(x'_k) - Σ_{k=ℓ+1}^{ℓ+j} U(x_k)]
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
  
  # Apply segment sampling while loop
  chain_r, old_pot, moved, staged, key = jax.lax.while_loop(
    segment_cond, segment_body, (chain_r, old_pot, moved, current_staged, key)
  )
  
  # Algorithm Step 2: Single bead sampling from set {x_i}^A_{i=1}
  # Sample individual beads using staging transformation with j=1
  def single_cond(carry):
    chain_r, old_pot, moved, staged, key = carry
    moved_chek = jnp.sum(moved[1:pbeads_r])
    return moved_chek < (pbeads_r - 1)
  
  def single_body(carry):
    chain_r, old_pot, moved, staged, key = carry
    key, subkey1, subkey2 = jr.split(key, 3)
    old_r = chain_r
    
    # Pick a random bead from the list of choices and sample that one bead 
    leftwall = lft_wall_choices[jr.randint(subkey1, (), 0, numchoices)] - 1
    
    # Sample
    chain_r_new, staged_new = do_intermediate_staging(chain_r, leftwall, staged, beads_sigma_sqrds, 1, pbeads_r, subkey2)
    
    # Get potential from proposal
    new_pot = compute_potential(chain_r_new, pbeads_r, real_mass, omga)
    
    # Accept or reject
    key, subkey = jr.split(key)
    exp_pot = jnp.exp(-beta * (new_pot - old_pot))
    pacc = jnp.minimum(1.0, exp_pot)
    accept = jr.uniform(subkey) < pacc
    
    chain_r_final, old_pot_new = jax.lax.cond(
      accept,
      lambda: (chain_r_new, new_pot),
      lambda: (old_r, old_pot)
    )
    
    # Recording which bead got moved (mark as attempted regardless of acceptance)
    moved_new = moved.at[leftwall + 1].set(1)
    
    return chain_r_final, old_pot_new, moved_new, staged_new, key
  
  # Apply single bead sampling while loop
  chain_r, old_pot, moved, staged, key = jax.lax.while_loop(
    single_cond, single_body, (chain_r, old_pot, moved, staged, key)
  )
  
  # Algorithm Step 3: Endpoint sampling
  # Sample first and last beads using Gaussian distributions
  def endpoint_cond(carry):
    chain_r, old_pot, moved, staged, key = carry
    moved_chek = jnp.sum(moved)
    return moved_chek < (pbeads_r + 1)
  
  def endpoint_body(carry):
    chain_r, old_pot, moved, staged, key = carry
    key, subkey1, subkey2 = jr.split(key, 3)
    old_r = chain_r
    
    # Pick first or last bead in the chain and then sample it
    rand_int = jr.randint(subkey1, (), 0, 2) * pbeads_r
    
    # Sample
    chain_r_new = endpoint_sampling(chain_r, rand_int, real_mass, beta, pbeads_r, subkey2)
    
    # Get potential from proposal
    new_pot = compute_potential(chain_r_new, pbeads_r, real_mass, omga)
    
    # Accept or reject
    key, subkey = jr.split(key)
    exp_pot = jnp.exp(-beta * (new_pot - old_pot))
    pacc = jnp.minimum(1.0, exp_pot)
    accept = jr.uniform(subkey) < pacc
    
    chain_r_final, old_pot_final = jax.lax.cond(
      accept,
      lambda: (chain_r_new, new_pot),
      lambda: (old_r, old_pot)
    )
    
    # Recording which bead got moved (mark as attempted regardless of acceptance)
    moved_new = moved.at[rand_int].set(1)
    
    return chain_r_final, old_pot_final, moved_new, staged, key
  
  # Apply endpoint sampling while loop
  chain_r, old_pot, moved, staged, key = jax.lax.while_loop(
    endpoint_cond, endpoint_body, (chain_r, old_pot, moved, staged, key)
  )
  
  return chain_r, old_pot, key

def monte_carlo_step_deterministic(chain_r, old_pot, key, jval_r):
  # this should move in increments of j, and at each step, update the j beads to the right by the staging transform and gaussian update
  # then, the left and right endpoints should be updated using the endpoint sampling
  # a jax for loop should be used to update the j beads, and the left and right endpoints should be done in sequence

  print("jval_r: ", jval_r)
  interior_key, boundary_key = jr.split(key)

  walls = jnp.arange(0, pbeads_r+1, jval_r)
  print("pbeads_r: ", pbeads_r)
  print("walls: ", walls)

  for wall in walls:
    # sampling the j beads to the right
    chain_r, staged = do_intermediate_staging(chain_r, wall, staged, beads_sigma_sqrds, jval_r, pbeads_r, interior_key)
    # do the endpoint sampling
    chain_r = endpoint_sampling(chain_r, wall, real_mass, beta, pbeads_r, boundary_key)

  return chain_r

  # raise Exception("Not implemented")

if __name__ == "__main__":
    import time
    
    # Run Monte Carlo simulation
    print(f"Starting Monte Carlo simulation with {mc_steps} steps...")
    print(f"Parameters: pbeads_r={pbeads_r}, jval_r={jval_r}")
    
    # Run the simulation
    start_time = time.time()
    samples = []
    for mcint in range(1, mc_steps+1):

        print(mcint)

        # monte_carlo_step_deterministic(chain_r, old_pot, key, jval_r)
        chain_r, old_pot, key = monte_carlo_step(chain_r, old_pot, key, lft_wall_choices, numchoices, jval_r, alt_jval_r, pbeads_r, beads_sigma_sqrds, staged_r, real_mass, omga, beta)
        
        # # Grabbing samples 
        if (mcint > mc_equilibrate):
            samples.append(chain_r[0] - chain_r[pbeads_r])
    
    # total_time = time.time() - start_time
    # print(f"✅ Simulation completed in {total_time:.2f} seconds")
    # print(f"Collected {len(samples)} samples")

    # m = 1
    # w = 1
    # # Make a histogram of the open path end-to-end distance
    # if len(samples) > 0:
    #     samples_np = np.array(samples)
    #     dist_hist, dist_bin_edges = np.histogram(samples_np, bins=50, density=True)
    #     ideal_x = np.arange(-10, 10, .1)
    #     ideal_prediction_x = get_harmonic_density(ideal_x, 1/kbt, m, w)
    #     ideal_prediction_x_np = np.array(ideal_prediction_x)
    #     plt.plot(ideal_x, ideal_prediction_x_np, linestyle='-', label='Exact', color='blue')
    #     plt.plot(bin_centers(dist_bin_edges), dist_hist, linestyle='--', label=r'$N(u_{P+1})$', color='red')
    #     plt.legend(loc='upper right')
    #     plt.xlabel(r'$u_{P+1}$')
    #     plt.ylabel(r'$N(u_{P+1})$')
    #     plt.title('Estimator for harmonic oscillator density matrix (JAX with JIT)')
    #     plt.savefig('h_density_jax.png')
    #     plt.clf()
    #     print(f"saved histogram to h_density_jax.png")
    # else:
    #     print("No samples collected - skipping histogram generation")
