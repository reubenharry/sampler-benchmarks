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
  
  # Apply transformation for k=1 to k=j using scan
  k_vals = jnp.arange(1, int(j) + 1)
  staged, _ = jax.lax.scan(lambda staged, k: (update_staged(staged, k), None), staged, k_vals)
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
  
  # Apply inverse transformation in reverse order (k=j down to k=1) using scan
  k_vals = jnp.arange(int(j), 0, -1)
  prims, _ = jax.lax.scan(lambda prims, k: (update_prim(prims, k), None), prims, k_vals)
  return prims 

def do_intermediate_staging(chain_pos, left_wall, sigma_sqrds, j, big_p, key):
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
  
  # Use scan instead of for loop
  k_vals = jnp.arange(int(j))
  current_staged, _ = jax.lax.scan(lambda staged, k: (update_staged(staged, k), None), current_staged, k_vals)
  
  # Part 3: Inverse staging transformation (Equation 83)
  # Convert new staging coordinates back to primitive coordinates
  chain_pos = partial_chain_u_to_x(chain_pos, big_p, left_wall, j, current_staged)
  return chain_pos, current_staged

def stage_and_update_wall(chain_pos, left_wall, sigma_sqrds, j, big_p, key, old_pot, real_mass, omga, beta):
  """Complete function that stages, updates with Gaussians, unstages, and performs MH acceptance/rejection"""
  
  # Store original chain for potential rejection
  old_chain = chain_pos
  
  # Perform staging transformation
  chain_pos, staged = do_intermediate_staging(chain_pos, left_wall, sigma_sqrds, j, big_p, key)
  
  # Compute new potential energy
  new_pot = compute_potential(chain_pos, big_p, real_mass, omga)
  
  # Metropolis-Hastings acceptance/rejection
  # Equation (84): P(x'|x) = min[1, exp(-βΔŨ)]
  exp_pot = jnp.exp(-beta * (new_pot - old_pot))
  pacc = jnp.minimum(1.0, exp_pot)
  
  # Split key for acceptance decision
  key, accept_key = jr.split(key)
  accept = jr.uniform(accept_key) < pacc
  
  # Accept or reject the move
  chain_final, pot_final = jax.lax.cond(
    accept,
    lambda: (chain_pos, new_pot),
    lambda: (old_chain, old_pot)
  )
  
  return chain_final, pot_final

def endpoint_sampling(chain_pos, left_bead, real_mass, inv_kt, big_p, key):
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
  
  return jax.lax.cond(left_bead, 
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

omega_p = jnp.sqrt(pbeads_r) / beta
beads_sigma_sqrds = jnp.zeros(jval_r)
beads_sigma_sqrds = get_sigma_vals(beta, omega_p, real_mass, jval_r, beads_sigma_sqrds)
staged_r = jnp.zeros(pbeads_r+1)
old_r = jnp.zeros(pbeads_r+1)

old_pot = compute_potential(chain_r, pbeads_r, real_mass, omga)

def monte_carlo_step_deterministic(chain_r, old_pot, key, jval_r):
  # this should move in increments of j, and at each step, update the j beads to the right by the staging transform and gaussian update
  # then, the left and right endpoints should be updated using the endpoint sampling
  # a jax for loop should be used to update the j beads, and the left and right endpoints should be done in sequence

  interior_key, boundary_key = jr.split(key)
  walls = jnp.arange(0, pbeads_r+1, jval_r)
  current_pot = old_pot

  def process_wall(carry, wall):
    chain_r, current_pot, interior_key = carry
    # Split the key for this wall to ensure different randomness
    interior_key, new_interior_key = jr.split(interior_key)
    # Complete staging, updating, unstaging, and MH acceptance/rejection
    chain_r, current_pot = stage_and_update_wall(chain_r, wall, beads_sigma_sqrds, jval_r, pbeads_r, interior_key, current_pot, real_mass, omga, beta)
    return (chain_r, current_pot, new_interior_key), None

  # Use scan to process all walls
  (chain_r, current_pot, _), _ = jax.lax.scan(process_wall, (chain_r, current_pot, interior_key), walls)
  
  # Do endpoint sampling with different keys
  boundary_key, endpoint_key1, endpoint_key2 = jr.split(boundary_key, 3)
  chain_r = endpoint_sampling(chain_r, False, real_mass, beta, pbeads_r, endpoint_key1)
  chain_r = endpoint_sampling(chain_r, True, real_mass, beta, pbeads_r, endpoint_key2)

  return chain_r

  # raise Exception("Not implemented")

if __name__ == "__main__":
    import time
    
    
    # Run Monte Carlo simulation
    
    # Run the simulation
    start_time = time.time()
    samples = []
    key = jr.PRNGKey(42)
    
    # Run more steps for better statistics
    mc_steps = 30
    mc_equilibrate = 10
    print(f"Starting Monte Carlo simulation with {mc_steps} steps...")
    print(f"Parameters: pbeads_r={pbeads_r}, jval_r={jval_r}")
    
    for mcint in range(1, mc_steps+1):
        key = jax.random.fold_in(key, mcint)
        print(mcint)
        
        if mcint % 50 == 0:  # Print progress every 50 steps
            print(f"Step {mcint}/{mc_steps}")
        
        chain_r = monte_carlo_step_deterministic(chain_r, old_pot, key, jval_r)
        
        # Collect samples after equilibration
        if (mcint > mc_equilibrate):
            endpoint_diff = chain_r[0] - chain_r[pbeads_r]
            samples.append(endpoint_diff)
    
    total_time = time.time() - start_time
    print(f"✅ Simulation completed in {total_time:.2f} seconds")
    print(f"Collected {len(samples)} samples")

    endpoint_diff = lambda chain_r: chain_r[0] - chain_r[pbeads_r]

    # Create histogram of endpoint differences
    if len(samples) > 0:
        samples_np = np.array(samples)
        # print(np.mean(samples_np))
        print("Expectations")
        print(jax.vmap(endpoint_diff)(jnp.array(samples)))
        print(np.mean(jax.vmap(endpoint_diff)(jnp.array(samples))**2))
        dist_hist, dist_bin_edges = np.histogram(samples_np, bins=50, density=True)
        
        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(samples_np, bins=50, density=True, alpha=0.7, color='blue', label='Monte Carlo Samples')
        plt.xlabel('Endpoint Difference (x_0 - x_P)')
        plt.ylabel('Density')
        plt.title(f'Histogram of Endpoint Differences\n(Deterministic Monte Carlo, {len(samples)} samples)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('endpoint_differences_deterministic.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved histogram to endpoint_differences_deterministic.png")
        
        # Print some statistics
        print(f"Mean endpoint difference: {np.mean(samples_np):.4f}")
        print(f"Std endpoint difference: {np.std(samples_np):.4f}")
        print(f"Min endpoint difference: {np.min(samples_np):.4f}")
        print(f"Max endpoint difference: {np.max(samples_np):.4f}")
    else:
        print("No samples collected - skipping histogram generation")
