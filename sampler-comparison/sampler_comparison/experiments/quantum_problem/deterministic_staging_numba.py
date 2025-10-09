import numba
from numba import jit
import numpy as np
import matplotlib.pyplot as plt

@jit(nopython=True)
def get_sigma_vals(inv_kt, bead_omega, mass, j, sigma_sqrds):
  for k in range(1,j+1):
    sigma_sqrds[k-1] = k / (inv_kt * (k+1) * mass * bead_omega * bead_omega)
  return sigma_sqrds

@jit(nopython=True)
def partial_chain_u_to_x(prims, num_beads, rand_bead, j, new_chain):
  for k in range(j,0,-1):
    prims[rand_bead + k] = new_chain[rand_bead + k] + ( k*prims[rand_bead + k + 1] / (k+1)) + (prims[rand_bead] / (k+1) )
  return prims 

@jit(nopython=True)
def do_intermediate_staging(chain_pos,  left_wall, staged, sigma_sqrds, j, big_p):
  staged = np.zeros(big_p+1)
  for k in range(j):
    staged[left_wall + k + 1] = np.random.normal(0, np.sqrt(sigma_sqrds[k]))
  chain_pos = partial_chain_u_to_x(chain_pos, big_p, left_wall, j, staged)
  return chain_pos

@jit(nopython=True)
def endpoint_sampling(chain_pos, beadval, real_mass, inv_kt, big_p):
  if (beadval == 0):
    chain_pos[0] = np.random.normal(chain_pos[1], np.sqrt(inv_kt / (real_mass*big_p)))
  else:
    chain_pos[big_p] = np.random.normal(chain_pos[big_p-1], np.sqrt(inv_kt / (real_mass*big_p)) )
  return chain_pos

def bin_centers(bin_edges):
    return (bin_edges[1:]+bin_edges[:-1])/2.

def get_harmonic_density(pos, inv_temp, mass, omega):
    normalization = np.sqrt((mass*omega) / (4*np.pi*np.tanh(inv_temp*omega/2)))
    exp_constant = np.pi * normalization**2
    return normalization * np.exp(-exp_constant * pos**2 )

@jit(nopython=True)
def do_mc_open_chain(mc_steps, mc_equilibrate, chain_r, pbeads_r, jval_r, real_mass, omga, kbt):
  beta = 1 / kbt
  omega_p = np.sqrt(pbeads_r) / beta
  
  beads_sigma_sqrds = np.zeros(jval_r)
  beads_sigma_sqrds = get_sigma_vals(beta, omega_p, real_mass, jval_r, beads_sigma_sqrds)
  
  staged_r = np.zeros(pbeads_r+1)
  old_r = np.zeros(pbeads_r+1)
  
  # Get spring potential from initial r variables
  old_pot = 0.0
  old_pot += 0.5 * 0.5 * real_mass * omga * omga * chain_r[0] * chain_r[0] / pbeads_r
  for h in range(1,pbeads_r):
    old_pot += 0.5 * real_mass * omga * omga * chain_r[h] * chain_r[h] / pbeads_r
  old_pot += 0.5 * 0.5 * real_mass * omga * omga * chain_r[pbeads_r] * chain_r[pbeads_r] / pbeads_r
  
  # Making the list of bead labels for segment sampling 
  numchoices = int(np.floor( (pbeads_r-1) / (jval_r+1) ))
  # Add 1 to account for the possibility of having a segment at the end of the chain that 
  #    is alt_jval_r beads long. Works even if alt_jval_r = 0
  numchoices += 1

  lft_wall_choices = np.zeros(numchoices, dtype=numba.types.int32)

  for h in range(numchoices):
    lft_wall_choices[h] = h*(jval_r+1) + 1
  # If we choose the last bead in lft_wall_choices and sample j beads to the right, we may only be able to 
  #   sample alt_jval_r beads where alt_jval_r <= jval_r. For example when P = 13 and j = 4, alt_jval_r = 2
  alt_jval_r = pbeads_r - lft_wall_choices[numchoices-1]
  
  # (subtracting 1 because python has 0 based indexing)
  lft_wall_choices -= 1 

  pacc_list = np.zeros(2)
  pacc_list[0] = 1.0

  samp_counter = 0
  samples = np.zeros(mc_steps-mc_equilibrate)
  for mcint in range(1,mc_steps+1):

    # Loop over all the left wall choices and sample j beads to the right
    for mcint2 in range(numchoices):
      # Save primitive chain before proposal
      for el in range(pbeads_r+1):
        old_r[el] = chain_r[el]
  
      #===============================================================================================
      # Select the left wall
      leftwall = lft_wall_choices[mcint2]
      # direction is now always to the right
      direction = 1
      #===============================================================================================
      # Sample beads
      if (leftwall == lft_wall_choices[numchoices-1] and direction == 1):
        chain_r = do_intermediate_staging(chain_r, leftwall, staged_r, beads_sigma_sqrds, alt_jval_r, pbeads_r)
      else:
        chain_r = do_intermediate_staging(chain_r, leftwall, staged_r, beads_sigma_sqrds, jval_r, pbeads_r)
      #===============================================================================================
      # Get potential from proposal
      new_pot = 0.0
      new_pot += 0.5 * 0.5 * real_mass * omga * omga * chain_r[0] * chain_r[0] / pbeads_r
      for h in range(1,pbeads_r):
        new_pot += 0.5 * real_mass * omga * omga * chain_r[h] * chain_r[h] / pbeads_r
      new_pot += 0.5 * 0.5 * real_mass * omga * omga * chain_r[pbeads_r] * chain_r[pbeads_r] / pbeads_r
      #===============================================================================================
      # Accept or reject
      exp_pot = np.exp( -beta * (new_pot - old_pot) ) 
      pacc_list[1] = exp_pot
      pacc = np.min(pacc_list)
      randval = np.random.rand()
      if (randval > pacc):
        for el in range(pbeads_r+1):
          chain_r[el] = old_r[el]
      else:
        old_pot = new_pot

    # Loop over all the left wall choices (except the first one because it is an endpoint bead) and 
    #    sample that bead
    for mcint2 in range(1,numchoices):
      # Save primitive chain before proposal
      for el in range(pbeads_r+1):
        old_r[el] = chain_r[el]
  
      #===============================================================================================
      # (subtract one to use the bead to the left of the choice as a wall for staging)
      leftwall = lft_wall_choices[mcint2] - 1
      #===============================================================================================
      # Sample
      chain_r = do_intermediate_staging(chain_r, leftwall, staged_r, beads_sigma_sqrds, 1, pbeads_r)
      #===============================================================================================
      # Get potential from proposal
      new_pot = 0.0
      new_pot += 0.5 * 0.5 * real_mass * omga * omga * chain_r[0] * chain_r[0] / pbeads_r
      for h in range(1,pbeads_r):
        new_pot += 0.5 * real_mass * omga * omga * chain_r[h] * chain_r[h] / pbeads_r
      new_pot += 0.5 * 0.5 * real_mass * omga * omga * chain_r[pbeads_r] * chain_r[pbeads_r] / pbeads_r
      #===============================================================================================
      # Accept or reject
      exp_pot = np.exp( -beta * (new_pot - old_pot) )
      pacc_list[1] = exp_pot
      pacc = np.min(pacc_list)
      randval = np.random.rand()
      if (randval > pacc):
        # Rejected
        for el in range(pbeads_r+1):
          chain_r[el] = old_r[el]
      else:
        # Accepted
        old_pot = new_pot

    for mcint2 in range(2):
      # Save primitive chain before proposal
      for el in range(pbeads_r+1):
        old_r[el] = chain_r[el]
  
      #===============================================================================================
      # Multiply iterator by P to get first or last bead in the chain
      rand_int = mcint2*pbeads_r
      #===============================================================================================
      # Sample
      chain_r = endpoint_sampling(chain_r, rand_int, real_mass, beta, pbeads_r)
      #===============================================================================================
      # Get potential from proposal
      new_pot = 0.0
      new_pot += 0.5 * 0.5 * real_mass * omga * omga * chain_r[0] * chain_r[0] / pbeads_r
      for h in range(1,pbeads_r):
        new_pot += 0.5 * real_mass * omga * omga * chain_r[h] * chain_r[h] / pbeads_r
      new_pot += 0.5 * 0.5 * real_mass * omga * omga * chain_r[pbeads_r] * chain_r[pbeads_r] / pbeads_r
      #===============================================================================================
      # Accept or reject
      exp_pot = np.exp( -beta * (new_pot - old_pot) )
      pacc_list[1] = exp_pot
      pacc = np.min(pacc_list)
      randval = np.random.rand()
      if (randval > pacc):
        for el in range(pbeads_r+1):
          chain_r[el] = old_r[el]
      else:
        old_pot = new_pot
  
    # Grabbing samples 
    if (mcint > mc_equilibrate):
      samples[samp_counter] = chain_r[0] - chain_r[pbeads_r]
      samp_counter += 1

  histvals, hist_bin_edges = np.histogram(samples,bins=50)
  return histvals, hist_bin_edges

#======================================================================================
# Parameters
numsteps = 30000
equilibration = 10000

bigp = 32
j_r = 8
harm_mass = 1.0
harm_omga = 1.0
inverse_kbt = 1.0/10.0
#======================================================================================

# Sampling u ~ U(0,1) for the initial r position values
r_chain = np.random.rand(bigp+1)

m = 1
w = 1
dist_hist, dist_bin_edges = do_mc_open_chain(numsteps, equilibration, r_chain, bigp, j_r, harm_mass, harm_omga, inverse_kbt)

# Normalize histogram
sum1 = 0.0
for s in range(len(dist_hist)):
  sum1 += dist_hist[s] * (dist_bin_edges[1] - dist_bin_edges[0])
dist_hist = dist_hist / sum1

# Make a histogram of the open path end-to-end distance
ideal_x = np.arange(-7,7,.05)
ideal_prediction_x = get_harmonic_density(ideal_x, 1/inverse_kbt, m, w)
plt.plot(ideal_x, ideal_prediction_x,linestyle='-',label='Exact', color='blue')
plt.plot(bin_centers(dist_bin_edges), dist_hist, 'o',label=r'$N(u_{P+1})$', color='red')
plt.legend(loc='upper right')
#plt.xlim(-6, 6)
#plt.ylim(0,0.35)
plt.xlabel(r'$u_{P+1}$')
plt.ylabel(r'$N(u_{P+1})$')
plt.title('Estimator for harmonic oscillator density matrix')
plt.savefig('the_density.png')
plt.clf()