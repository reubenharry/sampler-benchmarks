import numpy as np
import matplotlib.pyplot as plt

def get_sigma_vals(inv_kt, bead_omega, mass, j, sigma_sqrds):
  for k in range(1,j+1):
    sigma_sqrds[k-1] = k / (inv_kt * (k+1) * mass * bead_omega * bead_omega)
  return sigma_sqrds

def partial_chain_u_to_x(prims, num_beads, rand_bead, j, new_chain):
  for k in range(j,0,-1):
    prims[rand_bead + k] = new_chain[rand_bead + k] + ( k*prims[rand_bead + k + 1] / (k+1)) + (prims[rand_bead] / (k+1) )
  return prims 

def do_intermediate_staging(chain_pos,  left_wall, staged, sigma_sqrds, j, big_p):
  for k in range(j):
    staged[left_wall + k + 1] = np.random.normal(0, np.sqrt(sigma_sqrds[k]))
  chain_pos = partial_chain_u_to_x(chain_pos, big_p, left_wall, j, staged)
  return staged

def endpoint_sampling(chain_pos, beadval, real_mass, inv_kt, big_p):
  if (beadval == 0):
    chain_pos[0] = np.random.normal(chain_pos[1], np.sqrt(inv_kt / (real_mass*big_p)) )
  else:
    chain_pos[big_p] = np.random.normal(chain_pos[big_p-1], np.sqrt(inv_kt / (real_mass*big_p)) )
  return chain_pos

def bin_centers(bin_edges):
    return (bin_edges[1:]+bin_edges[:-1])/2.

def get_harmonic_density(pos, inv_temp, mass, omega):
    normalization = np.sqrt((mass*omega) / (4*np.pi*np.tanh(inv_temp*omega/2)))
    exp_constant = np.pi * normalization**2
    return normalization * np.exp(-exp_constant * pos**2 )

#======================================================================================
# Parameters
mc_steps = 300
mc_equilibrate = 1000

pbeads_r = 300
jval_r = 60
real_mass = 1.0
omga = 1.0
kbt = 1.0/10.0
#======================================================================================

beta = 1 / kbt
chain_r = np.random.normal(0, np.sqrt(kbt), pbeads_r+1)

#complex_t = complex(real_t, -(0.5*beta))
#abs_complex_t_sqrd = np.abs(complex_t)**2
#omega_p = np.sqrt(pbeads_r / abs_complex_t_sqrd)
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
lft_wall_choices = np.zeros(numchoices, dtype=int)
for h in range(1,numchoices+1):
  lft_wall_choices[h-1] = h*(jval_r+1) + 1
# If we choose the last bead in lft_wall_choices and sample j beads to the right, we may only be able to 
#  sample alt_jval_r beads where alt_jval_r <= jval_r. For example when P = 13 and j = 4, alt_jval_r = 2
alt_jval_r = pbeads_r - lft_wall_choices[numchoices-1]
# (subtracting 1 because python has 0 based indexing)
lft_wall_choices -= 1

samples = []
for mcint in range(1,mc_steps+1):
  # moved is an array and all_moved is a boolean I use to determine whether specific beads are 
  #   sampled or not
  moved = np.zeros(pbeads_r+1)
  all_moved = False
  while (all_moved == False):
    # Save primitive chain before proposal
    old_r = chain_r

    #===============================================================================================
    # Pick a random bead from the list of choices to act as a wall and sample j beads to the left
    #  or right 
    leftwall = lft_wall_choices[np.random.randint(numchoices)]
    # Get direction    0 = left  1 = right
    direction = np.random.randint(2)
    # If going to the left, leftwall is adjusted
    if (direction == 0):
      leftwall -= (jval_r + 1)
    #===============================================================================================
    # Sample beads
    if (leftwall == lft_wall_choices[numchoices-1] and direction == 1):
      for h in range(alt_jval_r):
        moved[leftwall + h + 1] = 1
      do_intermediate_staging(chain_r, leftwall, staged_r, beads_sigma_sqrds, alt_jval_r, pbeads_r)
    else:
      for h in range(jval_r):
        moved[leftwall + h + 1] = 1
      do_intermediate_staging(chain_r, leftwall, staged_r, beads_sigma_sqrds, jval_r, pbeads_r)
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
    pacc = np.min([1.0, exp_pot])
    if (np.random.rand() > pacc):
      chain_r = old_r
    else:
      old_pot = new_pot
      # Recording which beads got moved
      if ( leftwall == lft_wall_choices[numchoices-1] and direction == 1):
        for h in range(alt_jval_r):
          moved[leftwall + h + 1] = 1
      else:
        for h in range(jval_r):
          moved[leftwall + h + 1] = 1
    # Check if done with while loop
    moved_chek = 0
    for h in range(1,pbeads_r):
      if (moved[h] == 1):
        moved_chek += 1
    if (moved_chek == pbeads_r + 1 - 2 - numchoices):
      all_moved = True
   
  all_moved = False
  while (all_moved == False):
    # Save primitive chain before proposal
    old_r = chain_r
 
    #===============================================================================================
    # Pick a random bead from the list of choices and sample that one bead 
    #    (subtract one to use the bead to the left of the choice as a wall for staging)
    leftwall = lft_wall_choices[np.random.randint(numchoices)] - 1
    #===============================================================================================
    # Sample
    do_intermediate_staging(chain_r, leftwall, staged_r, beads_sigma_sqrds, 1, pbeads_r)
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
    pacc = np.min([1.0, exp_pot])
    if (np.random.rand() > pacc):
      # Rejected
      chain_r = old_r
    else:
      # Accepted
      old_pot = new_pot
      # Recording which bead got moved
      moved[leftwall + 1] = 1
    # Check if done with while loop
    moved_chek = 0
    for h in range(1,pbeads_r):
      if (moved[h] == 1):
        moved_chek += 1
    if (moved_chek == (pbeads_r - 1)):
      all_moved = True

  all_moved = False
  while (all_moved == False):
    # Save primitive chain before proposal
    old_r = chain_r

    #===============================================================================================
    # Pick first or last bead in the chain and then sample it
    rand_int = np.random.randint(2)*pbeads_r
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
    pacc = np.min([1.0, exp_pot])
    if (np.random.rand() > pacc):
      chain_r = old_r 
    else:
      old_pot = new_pot
      # Recording which bead got moved
      moved[rand_int] = 1
    # Check if done with while loop
    moved_chek = 0
    for h in range(pbeads_r+1):
      if (moved[h] == 1):
        moved_chek += 1
    if (moved_chek == (pbeads_r + 1)):
      all_moved = True

  # Grabbing samples 
  if (mcint > mc_equilibrate):
    samples.append(chain_r[0] - chain_r[pbeads_r])

m = 1
w = 1
# Make a histogram of the open path end-to-end distance
dist_hist, dist_bin_edges = np.histogram(samples,bins=50,density=True)
ideal_x = np.arange(-10,10,.1)
ideal_prediction_x = get_harmonic_density(ideal_x, 1/kbt, m, w)
plt.plot(ideal_x, ideal_prediction_x,linestyle='-',label='Exact', color='blue')
p = plt.plot(bin_centers(dist_bin_edges), dist_hist, linestyle='--',label=r'$N(u_{P+1})$', color='red')
plt.legend(loc='upper right')
#plt.xlim(-6, 6)
#plt.ylim(0,0.35)
plt.xlabel(r'$u_{P+1}$')
plt.ylabel(r'$N(u_{P+1})$')
plt.title('Estimator for harmonic oscillator density matrix')
plt.savefig('h_density.pdf')
plt.clf()