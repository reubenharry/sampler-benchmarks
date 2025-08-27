import sys
import jax
import os
jax.config.update("jax_enable_x64", True)

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()


sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")
from functools import partial
from sampler_evaluation.models import models
from sampler_comparison.samplers.general import initialize_model, make_log_density_fn
from sampler_evaluation.models.banana import banana
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.rosenbrock import Rosenbrock
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc, unadjusted_mclmc_no_tuning
from sampler_evaluation.models.neals_funnel import neals_funnel
import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import blackjax
import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
    sampler_grads_to_low_error,
)
from sampler_comparison.util import (
    calls_per_integrator_step,
    map_integrator_type_to_integrator,
)
from blackjax.adaptation.unadjusted_alba import unadjusted_alba


model= neals_funnel(ndims=5)

# model = IllConditionedGaussian(2,1)
initial_position = jnp.array([-10, 0.0, 0.0, 0.0, 0.0])

inverse_mass_matrix = jnp.ones(model.ndims)

logdensity_fn = make_log_density_fn(model)

warmup = unadjusted_alba(
            algorithm=blackjax.mclmc, 
            logdensity_fn=logdensity_fn, integrator=map_integrator_type_to_integrator["mclmc"]['mclachlan'], 
            target_eevpd=5e-4, 
            v=1., 
            num_alba_steps=5000,
            preconditioning=False,
            alba_factor=0.4,
            )
        
(blackjax_state_after_tuning, blackjax_mclmc_sampler_params), adaptation_info = warmup.run(jax.random.key(0), jax.random.normal(jax.random.key(0), (model.ndims,)), 10000)

initial_state = blackjax.mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=jax.random.key(0),
        # metric=metrics.default_metric(inverse_mass_matrix)
    )

sampler = partial(unadjusted_mclmc_no_tuning, integrator_type='mclachlan', L=blackjax_mclmc_sampler_params['L'], step_size=blackjax_mclmc_sampler_params['step_size'], inverse_mass_matrix=inverse_mass_matrix, initial_state=initial_state)

batch_size = 128

init_keys = jax.random.split(jax.random.key(3), batch_size)

keys = jax.random.split(jax.random.key(3), batch_size)


num_steps = 1000

samples, metadata = sampler(return_samples=True)(
        model=model, num_steps=num_steps, initial_position=initial_position, key=jax.random.key(3)
        )

# Convert JAX array to numpy for plotting
samples_np = np.array(samples)

# Plot MCLMC trajectory: 0th dimension vs -1th dimension
plt.figure(figsize=(8, 6))
plt.plot(samples_np[:, 0], samples_np[:, -1], 'b-', alpha=0.7, linewidth=0.5)
plt.scatter(samples_np[0, 0], samples_np[0, -1], c='red', s=100, label='Start', zorder=5)
plt.scatter(samples_np[-1, 0], samples_np[-1, -1], c='green', s=100, label='End', zorder=5)
plt.xlabel('Dimension 0 (Neck)')
plt.ylabel('Dimension -1 (Last)')
plt.title('MCLMC Trajectory in Neal\'s Funnel')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('neals_funnel_mclmc.png')
plt.close()

# Print some statistics about the trajectory
print(f"Trajectory statistics:")
print(f"Number of steps: {len(samples_np)}")
print(f"Starting position: {samples_np[0]}")
print(f"Ending position: {samples_np[-1]}")
print(f"Range in neck dimension: [{samples_np[:, 0].min():.3f}, {samples_np[:, 0].max():.3f}]")
print(f"Range in funnel dimension: [{samples_np[:, 1].min():.3f}, {samples_np[:, 1].max():.3f}]")

print("\n" + "="*50)
print("ULMC (Unadjusted Langevin Monte Carlo) Trajectory")
print("="*50)

# ULMC setup and sampling
from blackjax.adaptation.unadjusted_alba import unadjusted_alba

# Setup ULMC warmup
ulmc_warmup = unadjusted_alba(
    algorithm=blackjax.langevin, 
    logdensity_fn=logdensity_fn, 
    integrator=map_integrator_type_to_integrator["hmc"]['velocity_verlet'], 
    target_eevpd=3e-4, 
    v=jnp.sqrt(model.ndims), 
    num_alba_steps=5000,
    preconditioning=False,
    alba_factor=0.4,
)

# Run ULMC warmup
(ulmc_state_after_tuning, ulmc_sampler_params), ulmc_adaptation_info = ulmc_warmup.run(
    jax.random.key(1), 
    jax.random.normal(jax.random.key(1), (model.ndims,)), 
    10000
)

# Initialize ULMC state
ulmc_initial_state = blackjax.langevin.init(
    position=initial_position,
    logdensity_fn=logdensity_fn,
    random_generator_arg=jax.random.key(1),
)

# Create ULMC sampler
ulmc_sampler = partial(
    unadjusted_mclmc_no_tuning, 
    integrator_type='velocity_verlet', 
    L=ulmc_sampler_params['L'], 
    step_size=ulmc_sampler_params['step_size'], 
    inverse_mass_matrix=inverse_mass_matrix, 
    initial_state=ulmc_initial_state
)

# Run ULMC sampling
ulmc_samples, ulmc_metadata = ulmc_sampler(return_samples=True)(
    model=model, 
    num_steps=num_steps, 
    initial_position=initial_position, 
    key=jax.random.key(4)
)

# Convert ULMC samples to numpy for plotting
ulmc_samples_np = np.array(ulmc_samples)

# Plot ULMC trajectory: 0th dimension vs -1th dimension
plt.figure(figsize=(8, 6))
plt.plot(ulmc_samples_np[:, 0], ulmc_samples_np[:, -1], 'r-', alpha=0.7, linewidth=0.5)
plt.scatter(ulmc_samples_np[0, 0], ulmc_samples_np[0, -1], c='red', s=100, label='Start', zorder=5)
plt.scatter(ulmc_samples_np[-1, 0], ulmc_samples_np[-1, -1], c='green', s=100, label='End', zorder=5)
plt.xlabel('Dimension 0 (Neck)')
plt.ylabel('Dimension -1 (Last)')
plt.title('ULMC Trajectory in Neal\'s Funnel')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('neals_funnel_ulmc.png')
plt.close()

# Print ULMC statistics
print(f"ULMC Trajectory statistics:")
print(f"Number of steps: {len(ulmc_samples_np)}")
print(f"Starting position: {ulmc_samples_np[0]}")
print(f"Ending position: {ulmc_samples_np[-1]}")
print(f"Range in neck dimension: [{ulmc_samples_np[:, 0].min():.3f}, {ulmc_samples_np[:, 0].max():.3f}]")
print(f"Range in funnel dimension: [{ulmc_samples_np[:, 1].min():.3f}, {ulmc_samples_np[:, 1].max():.3f}]")

print("\nPlots saved as:")
print("- neals_funnel_mclmc.png (MCLMC trajectory)")
print("- neals_funnel_ulmc.png (ULMC trajectory)")








