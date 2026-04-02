import jax.interpreters.xla as xla
import jax.core
if not hasattr(xla, "pytype_aval_mappings"):
    xla.pytype_aval_mappings = jax.core.pytype_aval_mappings

# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"       # defrags GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"      # don't grab all VRAM up front
import sys
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

# print(os.listdir("../../../sampler-comparison"))

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable preallocation
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform allocator
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth

sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")

from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.experiments.benchmark import run

if __name__ == "__main__":

    

    print("Starting benchmark...")


    # models = [U1(Lt=side, Lx=side, beta=2.) for side in [512, 1024]]
    # models = [phi4(side, unreduce_lam(reduced_lam=4.0, side=side)) for side in [1024]]

    models = [
        IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log'),
        ]

   
    for model in models:
       run(
                key=jax.random.PRNGKey(4),
                models=[model],
                tuning_options=['alba'],
                mh_options = [True],
                canonical_options = [True],
                langevin_options = [False],
                integrator_type_options = ['velocity_verlet'],
                diagonal_preconditioning_options = [True],
                redo=True,
                compute_missing=True,
                redo_bad_results=True,
                pseudofermion=False,
            )
   