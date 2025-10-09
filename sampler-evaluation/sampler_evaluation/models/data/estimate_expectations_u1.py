import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"       # defrags GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"      # don't grab all VRAM up front

# from sampler_evaluation.models import brownian_motion
import sys
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")
from sampler_evaluation.models.data.estimate_expectations import estimate_ground_truth
# import sampler_evaluation
from sampler_evaluation.models.u1 import U1
import numpy as np



if __name__ == "__main__":

        # for side in [64, 128]:
        #         model = U1(Lt=side, Lx=side, beta=2., load_from_file=False)
        #         estimate_ground_truth(model, num_samples=50000, annealing=False)
        for side in [512]:
                model = U1(Lt=side, Lx=side, beta=2., load_from_file=False)
                estimate_ground_truth(model, num_samples=5000, annealing=False)
        for side in [1024]:
                model = U1(Lt=side, Lx=side, beta=2., load_from_file=False)
                estimate_ground_truth(model, num_samples=1000, annealing=False)