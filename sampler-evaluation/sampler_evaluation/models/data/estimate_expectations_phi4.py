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
from sampler_evaluation.models.phi4 import phi4
import numpy as np

def unreduce_lam(reduced_lam, side):
        """see Fig 3 in https://arxiv.org/pdf/2207.00283.pdf"""
        return 4.25 * (reduced_lam * np.power(side, -1.0) + 1.0)



if __name__ == "__main__":

        reduced_lam = 4.0
        for side in [512]:
                model = phi4(L=side, lam=unreduce_lam(reduced_lam=reduced_lam, side=side), load_from_file=False)
                estimate_ground_truth(model, num_samples=25000, annealing=False)