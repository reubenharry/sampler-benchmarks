# helper function to make a model with the same structure as inference gym models

from collections import namedtuple

SampleTransformation = namedtuple("SampleTransformation", ["ground_truth_mean", "ground_truth_standard_deviation", "fn"])

Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector", "sample_transformations", "exact_sample", "name", "sample_init" ])


def make_model(logdensity_fn, ndims, name, default_event_space_bijector, sample_transformations={}, exact_sample=None, sample_init=None):
    return Model(
        ndims = ndims,
        log_density_fn=logdensity_fn,
        default_event_space_bijector=default_event_space_bijector,
        name=name,
        sample_transformations=sample_transformations,
        exact_sample=exact_sample,
        sample_init=sample_init
    )

# def make_model(logdensity_fn, ndims, name, default_event_space_bijector, x_ground_truth_mean, x_ground_truth_std, x2_ground_truth_mean, x2_ground_truth_std, exact_sample=None, sample_init=None):
#     return Model(
#         ndims = ndims,
#         log_density_fn=logdensity_fn,
#         default_event_space_bijector=default_event_space_bijector,
#         sample_transformations={
#             "identity": SampleTransformation(ground_truth_mean=x_ground_truth_mean, ground_truth_standard_deviation=x_ground_truth_std),
#             "square": SampleTransformation(ground_truth_mean=x2_ground_truth_mean, ground_truth_standard_deviation=x2_ground_truth_std),
#         },
#         exact_sample=exact_sample,
#         name=name,
#         sample_init=sample_init
#     )
