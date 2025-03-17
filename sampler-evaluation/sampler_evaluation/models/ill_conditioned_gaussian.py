## TODO!

# import pickle
# import inference_gym.using_jax as gym
# from inference_gym.targets import model
# import jax.numpy as jnp

# def ill_conditioned_gaussian(ndims, gamma_shape_parameter, key):

#     ill_conditioned_gaussian = gym.targets.ill_conditioned_gaussian(ndims, gamma_shape_parameter=gamma_shape_parameter, seed=key)



#     var_x2 = 2 * jnp.square(e_x2)

#     ill_conditioned_gaussian.sample_transformations["square"] = model.Model.SampleTransformation(
#         fn=lambda params: ill_conditioned_gaussian.sample_transformations["identity"](params)
#         ** 2,
#         pretty_name="Square",
#         ground_truth_mean=e_x2,
#         ground_truth_standard_deviation=jnp.sqrt(var_x2),
#     )

#     ill_conditioned_gaussian.ndims = ndims


#     return ill_conditioned_gaussian
