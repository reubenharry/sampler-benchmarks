import jax

# crude annealing scheme

identity = lambda x:x # why tf is this not in the python standard library

def annealed(sampler, beta_schedule, intermediate_num_steps):
    def s(model, num_steps, initial_position, key):

        old_bijector = model.default_event_space_bijector
        base_density = model.log_density_fn

        model =model._replace(
            default_event_space_bijector = identity
        )

        for i, beta in enumerate(beta_schedule):
            key = jax.random.fold_in(key, i)

            model = model._replace(
                log_density_fn = lambda x: base_density(x) * beta
            )
        
            samples, _ = sampler(num_tuning_steps=1000, return_samples=True, return_only_final=False)(model, intermediate_num_steps, initial_position, key)


            initial_position = samples[-1]


        model =model._replace(
            default_event_space_bijector = old_bijector
        )
        model = model._replace(
            log_density_fn = base_density
        )

        key = jax.random.fold_in(key, i+1)
        return sampler(num_tuning_steps=1000, return_samples=False)(model, num_steps, initial_position, key)
    return s