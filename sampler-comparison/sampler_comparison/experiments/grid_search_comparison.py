  

L, step_size, ess, ess_avg, ess_corr_avg, rate, edge = grid_search_only_L(
        model=model,
        sampler=sampler_type,
        num_steps=models[model][sampler_type],
        num_chains=batch_size,
        integrator_type=integrator_type,
        key=keys_for_fast_grid,
        grid_size=10,
        grid_iterations=2,
        opt='max'
    )

    