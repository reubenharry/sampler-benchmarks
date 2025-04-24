run_benchmarks(
        models={model.name: model},
        samplers={"grid_search_hmc": partial(grid_search_hmc, num_tuning_steps=5000, integrator_type="velocity_verlet", num_chains=batch_size)},
        batch_size=batch_size,
        num_steps=400,
        save_dir=f"results/{model.name}",
        key=jax.random.key(20),
        map=lambda x : x,
        calculate_ess_corr=False,
    )