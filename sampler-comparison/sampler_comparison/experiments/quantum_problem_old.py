    # exact samples
    # M, Minv, K, alpha, gamma, r = make_M_Minv_K(
    #     32, 
    #     1, 
    #     U=lambda x : 0.5*m*(omega**2)*(x**2), 
    #     r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)), 
    #     beta=beta, 
    #     hbar=hbar,
    #     m=m)


    # raw_samples = jax.random.multivariate_normal(key=jax.random.PRNGKey(0), shape=(50000,), mean=0, cov=jnp.diag(K @ Minv @ K))

    # print(raw_samples.shape)

    # samples, weights = jax.vmap(lambda x : (xi(x,r=r,U=lambda x : 0.5*m*(omega**2)*(x**2),t=t,P=P,hbar=hbar,gamma=gamma), x[i]))(raw_samples)


    # make_histograms("third", samples, weights, K, Minv, 1)


    # jax.debug.print("foo {x}",x=raw_samples.shape)

    # ixs = jax.vmap(lambda k: jax.random.choice(key=k,a=jax.numpy.arange(50000)))(jax.random.split(jax.random.key(0),50000))

    # new_samples = samples[ixs]
    # new_weights = weights[ixs]


    # jax.debug.print("bar {x}", x=jax.vmap(lambda k: jax.random.choice(k, raw_samples, axis=0,))(jax.random.split(jax.random.key(0), 50000)).shape)