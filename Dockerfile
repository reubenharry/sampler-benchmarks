FROM ghcr.io/nvidia/jax:jax

RUN \
    pip install matplotlib && pip install jax_cosmo && pip install scipy && pip install pandas && pip install numpy && pip install jaxlib && pip install jaxopt && pip install fastprogress && pip install arviz && pip install tfds_nightly && pip install tf-nightly && pip install tfp_nightly[jax] && pip install tf_keras

