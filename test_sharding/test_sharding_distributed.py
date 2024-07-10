import jax

jax.config.update("jax_cpu_enable_gloo_collectives", True)
jax.distributed.initialize()

from test_sharding import *  # noqa
