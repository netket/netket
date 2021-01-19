import jax
from jax import numpy as jnp

from functools import partial

from jax.nn.initializers import zeros, ones, variance_scaling

lecun_normal = partial(variance_scaling, 1.0, "fan_in", "normal")

lecun_complex = partial(variance_scaling, 1.0, "fan_in", "normal", dtype=jnp.complex64)
