import jax
from jax import numpy as jnp

from functools import partial

from flax.linen.initializers import *

lecun_complex = partial(variance_scaling, 1.0, "fan_in", "normal", dtype=jnp.complex64)
