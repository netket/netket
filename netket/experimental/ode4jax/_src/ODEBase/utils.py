
from numbers import Number
import jax.numpy as jnp 

def strong_dtype(x):
    if isinstance(x, Number):
        x = jnp.asarray(x)

    return jnp.array(x, dtype=x.dtype)