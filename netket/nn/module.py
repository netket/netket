import flax
from flax import linen as nn
from flax import struct
import jax
from jax import numpy as jnp

from netket.hilbert import AbstractHilbert
from typing import Union, Tuple, Optional, Any, Callable

from jax.random import PRNGKey

from flax.linen import Module


class JaxWrapModule(nn.Module):
    """
    Wrapper for Jax bare modules made by a init_fun and apply_fun
    """

    init_fun: Callable
    apply_fun: Callable

    @nn.compact
    def __call__(self, x):
        if jnp.ndim(x) == 1:
            x = jnp.atleast_1d(x)
        pars = self.param(
            "jax", lambda rng, shape: self.init_fun(rng, shape)[1], x.shape
        )

        return self.apply_fun(pars, x)


def wrap_jax(mod):
    """
    Wrap a Jax module into a flax module
    """
    return JaxWrapModule(*mod)
