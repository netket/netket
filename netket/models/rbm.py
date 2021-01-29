from typing import Union, Optional, Tuple, Any, Callable, Iterable

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn

from netket import nn as nknn

from netket.hilbert import AbstractHilbert
from netket.graph import AbstractGraph


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any


class RBM(nn.Module):
    dtype: Any = np.float64
    activation: Any = nknn.logcosh
    alpha: Union[float, int] = 1
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        x = nknn.Dense(
            name="Dense",
            features=self.alpha * x.shape[-1],
            dtype=self.dtype,
            use_bias=self.use_bias,
        )(x)
        x = self.activation(x)
        return jnp.sum(x, axis=-1)


class RBMModPhase(nn.Module):
    dtype: Any = np.float64
    activation: Any = nknn.logcosh
    alpha: Union[float, int] = 1
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        re = nknn.Dense(
            features=self.alpha * x.shape[-1], dtype=self.dtype, use_bias=self.use_bias
        )(x)
        re = self.activation(re)
        re = jnp.sum(re, axis=-1)

        im = nknn.Dense(
            features=self.alpha * x.shape[-1], dtype=self.dtype, use_bias=self.use_bias
        )(x)
        im = self.activation(im)
        im = jnp.sum(im, axis=-1)

        return mod + 1j * im
