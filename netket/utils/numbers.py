from numbers import Number

from typing import Any, Union
from plum import dispatch

import numpy as np
import jax
import jaxlib

# compatibility with jaxlib<=0.1.61
# we don't really support this old jaxlib, because previous
# versions had bugs and dont work with mpi4jax, but some people
# do use that because of old computer without AVX so...
# eventually delete this.
try:
    _DeviceArray = jaxlib.xla_extension.DeviceArray
except:
    _DeviceArray = jax.interpreters.xla._DeviceArray

ArrayT = Union[np.ndarray, _DeviceArray, jax.core.Tracer]


@dispatch
def dtype(x: Number):
    return type(x)


@dispatch
def dtype(x: ArrayT):
    return x.dtype


@dispatch
def is_scalar(x: Any):
    return False


@dispatch
def is_scalar(x: Number):
    return True


@dispatch
def is_scalar(x: ArrayT):
    return x.ndim == 0
