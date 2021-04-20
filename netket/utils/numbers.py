from numbers import Number

from typing import Any
from plum import dispatch, Union

import numpy as np
import jax
import jaxlib

# TODO: when next version of Plum is released, use typing.Union and Union[...]
ArrayT = Union(np.ndarray, jaxlib.xla_extension.DeviceArray, jax.core.Tracer)


@dispatch.annotations()
def dtype(x: Number):
    return type(x)


@dispatch.annotations()
def dtype(x: ArrayT):
    return x.dtype


@dispatch.annotations()
def is_scalar(x: object):
    return False


@dispatch.annotations()
def is_scalar(x: Number):
    return True


@dispatch.annotations()
def is_scalar(x: ArrayT):
    return x.ndim == 0
