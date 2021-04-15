from numbers import Number

from typing import Union, Any
from plum import dispatch

import numpy as np
import jax
import jaxlib

ArrayT = {np.ndarray, jaxlib.xla_extension.DeviceArray, jax.core.Tracer}


@dispatch.annotations()
def dtype(x: Number):
    return type(x)


@dispatch(ArrayT)
def dtype(x: ArrayT):
    return x.dtype


dispatch.annotations()


def is_scalar(x: Any):
    return False


@dispatch.annotations()
def is_scalar(x: Number):
    return True


@dispatch(ArrayT)
def is_scalar(x: ArrayT):
    return x.ndim == 0
