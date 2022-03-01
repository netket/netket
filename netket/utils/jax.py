# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import jax.numpy as jnp

from .partial import HashablePartial
from . import struct


def get_afun_if_module(mod_or_fun) -> Callable:
    """Returns the apply function if it's a module. Does nothing otherwise."""
    if hasattr(mod_or_fun, "apply"):
        return mod_or_fun.apply
    else:
        return mod_or_fun


@struct.dataclass
class WrappedApplyFun:
    """Wraps a callable to be a module-like object with the method `apply`."""

    apply: Callable
    """The wrapped callable."""

    def __repr__(self):
        return f"{type(self).__name__}(apply={self.apply}, hash={hash(self)})"


def wrap_afun(mod_or_fun):
    """Wraps a callable to be a module-like object with the method `apply`.
    Does nothing if it already has an apply method.
    """
    if hasattr(mod_or_fun, "apply"):
        return mod_or_fun
    else:
        return WrappedApplyFun(mod_or_fun)


def wrap_to_support_scalar(fun):
    """
    Wraps the flax-compatible apply function, assuming that the state input is the
    second argument, so that it always calls the wrapped function with a tensor with at
    least 2 dimensions.

    If the input was 1-dimensional, returns a scalar instead of a vector with 1 element.

    DEVNOTE: This function is used because some parts of NetKet make use of the fact
    that when we call the logψ with a single bitstring, we get a scalar out. This is
    useful when calling `jax.grad(logψ)`.
    This also makes sure that users only need to write networks that work on batches,
    and not necessarily on scalars.

    Args:
        fun: A flax-compatible function.

    Returns:
        A wrapped function, returned as an `HashablePartial` in order not to retrigger
        compilation.
    """

    def maybe_scalar_fun(apply_fun, pars, x, *args, **kwargs):
        xb = jnp.atleast_2d(x)
        res = apply_fun(pars, xb, *args, **kwargs)
        # support models with mutable state
        if isinstance(res, tuple):
            res_val = res[0]
            res_val = res_val.reshape(()) if x.ndim == 1 else res_val
            res = (res_val, res[1])
        else:
            res = res.reshape(()) if x.ndim == 1 else res
        return res

    return HashablePartial(maybe_scalar_fun, fun)
