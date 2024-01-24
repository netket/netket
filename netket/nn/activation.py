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

import functools
from functools import wraps

import jax
from jax import numpy as jnp

from jax.nn import celu
from jax.nn import elu
from jax.nn import gelu
from jax.nn import glu
from jax.nn import leaky_relu

from jax.nn import log_sigmoid
from jax.nn import log_softmax
from jax.nn import standardize
from jax.nn import relu

from jax.nn import sigmoid
from jax.nn import soft_sign
from jax.nn import softmax

from jax.nn import softplus
from jax.nn import swish
from jax.nn import silu
from jax.nn import selu
from jax.nn import hard_tanh
from jax.nn import relu6
from jax.nn import hard_sigmoid
from jax.nn import hard_swish

from jax.numpy import tanh
from jax.numpy import cosh
from jax.numpy import sinh


from netket.jax import HashablePartial
from netket.utils import deprecated_new_name as _deprecated_new_name


def reim(f):
    r"""Modifies a non-linearity to act separately on the real and imaginary parts"""

    @wraps(f)
    def reim_f(f, x):
        sqrt2 = jnp.sqrt(jnp.array(2, dtype=x.real.dtype))
        if jnp.iscomplexobj(x):
            return jax.lax.complex(f(sqrt2 * x.real), f(sqrt2 * x.imag)) / sqrt2
        else:
            return f(x)

    fun = HashablePartial(reim_f, f)

    fun.__name__ = f"reim_{f.__name__}"
    fun.__doc__ = (
        f"{f.__name__} applied separately to the real and"
        f"imaginary parts of it's input.\n\n"
        f"The docstring to the original function follows.\n\n"
        f"{f.__doc__}"
    )

    return fun


def log_cosh(x):
    """
    Logarithm of the hyperbolic cosine, implemented in a more stable way.
    """
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def log_sinh(x):
    """
    Logarithm of the hyperbolic sine.
    """
    return jax.numpy.log(jax.numpy.sinh(x))


def log_tanh(x):
    """
    Logarithm of the hyperbolic tangent.
    """
    return jax.numpy.log(jax.numpy.tanh(x))


reim_selu = reim(selu)
r"""Returns the selu non-linearity, applied separately to the real and imaginary parts"""

reim_relu = reim(relu)
r"""Returns the relu non-linearity, applied separately to the real and imaginary parts"""


# TODO: Deprecated in January 2024, remove in 2025
@_deprecated_new_name("standardize", reason="Because it is deprecated by jax as well")
def normalize(*args, **kwargs):
    return standardize(*args, **kwargs)
