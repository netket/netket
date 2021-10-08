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

from netket.utils import deprecated

from jax.nn import celu
from jax.nn import elu
from jax.nn import gelu
from jax.nn import glu
from jax.nn import leaky_relu

from jax.nn import log_sigmoid
from jax.nn import log_softmax
from jax.nn import normalize
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


def reim(f):
    r"""Modifies a non-linearity to act seperately on the real and imaginary parts"""

    def reim_activation(f, x):
        sqrt2 = jnp.sqrt(jnp.array(2, dtype=x.real.dtype))
        if jnp.iscomplexobj(x):
            return jax.lax.complex(f(sqrt2 * x.real), f(sqrt2 * x.imag)) / sqrt2
        else:
            return f(x)

    return HashablePartial(reim_activation, f)


def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def log_sinh(x):
    return jax.numpy.log(jax.numpy.sinh(x))


def log_tanh(x):
    return jax.numpy.log(jax.numpy.tanh(x))


reim_selu = reim(selu)
r"""Returns the selu non-linearity, applied seperately to the real and imaginary parts"""

reim_relu = reim(relu)
r"""Returns the relu non-linearity, applied seperately to the real and imaginary parts"""


# TODO: DEPRECATION 3.1
@deprecated("Deprecated. Use log_cosh instead")
def logcosh(x):
    return log_cosh(x)


# TODO: DEPRECATION 3.1
@deprecated("Deprecated. Use log_tanh instead")
def logtanh(x):
    return log_tanh(x)


# TODO: DEPRECATION 3.1
@deprecated("Deprecated. Use log_sinh instead")
def logsinh(x):
    return log_sinh(x)
