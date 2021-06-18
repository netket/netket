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


def iscomplex(x):
    return jnp.issubdtype(x.dtype, jnp.complexfloating)


def add_complex_wrap(jax_fun):
    """
    Wraps the function `jax_fun`, dispatching to the
    decorated function if the argument is complex.
    """

    def add_complex_wrap_decorator(complex_fun):
        @wraps(jax_fun)
        def wrapped_fun(x):
            if iscomplex(x):
                return jax_fun(x)
            else:
                return complex_fun(x)

        return wrapped_fun

    return add_complex_wrap_decorator


@add_complex_wrap(jax.nn.softplus)
def softplus(x):
    return jnp.log(1 + jnp.exp(x))


@add_complex_wrap(jax.nn.sigmoid)
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


@add_complex_wrap(jax.nn.swish)
def swish(x):
    return x / (1 + jnp.exp(-x))


from jax.nn import celu
from jax.nn import elu
from jax.nn import gelu
from jax.nn import glu
from jax.nn import leaky_relu
from jax.nn import log_sigmoid
from jax.nn import log_softmax
from jax.nn import normalize
from jax.nn import relu

# sigmoid
from jax.nn import soft_sign
from jax.nn import softmax

# softplus
# swish
from jax.nn import silu
from jax.nn import selu
from jax.nn import hard_tanh
from jax.nn import relu6
from jax.nn import hard_sigmoid
from jax.nn import hard_swish

from jax.numpy import tanh
from jax.numpy import cosh
from jax.numpy import sinh


def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def log_sinh(x):
    return jax.numpy.log(jax.numpy.sinh(x))


def log_tanh(x):
    return jax.numpy.log(jax.numpy.tanh(x))


# TODO: DEPRECATION 3.1
@deprecated("Deprecated. Use log_cosh instead")
def logcosh(x):
    return log_cosh(x)


# TODO: DEPRECATION 3.1
@deprecated("Deprecated. Use log_cosh instead")
def logtanh(x):
    return log_tanh(x)


# TODO: DEPRECATION 3.1
@deprecated("Deprecated. Use log_cosh instead")
def logsinh(x):
    return log_sinh(x)
