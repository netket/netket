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

from functools import wraps

import jax
from jax import numpy as jnp

from jax.nn import celu as _deprecated_celu
from jax.nn import elu as _deprecated_elu
from jax.nn import gelu as _deprecated_gelu
from jax.nn import glu as _deprecated_glu
from jax.nn import leaky_relu as _deprecated_leaky_relu

from jax.nn import log_sigmoid as _deprecated_log_sigmoid
from jax.nn import log_softmax as _deprecated_log_softmax
from jax.nn import standardize as _deprecated_standardize
from jax.nn import relu as _deprecated_relu

from jax.nn import sigmoid as _deprecated_sigmoid
from jax.nn import soft_sign as _deprecated_soft_sign
from jax.nn import softmax as _deprecated_softmax

from jax.nn import softplus as _deprecated_softplus
from jax.nn import swish as _deprecated_swish
from jax.nn import silu as _deprecated_silu
from jax.nn import selu as _deprecated_selu
from jax.nn import hard_tanh as _deprecated_hard_tanh
from jax.nn import relu6 as _deprecated_relu6
from jax.nn import hard_sigmoid as _deprecated_hard_sigmoid
from jax.nn import hard_swish as _deprecated_hard_swish

from jax.numpy import tanh as _deprecated_tanh
from jax.numpy import cosh as _deprecated_cosh
from jax.numpy import sinh as _deprecated_sinh


from netket.jax import HashablePartial

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr


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

    fun.__name__ = f"reim_{f.__name__}"  # type: ignore
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
    return jnp.log(jnp.sinh(x))


def log_tanh(x):
    """
    Logarithm of the hyperbolic tangent.
    """
    return jnp.log(jnp.tanh(x))


reim_selu = reim(_deprecated_selu)
r"""Returns the selu non-linearity, applied separately to the real and imaginary parts"""

reim_relu = reim(_deprecated_relu)
r"""Returns the relu non-linearity, applied separately to the real and imaginary parts"""


# Deprecation machinery for re-exported activation functions
_deprecations = {
    # September 2025, NetKet 3.20
    "celu": (
        "netket.nn.activation.celu is deprecated: use jax.nn.celu directly",
        _deprecated_celu,
    ),
    "elu": (
        "netket.nn.activation.elu is deprecated: use jax.nn.elu directly",
        _deprecated_elu,
    ),
    "gelu": (
        "netket.nn.activation.gelu is deprecated: use jax.nn.gelu directly",
        _deprecated_gelu,
    ),
    "glu": (
        "netket.nn.activation.glu is deprecated: use jax.nn.glu directly",
        _deprecated_glu,
    ),
    "leaky_relu": (
        "netket.nn.activation.leaky_relu is deprecated: use jax.nn.leaky_relu directly",
        _deprecated_leaky_relu,
    ),
    "log_sigmoid": (
        "netket.nn.activation.log_sigmoid is deprecated: use jax.nn.log_sigmoid directly",
        _deprecated_log_sigmoid,
    ),
    "log_softmax": (
        "netket.nn.activation.log_softmax is deprecated: use jax.nn.log_softmax directly",
        _deprecated_log_softmax,
    ),
    "standardize": (
        "netket.nn.activation.standardize is deprecated: use jax.nn.standardize directly",
        _deprecated_standardize,
    ),
    "relu": (
        "netket.nn.activation.relu is deprecated: use jax.nn.relu directly",
        _deprecated_relu,
    ),
    "sigmoid": (
        "netket.nn.activation.sigmoid is deprecated: use jax.nn.sigmoid directly",
        _deprecated_sigmoid,
    ),
    "soft_sign": (
        "netket.nn.activation.soft_sign is deprecated: use jax.nn.soft_sign directly",
        _deprecated_soft_sign,
    ),
    "softmax": (
        "netket.nn.activation.softmax is deprecated: use jax.nn.softmax directly",
        _deprecated_softmax,
    ),
    "softplus": (
        "netket.nn.activation.softplus is deprecated: use jax.nn.softplus directly",
        _deprecated_softplus,
    ),
    "swish": (
        "netket.nn.activation.swish is deprecated: use jax.nn.swish directly",
        _deprecated_swish,
    ),
    "silu": (
        "netket.nn.activation.silu is deprecated: use jax.nn.silu directly",
        _deprecated_silu,
    ),
    "selu": (
        "netket.nn.activation.selu is deprecated: use jax.nn.selu directly",
        _deprecated_selu,
    ),
    "hard_tanh": (
        "netket.nn.activation.hard_tanh is deprecated: use jax.nn.hard_tanh directly",
        _deprecated_hard_tanh,
    ),
    "relu6": (
        "netket.nn.activation.relu6 is deprecated: use jax.nn.relu6 directly",
        _deprecated_relu6,
    ),
    "hard_sigmoid": (
        "netket.nn.activation.hard_sigmoid is deprecated: use jax.nn.hard_sigmoid directly",
        _deprecated_hard_sigmoid,
    ),
    "hard_swish": (
        "netket.nn.activation.hard_swish is deprecated: use jax.nn.hard_swish directly",
        _deprecated_hard_swish,
    ),
    "tanh": (
        "netket.nn.activation.tanh is deprecated: use jax.numpy.tanh directly",
        _deprecated_tanh,
    ),
    "cosh": (
        "netket.nn.activation.cosh is deprecated: use jax.numpy.cosh directly",
        _deprecated_cosh,
    ),
    "sinh": (
        "netket.nn.activation.sinh is deprecated: use jax.numpy.sinh directly",
        _deprecated_sinh,
    ),
}


__getattr__ = _deprecation_getattr(__name__, _deprecations)

del _deprecation_getattr
