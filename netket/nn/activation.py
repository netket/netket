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

import jax as _jax
from jax import numpy as _jnp

from netket.utils import deprecated as _deprecated

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

from netket.jax import HashablePartial as _HashablePartial


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
    return x + _jnp.log1p(_jnp.exp(-2.0 * x)) - _jnp.log(2.0)


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


reim_selu = reim(selu)
r"""Returns the selu non-linearity, applied separately to the real and imaginary parts"""

reim_relu = reim(relu)
r"""Returns the relu non-linearity, applied separately to the real and imaginary parts"""


# TODO: DEPRECATION 3.1
@_deprecated("Deprecated. Use log_cosh instead")
def logcosh(x):
    return log_cosh(x)


# TODO: DEPRECATION 3.1
@_deprecated("Deprecated. Use log_tanh instead")
def logtanh(x):
    return log_tanh(x)


# TODO: DEPRECATION 3.1
@_deprecated("Deprecated. Use log_sinh instead")
def logsinh(x):
    return log_sinh(x)

# TODO: Deprecation in 3.3 : Remove in the future
_func_names = [
    "celu",
    "elu",
    "gelu",
    "glu",
    "leaky_relu",

    "log_sigmoid",
    "log_softmax",
    "normalize",
    "relu",

    "sigmoid",
    "soft_sign",
    "softmax",

    "softplus",
    "swish",
    "silu",
    "selu",
    "hard_tanh",
    "relu6",
    "hard_sigmoid",
    "hard_swish",
]

def _depmsg(func_name, new_module):
    msg = f"""
          `netket.nn.{func_name}` and `netket.nn.activation.{func_name}` are deprecated. 
          Use `{new_module}.{func_name}` instead.
          
          Now that jax correctly supports complex numbers through it's activation functions,
          please use them directly. `netket.nn.activation` will only contain activation
          funtions specific to NetKet that are not available in `jax`.
          
          This warning will turn into an error in a future version of NetKet.
          """
    return msg

for func_name in _func_names:
    locals()[func_name] = _deprecated(_depmsg(func_name, new_module="[jax.nn/nn.activation]"), func_name)(getattr(_jax.nn, func_name))

for func_name in ["cosh", "tanh", "sinh"]:
    locals()[func_name] = _deprecated(_depmsg(func_name, new_module="jax.numpy"), func_name)(getattr(_jnp, func_name))
