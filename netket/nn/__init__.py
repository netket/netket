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

# Import NetKet-specific activation functions for deprecation machinery
from .activation import (
    log_cosh as _deprecated_log_cosh,
    log_sinh as _deprecated_log_sinh,
    log_tanh as _deprecated_log_tanh,
    reim_selu as _deprecated_reim_selu,
    reim_relu as _deprecated_reim_relu,
    reim as _deprecated_reim,
)

# Import deprecated functions for deprecation machinery
from jax.nn import celu as _deprecated_celu
from jax.nn import elu as _deprecated_elu
from jax.nn import gelu as _deprecated_gelu
from jax.nn import glu as _deprecated_glu
from jax.nn import leaky_relu as _deprecated_leaky_relu
from jax.nn import log_sigmoid as _deprecated_log_sigmoid
from jax.nn import log_softmax as _deprecated_log_softmax
from jax.nn import relu as _deprecated_relu
from jax.nn import sigmoid as _deprecated_sigmoid
from jax.nn import soft_sign as _deprecated_soft_sign
from jax.nn import softmax as _deprecated_softmax
from jax.nn import softplus as _deprecated_softplus
from jax.nn import swish as _deprecated_swish
from jax.nn import silu as _deprecated_silu
from jax.numpy import tanh as _deprecated_tanh
from jax.numpy import cosh as _deprecated_cosh
from jax.numpy import sinh as _deprecated_sinh

from .symmetric_linear import (
    DenseSymm,
    DenseEquivariant,
)

from netket.nn.masked_linear import MaskedDense1D, MaskedConv1D, MaskedConv2D
from netket.nn.fast_masked_linear import (
    FastMaskedDense1D,
    FastMaskedConv1D,
    FastMaskedConv2D,
)

from netket.nn.utils import (
    to_array,
    to_matrix,
    binary_encoding,
)

from netket.nn import blocks as blocks
from netket.nn import activation as activation

from netket._src.operator.hpsi_utils import make_logpsi_op_afun as make_logpsi_op_afun


# Deprecation machinery for activation functions
_deprecations = {
    # December 2024, NetKet 3.15
    "celu": (
        "netket.nn.celu is deprecated: use jax.nn.celu directly",
        _deprecated_celu,
    ),
    "elu": (
        "netket.nn.elu is deprecated: use jax.nn.elu directly",
        _deprecated_elu,
    ),
    "gelu": (
        "netket.nn.gelu is deprecated: use jax.nn.gelu directly",
        _deprecated_gelu,
    ),
    "glu": (
        "netket.nn.glu is deprecated: use jax.nn.glu directly",
        _deprecated_glu,
    ),
    "leaky_relu": (
        "netket.nn.leaky_relu is deprecated: use jax.nn.leaky_relu directly",
        _deprecated_leaky_relu,
    ),
    "log_sigmoid": (
        "netket.nn.log_sigmoid is deprecated: use jax.nn.log_sigmoid directly",
        _deprecated_log_sigmoid,
    ),
    "log_softmax": (
        "netket.nn.log_softmax is deprecated: use jax.nn.log_softmax directly",
        _deprecated_log_softmax,
    ),
    "relu": (
        "netket.nn.relu is deprecated: use jax.nn.relu directly",
        _deprecated_relu,
    ),
    "sigmoid": (
        "netket.nn.sigmoid is deprecated: use jax.nn.sigmoid directly",
        _deprecated_sigmoid,
    ),
    "soft_sign": (
        "netket.nn.soft_sign is deprecated: use jax.nn.soft_sign directly",
        _deprecated_soft_sign,
    ),
    "softmax": (
        "netket.nn.softmax is deprecated: use jax.nn.softmax directly",
        _deprecated_softmax,
    ),
    "softplus": (
        "netket.nn.softplus is deprecated: use jax.nn.softplus directly",
        _deprecated_softplus,
    ),
    "swish": (
        "netket.nn.swish is deprecated: use jax.nn.swish directly",
        _deprecated_swish,
    ),
    "silu": (
        "netket.nn.silu is deprecated: use jax.nn.silu directly",
        _deprecated_silu,
    ),
    "tanh": (
        "netket.nn.tanh is deprecated: use jax.numpy.tanh directly",
        _deprecated_tanh,
    ),
    "cosh": (
        "netket.nn.cosh is deprecated: use jax.numpy.cosh directly",
        _deprecated_cosh,
    ),
    "sinh": (
        "netket.nn.sinh is deprecated: use jax.numpy.sinh directly",
        _deprecated_sinh,
    ),
    # NetKet-specific activation functions that should point to netket.nn.activation
    "log_cosh": (
        "netket.nn.log_cosh is deprecated: use netket.nn.activation.log_cosh",
        _deprecated_log_cosh,
    ),
    "log_sinh": (
        "netket.nn.log_sinh is deprecated: use netket.nn.activation.log_sinh",
        _deprecated_log_sinh,
    ),
    "log_tanh": (
        "netket.nn.log_tanh is deprecated: use netket.nn.activation.log_tanh",
        _deprecated_log_tanh,
    ),
    "reim_selu": (
        "netket.nn.reim_selu is deprecated: use netket.nn.activation.reim_selu",
        _deprecated_reim_selu,
    ),
    "reim_relu": (
        "netket.nn.reim_relu is deprecated: use netket.nn.activation.reim_relu",
        _deprecated_reim_relu,
    ),
    "reim": (
        "netket.nn.reim is deprecated: use netket.nn.activation.reim",
        _deprecated_reim,
    ),
}


from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

__getattr__ = _deprecation_getattr(__name__, _deprecations)

del _deprecation_getattr
