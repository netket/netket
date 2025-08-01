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

import flax as _flax

from .activation import (
    celu,
    elu,
    gelu,
    glu,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    relu,
    sigmoid,
    soft_sign,
    softmax,
    softplus,
    swish,
    silu,
    tanh,
    cosh,
    sinh,
    log_cosh,
    log_sinh,
    log_tanh,
    reim_selu,
    reim_relu,
    reim,
)

from .symmetric_linear import (
    DenseSymm,
    DenseEquivariant,
)

from .masked_linear import MaskedDense1D, MaskedConv1D, MaskedConv2D
from .fast_masked_linear import FastMaskedDense1D, FastMaskedConv1D, FastMaskedConv2D

from .utils import (
    to_array,
    to_matrix,
    binary_encoding,
)

from . import blocks

from netket._src.operator.hpsi_utils import make_logpsi_U_afun as make_logpsi_U_afun
