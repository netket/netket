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

from .initializers import zeros, ones

from .utils import (
    to_array,
    to_matrix,
    split_array_mpi,
    update_dense_symm,
    binary_encoding,
    states_to_numbers,
)

from .deprecation import (
    Dense,
    DenseGeneral,
    Conv,
    ConvTranspose,
    Embed,
    SelfAttention,
    dot_product_attention,
    make_attention_mask,
    make_causal_mask,
    combine_masks,
)

from . import blocks


# TODO: Eventually remove These (deprecated in 3.5)
# These were never supposed to be re-exported, but they slipped and I used them in quite
# some tutorials so we should keep them for a long time.
_deprecated_names = ["Module", "compact"]


from netket.utils import warn_deprecation as _warn_deprecation


def __getattr__(name):
    import sys

    if name in _deprecated_names:
        _warn_deprecation(
            f" \n"
            f" \n"
            f"          =======================================================================\n"
            f"          `nk.nn.{name}` is deprecated. Use `flax.linen.{name}` directly instead.\n"
            f"          =======================================================================\n"
            f" \n"
            f"If you imported `flax.linen as nn`, as is customary to do, you can use `nn.{name}` "
            f"directly. There are no functionality changes.\n"
        )
        import flax

        return getattr(flax.linen, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")
