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
    logcosh,
    logsinh,
    logtanh,
)
from flax.linen import (
    MultiHeadDotProductAttention,
    SelfAttention,
    dot_product_attention,
    make_attention_mask,
    make_causal_mask,
    combine_masks,
)
from .linear import (
    Conv,
    ConvTranspose,
    Dense,
    DenseGeneral,
    DenseSymm,
    DenseEquivariant,
)
from .module import Module
from flax.linen.module import compact, enable_named_call, disable_named_call, Variable

from .initializers import zeros, ones

from flax.linen import Embed

from flax.linen import compact


def to_array(hilbert, machine, params, normalize=True):
    import numpy as np
    from jax import numpy as jnp
    from netket.utils import get_afun_if_module

    machine = get_afun_if_module(machine)

    if hilbert.is_indexable:
        xs = hilbert.all_states()
        psi = machine(params, xs)
        logmax = psi.real.max()
        psi = jnp.exp(psi - logmax)

        if normalize:
            norm = jnp.linalg.norm(psi)
            psi /= norm

        return psi
    else:
        raise RuntimeError("The hilbert space is not indexable")


def to_matrix(hilbert, machine, params, normalize=True):
    import numpy as np
    from jax import numpy as jnp
    from netket.utils import get_afun_if_module

    machine = get_afun_if_module(machine)

    if hilbert.is_indexable:
        xs = hilbert.all_states()
        psi = machine(params, xs)
        logmax = psi.real.max()
        psi = jnp.exp(psi - logmax)

        L = hilbert.physical.n_states
        rho = psi.reshape((L, L))
        if normalize:
            trace = jnp.trace(rho)
            rho /= trace

        return rho
    else:
        raise RuntimeError("The hilbert space is not indexable")
