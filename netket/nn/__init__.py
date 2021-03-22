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
from .linear import Conv, ConvTranspose, Dense, DenseGeneral, DenseSymm
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
