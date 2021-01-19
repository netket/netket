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
from .linear import Conv, ConvTranspose, Dense, DenseGeneral
from .module import Module
from flax.linen.module import compact, enable_named_call, disable_named_call, Variable

from .initializers import zeros, ones

from flax.linen import Embed

from flax.linen import compact

from . import models


def to_array(hilbert, machine, params, normalize=True):
    import numpy as _np
    from netket.utils import get_afun_if_module

    machine = get_afun_if_module(machine)

    if hilbert.is_indexable:
        xs = hilbert.all_states()
        psi = machine(params, xs)
        logmax = psi.real.max()
        psi = _np.exp(psi - logmax)

        if normalize:
            norm = _np.linalg.norm(psi)
            psi /= norm

        return psi
    else:
        raise RuntimeError("The hilbert space is not indexable")
