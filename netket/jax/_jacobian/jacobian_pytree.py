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

from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp

import numpy as np

from netket import jax as nkjax
from netket.utils import wrap_to_support_scalar
from netket.utils.types import Array, Callable, PyTree


def jacobian_real_holo(forward_fn: Callable, params: PyTree, samples: Array) -> PyTree:
    """Calculates Jacobian entries by vmapping grad.
    Assumes the function is R→R or holomorphic C→C, so single grad is enough

    Args:
        forward_fn: the log wavefunction ln Ψ
        params : a pytree of parameters p
        samples : an array of n samples σ

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """

    y, vjp_fun = jax.vjp(
        lambda pars: wrap_to_support_scalar(forward_fn)(pars, samples), params
    )
    (res,) = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    return res


def _jacobian_cplx(
    forward_fn: Callable,
    params: PyTree,
    samples: Array,
    *,
    _build_fn: Callable = partial(jax.tree_util.tree_map, jax.lax.complex),
) -> PyTree:
    """Calculates Jacobian entries by vmapping grad.
    Assumes the function is R→C, backpropagates 1 and -1j

    Args:
        forward_fn: the log wavefunction ln Ψ
        params : a pytree of parameters p
        samples : an array of n samples σ

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """

    y, vjp_fun = jax.vjp(
        lambda pars: wrap_to_support_scalar(forward_fn)(pars, samples), params
    )
    (gr,) = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    (gi,) = vjp_fun(np.array(-1.0j, dtype=jnp.result_type(y)))
    return _build_fn(gr, gi)


jacobian_cplx = partial(_jacobian_cplx, _build_fn=lambda *x: x)


def stack_jacobian_tuple(ok_re_im):
    """
    stack the real and imaginary parts of ΔOⱼₖ along the sample axis

    Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]

    Args:
        centered_oks_re_im : a tuple (ΔOᵣ, ΔOᵢ) of two PyTrees representing the real and imag part of ΔOⱼₖ
    """
    re, im = ok_re_im
    return jax.tree_util.tree_map(lambda re, im: jnp.stack([re, im], axis=0), re, im)


jacobian_real_holo_fun = jacobian_real_holo
jacobian_cplx_fun = nkjax.compose(stack_jacobian_tuple, jacobian_cplx)
