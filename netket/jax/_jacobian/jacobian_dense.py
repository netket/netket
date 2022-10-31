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

from jax import numpy as jnp

from netket.utils.types import Array, PyTree
import netket.jax as nkjax

from .jacobian_pytree import (
    jacobian_real_holo,
    jacobian_cplx,
)


def ravel(x: PyTree) -> Array:
    """
    shorthand for tree_ravel
    """
    dense, _ = nkjax.tree_ravel(x)
    return dense


def stack_jacobian_tuple(ok_re_im):
    """
    stack the real and imaginary parts of ΔOⱼₖ along a new axis.
    First all the real part then the imaginary part.

    Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]

    Args:
        ok_re_im : a tuple (ΔOᵣ, ΔOᵢ) of two PyTrees representing the real and imag part of ΔOⱼₖ
    """
    re, im = ok_re_im
    return jnp.stack([ravel(re), ravel(im)], axis=0)


jacobian_real_holo_fun = nkjax.compose(ravel, jacobian_real_holo)
jacobian_cplx_fun = nkjax.compose(stack_jacobian_tuple, jacobian_cplx)
