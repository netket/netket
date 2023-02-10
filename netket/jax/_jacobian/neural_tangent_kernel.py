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

from os import truncate
from jax import numpy as jnp

import jax
from netket.utils.types import Array, Callable, PyTree
import netket.jax as nkjax
from operator import add
from jax.tree_util import tree_map, tree_reduce
from functools import partial

from .logic import jacobian


def OuterProduct(x: PyTree) -> Array:
    r"""Computes A A^T where A is stored as a pytree of parameters

    Args:
        x: A pytree of arrays with equal first dimension dim0

    Returns:
        An array of shape [dim0,dim0] summed over all other dimensions
        in the pytree arrays
    """

    def array_outer(x):
        x = x.reshape(x.shape[0], -1)
        return jnp.matmul(x.conj(), x.T)

    return tree_reduce(add, tree_map(array_outer, x))


@partial(jax.jit, static_argnames=("apply_fun", "mode"))
def NeuralTangentKernel(
    
    r"""Computes the neural tangent kernel which is defined as follows
    
    .. math ::
        N_{s,s'} = \sum_{\theta} \frac{d log(\psi_s)}{d \theta} \frac{d log(\psi_s')}{d \theta}

    Args:
        apply_fun: A function that takes a basis state s as input and returns the log amplitude of the wavefunction
        params: A PyTree with the differential parameters of apply_fun
        samples: The samples s at which the neural tangent kernel is evaluated
        mode: A specification of the differentiation properties of the function as detailed in :def:`netket.jax.jacobian`
        r_cond: A value such that all eigenvalues v below r_cond*v_max are truncated 

    Returns:
        The neural tangent kernel
    """

    
    apply_fun: Callable, params: PyTree, samples: Array, mode: str
) -> Array:
    
    jac = jacobian(apply_fun, params, samples, mode=mode, center=True)
    
    return OuterProduct(jac)



@partial(jax.jit, static_argnames=("apply_fun", "mode", "r_cond"))
def NeuralTangentKernelInverse(
    apply_fun: Callable,
    params: PyTree,
    samples: Array,
    mode: str,
    r_cond: float = 1e-12,
) -> Array:
    
    
    r"""Computes the pseudo-inverse of the neural tangent kernel which is defined as follows
    
    .. math ::
        N_{s,s'} = \sum_{\theta} \frac{d log(\psi_s)}{d \theta} \frac{d log(\psi_s')}{d \theta}

    Args:
        apply_fun: A function that takes a basis state s as input and returns the log amplitude of the wavefunction
        params: A PyTree with the differential parameters of apply_fun
        samples: The samples s at which the neural tangent kernel is evaluated
        mode: A specification of the differentiation properties of the function as detailed in :def:`netket.jax.jacobian`
        r_cond: A value such that all eigenvalues v below r_cond*v_max are truncated 

    Returns:
        The pseudo-inverse of the neural tangent kernel 
    """
    
  
    jac = jacobian(apply_fun, params, samples, mode=mode, center=True)
    jac = OuterProduct(jac)

    return jnp.linalg.pinv(jac, rcond=r_cond, hermitian=True)
