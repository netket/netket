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


def OuterProduct(x: PyTree, x2: PyTree) -> Array:
    r"""Computes A B^T where A and B are represent the pytrees of x and x2
    flattened over all but the first dimension

    Args:
        x: A pytree of arrays with equal first dimension
        x2: A pytree of arrays with dimensions to x outside of the first dimension

    Returns:
        An array of shape [len(x),len(x2)] where len measures the first dimension in the pytree
    """

    def array_outer(x, x2):
        x = x.reshape(x.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        return jnp.matmul(x, x2.conj().T)

    return tree_reduce(add, tree_map(array_outer, x, x2))


@partial(jax.jit, static_argnames=("apply_fun", "mode"))
def NeuralTangentKernel(
    apply_fun: Callable, params: PyTree, offset: PyTree, σ1: Array, σ2: Array, mode: str
) -> Array:

    r"""Computes the neural tangent kernel with respect to two sets of samples σ1 and σ2 where the jacobians are offset

    .. math ::
        N_{s,s'} = \sum_{\theta} \frac{d log(\psi_s)}{d \theta} \frac{d log(\psi_s')}{d \theta}

    Args:
        apply_fun: A function that takes a basis state s as input and returns the log amplitude of the wavefunction
        params: A PyTree with the differential parameters of apply_fun
        offset: An offset that is subtracted from each of the Jacobians.
        σ1, σ2: The samples at which the neural tangent kernel is evaluated
        mode: A specification of the differentiation properties of the function as detailed in :def:`netket.jax.jacobian`
        r_cond: A value such that all eigenvalues v below r_cond*v_max are truncated

    Returns:
        The neural tangent kernel

    """

    def subtract_mean(j, mean):
        return tree_map(lambda x, y: x - jnp.expand_dims(y, 0), j, mean)

    jac = subtract_mean(jacobian(apply_fun, params, σ1, mode=mode), offset)
    jac2 = subtract_mean(jacobian(apply_fun, params, σ2, mode=mode), offset)

    return OuterProduct(jac, jac2)
