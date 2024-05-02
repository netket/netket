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

from typing import Any

import jax
import jax.numpy as jnp

from netket.hilbert import AbstractHilbert
from netket.utils.dispatch import dispatch


def check_hilbert(A: AbstractHilbert, B: AbstractHilbert):
    if not A == B:
        raise NotImplementedError(  # pragma: no cover
            f"Non matching hilbert spaces {A} and {B}"
        )


@dispatch.abstract
def get_local_kernel_arguments(vstate: Any, Ô: Any):
    """
    Returns the samples of vstate used to compute the expectation value
    of the operator O, and the connected elements and matrix elements.

    Args:
        vstate: the variational state
        Ô: the operator

    Returns:
        A Tuple with 2 elements (sigma, args), where the first elements
        should be the samples over which the classical expectation value
        should be computed, while the latter is anything that can be fed
        as input to the local_kernel.
    """


@dispatch.abstract
def get_local_kernel(vstate: Any, Ô: Any):
    """
    Returns the function computing the local estimator for the given variational
    state and operator.

    Args:
        vstate: the variational state
        Ô: the operator

    Returns:
        A callable accepting the output of `get_configs(vstate, O)`.
    """


@jax.jit
def force_to_grad(Ō_grad, parameters):
    """
    Converts the forces vector F_k = cov(O_k, E_loc) to the observable gradient.
    In case of a complex target (which we assume to correspond to a holomorphic
    parametrization), this is the identity. For real-valued parameters, the gradient
    is 2 Re[F].
    """
    Ō_grad = jax.tree_util.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )
    Ō_grad = jax.tree_util.tree_map(lambda x: 2 * x, Ō_grad)
    return Ō_grad
