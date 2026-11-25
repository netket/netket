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
def get_local_kernel(vstate: Any, Ô: Any, chunk_size: int | None):
    """
    Returns the function computing the local estimator for the given variational
    state and operator.

    Args:
        vstate: the variational state
        Ô: the operator
        chunk_size: The optional chunk size to be used for the calculation.

    Returns:
        A callable accepting the output of `get_configs(vstate, O)`.
    """


# Default dispatch for when chunk_size is not explicitly specified.
@dispatch(precedence=-10)
def get_local_kernel(vstate: Any, Ô: Any, chunk_size: None):  # noqa: F811
    return get_local_kernel(vstate, Ô)


@dispatch.abstract
def local_estimators(vstate: Any, op: Any, chunk_size: int | None):
    """
    Compute per-sample local estimator data for operator op on vstate.

    Returns a LocalEstimators (scalar, data shape (n_chains, chain_len)) or a
    LocalEstimatorsBatch (K-channel, data shape (n_chains, chain_len, K)).

    The default dispatch uses get_local_kernel_arguments + get_local_kernel and
    returns a scalar LocalEstimators. Override this for operators that do not fit
    the kernel pattern (e.g. variance, infidelity).
    """


@dispatch(precedence=-100)
def local_estimators(vstate: Any, op: Any, chunk_size: Any):  # noqa: F811
    raise NotImplementedError(
        f"local_estimators is not implemented for the combination of vstate type "
        f"{type(vstate).__name__} and operator type {type(op).__name__}.\n"
        f"To add support, register a dispatch. For MCState + custom "
        f"AbstractOperator, define separate chunk_size=None and chunk_size=int "
        f"overloads to avoid ambiguity:\n"
        f"    @nk.vqs.mc.local_estimators.dispatch\n"
        f"    def _(vstate: YourState, op: YourOp, chunk_size: None):\n"
        f"        ...\n"
        f"    @nk.vqs.mc.local_estimators.dispatch\n"
        f"    def _(vstate: YourState, op: YourOp, chunk_size: int):\n"
        f"        ..."
    )


@jax.jit
def force_to_grad(Ō_grad, parameters):
    """
    Converts the forces vector F_k = cov(O_k, E_loc) into the gradient used by the
    optimizers, g_k = dE/dx_k + i dE/dy_k for theta_k = x_k + i y_k.

    For both real and complex (holomorphic) parameters this is a factor 2:

      - complex parameters:  g = 2 F
      - real parameters:     g = 2 Re[F]

    so that a wavefunction parametrised by a complex leaf and the same wavefunction
    parametrised by two real leaves receive identical updates.

    Note that this is the conjugate of jax.grad's convention for f: C -> R; it is the
    one for which `params + optax_update` is correct gradient descent.
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
