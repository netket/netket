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

"""
This module implements some common kernels used by MCState and MCMixedState.
"""

from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp

from netket.utils.types import PyTree, Array
import netket.jax as nkjax


def batch_discrete_kernel(kernel):
    """
    Batch a kernel that only works with 1 sample so that it works with a
    batch of samples.

    Works only for discrete-kernels who take two args as inputs
    """

    def vmapped_kernel(logpsi, pars, σ, args):
        """
        local_value kernel for MCState and generic operators
        """
        σp, mels = args

        if jnp.ndim(σp) != 3:
            σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
            mels = mels.reshape(σp.shape[:-1])

        vkernel = jax.vmap(kernel, in_axes=(None, None, 0, (0, 0)), out_axes=0)
        return vkernel(logpsi, pars, σ, (σp, mels))

    return vmapped_kernel


@batch_discrete_kernel
def local_value_kernel(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCState and generic operators
    """
    σp, mel = args
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


def local_value_squared_kernel(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return jnp.abs(local_value_kernel(logpsi, pars, σ, args)) ** 2


@batch_discrete_kernel
def local_value_op_op_cost(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCMixedState and generic operators
    """
    σp, mel = args

    σ_σp = jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(σp, σ)
    σ_σ = jnp.hstack((σ, σ))
    return jnp.sum(mel * jnp.exp(logpsi(pars, σ_σp) - logpsi(pars, σ_σ)))


## Chunked versions of those kernels are defined below.


def local_value_kernel_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    """
    local_value kernel for MCState and generic operators
    """
    σp, mels = args

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    logpsi_chunked = nkjax.vmap_chunked(
        partial(logpsi, pars), in_axes=0, chunk_size=chunk_size
    )
    N = σ.shape[-1]

    logpsi_σ = logpsi_chunked(σ.reshape((-1, N))).reshape(σ.shape[:-1] + (1,))
    logpsi_σp = logpsi_chunked(σp.reshape((-1, N))).reshape(σp.shape[:-1])

    return jnp.sum(mels * jnp.exp(logpsi_σp - logpsi_σ), axis=-1)


def local_value_squared_kernel_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return (
        jnp.abs(
            local_value_kernel_chunked(logpsi, pars, σ, args, chunk_size=chunk_size)
        )
        ** 2
    )


def local_value_op_op_cost_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: Optional[int] = None,
):
    """
    local_value kernel for MCMixedState and generic operators
    """
    σp, mels = args

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    σ_σp = jax.vmap(
        lambda σpi, σi: jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(
            σpi, σi
        ),
        in_axes=(0, 0),
        out_axes=0,
    )(σp, σ)
    σ_σ = jax.vmap(lambda σi: jnp.hstack((σi, σi)), in_axes=0)(σ)

    return local_value_kernel_chunked(
        logpsi, pars, σ_σ, (σ_σp, mels), chunk_size=chunk_size
    )
