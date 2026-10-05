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

from collections.abc import Callable

import jax
import jax.numpy as jnp

from netket.utils.types import PyTree, Array
import netket.jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket._src.operator.compact_conn import (
    CompactConnOperator,
    check_overflow,
    get_conn_compact,
)


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


def local_value_kernel_jax(
    logpsi: Callable, pars: PyTree, σ: Array, O: DiscreteJaxOperator
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    σp, mel = O.get_conn_padded(σ)
    logpsi_σ = logpsi(pars, σ)
    logpsi_σp = logpsi(pars, σp.reshape(-1, σp.shape[-1])).reshape(σp.shape[:-1])
    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def local_value_kernel_jax_conn_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    chunk_size: int,
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    # IMPORTANT: pars must be passed as explicit arg (not captured in lambda) so that
    # shard_map's pvary/pcast mechanism can give it Manual sharding inside shard_map.
    apply_conn = nkjax.apply_chunked(logpsi, in_axes=(None, 0), chunk_size=chunk_size)

    σp, mel = O.get_conn_padded(σ)

    logpsi_σ = apply_conn(pars, σ)
    logpsi_σp = apply_conn(pars, σp.reshape(-1, σ.shape[-1])).reshape(σp.shape[:-1])

    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


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
    chunk_size: int | None = None,
):
    """
    local_value kernel for MCState and generic operators
    """
    σp, mels = args

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    # IMPORTANT: pars must be passed as explicit arg (not captured in partial) so that
    # shard_map's pvary/pcast mechanism can give it Manual sharding inside shard_map.
    logpsi_chunked = nkjax.vmap_chunked(
        logpsi, in_axes=(None, 0), chunk_size=chunk_size
    )
    N = σ.shape[-1]

    logpsi_σ = logpsi_chunked(pars, σ.reshape((-1, N))).reshape(σ.shape[:-1] + (1,))
    logpsi_σp = logpsi_chunked(pars, σp.reshape((-1, N))).reshape(σp.shape[:-1])

    return jnp.sum(mels * jnp.exp(logpsi_σp - logpsi_σ), axis=-1)


def local_value_squared_kernel_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: int | None = None,
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
    chunk_size: int | None = None,
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


def local_value_kernel_jax_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: int | None = None,
):
    """
    local_value kernel for MCState and jaxcoompatible operators
    """
    if chunk_size >= O.max_conn_size:
        # IMPORTANT: pars must be passed as explicit arg (not captured in lambda) so that
        # shard_map's pvary/pcast mechanism can give it Manual sharding inside shard_map.
        def _local_value_kernel(pars, s, O):
            return local_value_kernel_jax(logpsi, pars, s, O)

        local_value_chunked = nkjax.apply_chunked(
            _local_value_kernel,
            in_axes=(None, 0, None),
            chunk_size=max(1, chunk_size // O.max_conn_size),
            pvary_argnums=(0,),
        )
        return local_value_chunked(pars, σ, O)
    else:
        return local_value_kernel_jax_conn_chunked(logpsi, pars, σ, O, chunk_size)


def local_value_kernel_jax_compact(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: int | None = None,
):
    r"""
    local_value kernel for MCState and jax-compatible operators declaring a
    bound :attr:`~netket.operator.DiscreteJaxOperator.max_offdiag_conn_size`
    on the number of nonzero off-diagonal connected elements of every sample.

    The nonzero off-diagonal connected elements of every sample are packed in
    ``max_offdiag_conn_size`` entries, and the network is evaluated only on
    those: ``n_samples × max_offdiag_conn_size`` configurations instead of
    ``n_samples × max_conn_size``. The diagonal elements are summed without
    evaluating the network. Every operation acts on each sample independently,
    so the kernel can be sharded along the samples without communication.

    A sample with more nonzero off-diagonal elements than the bound gets a NaN
    local value, and a :class:`RuntimeWarning` is issued.

    Args:
        logpsi: the log-amplitude function.
        pars: the variables of the model.
        σ: the samples, of shape ``(n_samples, hilbert.size)``.
        O: the operator.
        chunk_size: the number of connected configurations on which the
            network is evaluated at once.
    """
    if chunk_size is not None:
        # IMPORTANT: pars must be passed as explicit arg (not captured in lambda) so
        # that shard_map's pvary/pcast mechanism can give it Manual sharding.
        def _local_value_kernel(pars, s, O):
            return local_value_kernel_jax_compact(logpsi, pars, s, O)

        return nkjax.apply_chunked(
            _local_value_kernel,
            in_axes=(None, 0, None),
            chunk_size=max(1, chunk_size // max(1, O.max_offdiag_conn_size)),
            pvary_argnums=(0,),
        )(pars, σ, O)

    K = O.max_offdiag_conn_size
    if isinstance(O, CompactConnOperator):
        O = O.operator

    mels_diag, σp, mels, n_offdiag = get_conn_compact(O, σ, K)
    if K == 0:
        return check_overflow(mels_diag, n_offdiag, K)

    logpsi_σ = logpsi(pars, σ)
    logpsi_σp = logpsi(pars, σp.reshape(-1, σp.shape[-1])).reshape(σp.shape[:-1])
    mels_offdiag = jnp.sum(
        mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1
    )
    return check_overflow(mels_diag + mels_offdiag, n_offdiag, K)
