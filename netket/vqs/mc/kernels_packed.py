# Copyright 2026 The NetKet Authors - All rights reserved.
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
The packed local-value kernel of jax operators, which evaluates the network only
on the connected configurations that differ from the sample.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from netket.utils.types import PyTree, Array
import netket.jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket.jax.sharding import sharding_decorator
from netket.jax._apply_packed import _N_CHUNKS_UNCHUNKED


def local_value_kernel_jax_packed(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: int | None = None,
    min_chunk_size: int | None = None,
):
    r"""
    local_value kernel for MCState and jax-compatible operators that evaluates
    :math:`\log\psi` only on the connected configurations :math:`x' \neq x`.

    :func:`~netket.vqs.mc.kernels.local_value_kernel_jax` evaluates the network
    on all ``n_samples × max_conn_size`` configurations returned by
    :meth:`~netket.operator.DiscreteJaxOperator.get_conn_padded`, including
    the diagonal :math:`x' = x` and the padding entries, which NetKet's
    operators fill with :math:`x` and a zero matrix element. This kernel
    instead, on every device:

    - uses :math:`\log\psi(x)` for the connected configurations equal to
      :math:`x`, without evaluating the network;
    - evaluates :math:`\log\psi` on the other ones with
      :func:`netket.jax.apply_packed`, packed into chunks of ``chunk_size``
      configurations (``n_local_samples × max_conn_size / 16`` if
      ``chunk_size`` is None) and of decreasing powers of two down to
      ``min_chunk_size``. Their number changes from batch to batch, but their
      shapes are static: a varying number of connected elements never
      triggers a recompilation, and there are only logarithmically many
      :func:`jax.lax.cond`;
    - computes the local values with the same summation as
      :func:`~netket.vqs.mc.kernels.local_value_kernel_jax`.

    ``chunk_size`` is the number of configurations on which the network is
    evaluated at once. :func:`~netket.vqs.mc.kernels.local_value_kernel_jax_chunked`
    instead evaluates the connected configurations of
    ``chunk_size // max_conn_size`` samples at once, which can be as few as
    half as many: to bound the memory of the network to the same value, pass
    ``(chunk_size // max_conn_size) * max_conn_size``.

    The selection only depends on the configurations, not on the values of
    the matrix elements: a connected configuration whose matrix element is
    zero for the current coefficients of the operator is still evaluated, so
    that the derivatives with respect to those coefficients are correct.

    The kernel can be differentiated in forward and reverse mode. Like a
    :func:`jax.lax.scan` over all the chunks, the reverse mode stores the
    intermediate values of the network for all the selected connected
    configurations, also for the chunks that are skipped.

    No step accumulates several terms in an order that can change from call
    to call (unlike a scatter-add on GPU), so the result is deterministic.

    The kernel runs per device (inside :func:`jax.shard_map` over the sample
    axis), because the packing is a global operation that GSPMD would
    otherwise replicate on every device.

    The result agrees with :func:`~netket.vqs.mc.kernels.local_value_kernel_jax`
    up to the rounding of :math:`\log\psi`, which can depend on the batch it is
    evaluated in.

    Args:
        logpsi: the log-amplitude function.
        pars: the variables of the model.
        σ: the samples, of shape ``(n_samples, hilbert.size)``.
        O: the operator.
        chunk_size: the number of connected configurations on which the
            network is evaluated at once.
        min_chunk_size: the size of the smallest chunk (``chunk_size // 8``
            if None, rounded down to a power of two).
    """
    kernel = nkjax.HashablePartial(
        _local_value_kernel_jax_packed,
        logpsi,
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
    )
    # IMPORTANT: pars must be passed as explicit arg (not captured in the partial)
    # so that shard_map's pvary/pcast mechanism can give it Manual sharding.
    return sharding_decorator(
        kernel,
        sharded_args_tree=(False, True, False),
        pvary_args_tree=(True, False, False),
    )(pars, σ, O)


def _local_value_kernel_jax_packed(logpsi, pars, σ, O, *, chunk_size, min_chunk_size):
    # Runs on the samples of a single device.
    n_samples, N = σ.shape
    # The network is called in many places, on a few shapes of chunks: as a
    # jitted function, it is traced and lowered once per shape instead of once
    # per call, which takes most of the compilation time of large models.
    logpsi = jax.jit(logpsi)

    max_conn_size = O.max_conn_size
    if chunk_size is None:
        chunk_size = -(-n_samples * max_conn_size // _N_CHUNKS_UNCHUNKED)
    chunk_size = max(1, chunk_size)
    logpsi_σ = nkjax.apply_chunked(
        logpsi, in_axes=(None, 0), chunk_size=chunk_size, axis_0_is_sharded=False
    )(pars, σ)

    σp, mels = O.get_conn_padded(σ)
    # The connected configurations equal to σ (the diagonal and the padding)
    # are found by comparison, so that we do not rely on the operator placing
    # them at a given position.
    is_new = jnp.any(σp != jnp.expand_dims(σ, -2), axis=-1)
    logpsi_σp = nkjax.apply_packed(
        logpsi,
        pars,
        σp.reshape(-1, N),
        is_new.reshape(-1),
        jnp.repeat(logpsi_σ, max_conn_size),
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
    ).reshape(is_new.shape)
    return jnp.sum(mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)
