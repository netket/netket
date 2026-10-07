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
from functools import partial

import jax
import jax.numpy as jnp

from netket.utils.types import PyTree, Array
import netket.jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket.jax.sharding import sharding_decorator


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
            chunk_size=max(1, chunk_size // max(1, O.max_conn_size)),
            pvary_argnums=(0,),
        )
        return local_value_chunked(pars, σ, O)
    else:
        return local_value_kernel_jax_conn_chunked(logpsi, pars, σ, O, chunk_size)


## Flattened kernel for jax operators, skipping the zero connected elements.

_FLATTENED_N_CHUNKS_UNCHUNKED = 16
"""Number of chunks the padded connected configurations are split into by
:func:`local_value_kernel_jax_flattened` when no ``chunk_size`` is given."""


def local_value_kernel_jax_flattened(
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

    :func:`local_value_kernel_jax` evaluates the network on all
    ``n_samples × max_conn_size`` configurations returned by
    :meth:`~netket.operator.DiscreteJaxOperator.get_conn_padded`, including
    the diagonal :math:`x' = x` and the padding entries, which NetKet's
    operators fill with :math:`x` and a zero matrix element. This kernel
    instead, on every device:

    - uses :math:`\log\psi(x)` for the connected configurations equal to
      :math:`x`, without evaluating the network;
    - compacts the flattened mask of the other ones with
      :func:`jax.numpy.nonzero` into an index buffer of static size
      ``n_local_samples × max_conn_size``;
    - evaluates :math:`\log\psi` on the selected configurations in chunks of
      ``chunk_size`` configurations (``n_local_samples × max_conn_size / 16``
      if ``chunk_size`` is None), and the remaining ones in chunks of
      decreasing powers of two down to ``min_chunk_size``; the last chunk is
      padded to ``min_chunk_size``. The number of chunks depends on the number
      of selected configurations, so it changes from batch to batch, but their
      shapes are static: a varying number of connected elements never triggers
      a recompilation;
    - gathers the values back to the padded layout and computes the local
      values with the same summation as :func:`local_value_kernel_jax`.

    The full chunks are evaluated in blocks of :math:`2^k` chunks, one for
    every bit of their number, and every chunk smaller than ``chunk_size``
    at most once, each block or chunk selected with a :func:`jax.lax.cond`.
    Every loop has a static number of iterations: on GPU, a loop whose number
    of iterations depends on the data waits at every iteration for the device
    to send the loop condition back to the host, which can cost more than a
    small network. A :func:`jax.lax.cond` waits once, so there are only
    logarithmically many waits. Every chunk size is compiled separately, so
    ``min_chunk_size`` trades the padding of the last chunk for the
    compilation time: ``min_chunk_size=1`` pads nothing,
    ``min_chunk_size=chunk_size`` compiles a single chunk size.

    The kernel can be differentiated in forward and reverse mode. Like a
    :func:`jax.lax.scan` over all the chunks, the reverse mode stores the
    intermediate values of the network for all the
    ``n_local_samples × max_conn_size`` connected configurations, also for
    the chunks that are skipped.

    The selection only depends on the configurations, not on the values of
    the matrix elements: a connected configuration whose matrix element is
    zero for the current coefficients of the operator is still evaluated, so
    that the derivatives with respect to those coefficients are correct.

    No step accumulates several terms in an order that can change from call
    to call (unlike a scatter-add on GPU), so the result is deterministic.

    The kernel runs per device (inside :func:`jax.shard_map` over the sample
    axis), because the compaction is a global operation that GSPMD would
    otherwise replicate on every device.

    The result agrees with :func:`local_value_kernel_jax` up to the rounding
    of :math:`\log\psi`, which can depend on the batch it is evaluated in.

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
        _local_value_kernel_jax_flattened,
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


def _local_value_kernel_jax_flattened(
    logpsi, pars, σ, O, *, chunk_size, min_chunk_size
):
    # Runs on the samples of a single device.
    n_samples, N = σ.shape
    σp, mels = O.get_conn_padded(σ)
    max_conn_size = mels.shape[-1]
    n_conns = n_samples * max_conn_size

    # The network is called in many places below, on a few shapes of chunks:
    # as a jitted function, it is traced and lowered once per shape instead of
    # once per call, which takes most of the compilation time of large models.
    logpsi = jax.jit(logpsi)

    if n_conns == 0:
        # No connected elements (e.g. an empty operator): the local values are
        # zero, and the chunks below cannot be traced on an empty buffer. The
        # zeros are built from σ so that they are varying inside shard_map.
        dtype = jnp.result_type(mels.dtype, jax.eval_shape(logpsi, pars, σ).dtype)
        return jnp.zeros_like(σ[:, 0], dtype=dtype)

    # The connected configurations equal to σ (the diagonal and the padding)
    # are found by comparison, so that we do not rely on the operator placing
    # them at a given position.
    is_conn = jnp.any(σp != jnp.expand_dims(σ, -2), axis=-1).reshape(-1)

    if chunk_size is None:
        chunk_size = -(-n_conns // _FLATTENED_N_CHUNKS_UNCHUNKED)
    chunk_size = max(1, min(chunk_size, n_conns))
    if min_chunk_size is None:
        min_chunk_size = chunk_size // 8
    min_chunk_size = max(1, min(min_chunk_size, chunk_size))
    if min_chunk_size < chunk_size:
        min_chunk_size = 1 << (min_chunk_size.bit_length() - 1)
    n_chunks_max = -(-n_conns // chunk_size)
    # The padded last chunk can end after the last full chunk.
    buffer_size = n_chunks_max * chunk_size + min_chunk_size

    # Indices of the selected configurations, sample-major. The filling only
    # needs to be a valid configuration.
    n_conn = is_conn.sum()
    (idx,) = jnp.nonzero(is_conn, size=buffer_size, fill_value=n_conns - 1)

    logpsi_σ = nkjax.apply_chunked(
        logpsi, in_axes=(None, 0), chunk_size=chunk_size, axis_0_is_sharded=False
    )(pars, σ)
    logpsi_conn = _flattened_logpsi(
        logpsi,
        chunk_size,
        min_chunk_size,
        n_chunks_max,
        pars,
        σp.reshape(-1, N),
        idx,
        n_conn,
    )

    # Gather the values back to the padded layout, from the position of every
    # selected configuration in the compact buffer. No terms are accumulated in
    # the forward pass, and in the backward pass (a scatter-add) every entry
    # receives at most one nonzero term, so the result does not depend on the
    # order of the updates. (A scatter of complex numbers is also very slow on
    # GPU.)
    position = jnp.maximum(jnp.cumsum(is_conn) - 1, 0)
    logpsi_σp = jnp.where(
        is_conn, logpsi_conn[position], jnp.repeat(logpsi_σ, max_conn_size)
    ).reshape(n_samples, max_conn_size)
    return jnp.sum(mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def _flattened_logpsi(
    logpsi, chunk_size, min_chunk_size, n_chunks_max, pars, σp, idx, n_conn
):
    # log ψ of the first n_conn selected configurations σp[idx]. The entries
    # after them are zero or hold the values of the padding of the last chunk.
    dtype = jax.eval_shape(logpsi, pars, σp[:1]).dtype
    # Built from the data so that, inside shard_map, it is varying over the
    # sample axis like the updates.
    out = jnp.zeros_like(idx, dtype=dtype)

    def evaluate(size, n_chunks, out, start):
        # n_chunks consecutive chunks of size configurations, from start.
        def body(c, out):
            offset = start + c * size
            i = jax.lax.dynamic_slice_in_dim(idx, offset, size)
            return jax.lax.dynamic_update_slice_in_dim(
                out, logpsi(pars, σp[i]), offset, axis=0
            )

        if n_chunks == 1:
            return body(0, out)
        return jax.lax.fori_loop(0, n_chunks, body, out)

    def evaluate_if(pred, size, n_chunks, out, start):
        return jax.lax.cond(
            pred,
            partial(evaluate, size, n_chunks),
            lambda out, start: out,
            out,
            start,
        )

    # The full chunks, in blocks of 2^k chunks for every bit k of their number.
    # Blocks of 1, 2, ..., 2^(t-1) chunks cover up to 2^t - 1 chunks, and the
    # first block covers the ones above, so that the blocks hold n_chunks_max
    # chunks in total: the reverse-mode derivative stores the values of every
    # block, also of those that are skipped.
    n_full = n_conn // chunk_size
    t = (n_chunks_max + 1).bit_length() - 1
    n_first = n_chunks_max - (2**t - 1)
    if n_first > 0:
        has_first = n_full >= 2**t
        out = evaluate_if(has_first, chunk_size, n_first, out, 0)
        first = jnp.where(has_first, n_first, 0)
    else:
        first = 0
    n_other = n_full - first
    for k in reversed(range(t)):
        start = (first + ((n_other >> (k + 1)) << (k + 1))) * chunk_size
        out = evaluate_if(((n_other >> k) & 1) == 1, chunk_size, 1 << k, out, start)

    # The configurations after the full chunks, in chunks of the powers of two
    # smaller than chunk_size, down to min_chunk_size, for every bit of their
    # number.
    n_tail = n_conn - n_full * chunk_size
    for k in reversed(range((chunk_size - 1).bit_length())):
        size = 1 << k
        if size < min_chunk_size:
            break
        start = n_full * chunk_size + ((n_tail >> (k + 1)) << (k + 1))
        out = evaluate_if(((n_tail >> k) & 1) == 1, size, 1, out, start)

    # The last configurations, fewer than min_chunk_size, in a padded chunk.
    if min_chunk_size > 1:
        start = n_full * chunk_size + (n_tail // min_chunk_size) * min_chunk_size
        out = evaluate_if(n_tail % min_chunk_size > 0, min_chunk_size, 1, out, start)
    return out
