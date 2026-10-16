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

_CONN_CHUNK_BYTES = 2**28
"""Default size in bytes of the padded connected configurations computed at
once by :func:`local_value_kernel_jax_packed`."""


def local_value_kernel_jax_packed(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: int | None = None,
    min_chunk_size: int | None = None,
    conn_chunk_size: int | None = None,
):
    r"""
    local_value kernel for MCState and jax-compatible operators that evaluates
    :math:`\log\psi` only on the connected configurations :math:`x' \neq x`.

    .. warning::
        This function is experimental: it is public, but it can change in a
        minor release of NetKet, with a note in the changelog.

    :class:`~netket.vqs.MCState` does not use it: it is called explicitly,
    with the same arguments as
    :func:`~netket.vqs.mc.kernels.local_value_kernel_jax`, for example

    >>> import netket as nk
    >>> from netket.vqs.mc.kernels_packed import local_value_kernel_jax_packed
    >>> hi = nk.hilbert.Spin(0.5, 8)
    >>> H = nk.operator.IsingJax(hi, nk.graph.Chain(8), h=1.0)
    >>> sampler = nk.sampler.MetropolisLocal(hi)
    >>> vstate = nk.vqs.MCState(sampler, nk.models.RBM(), n_samples=64)
    >>> σ = vstate.samples.reshape(-1, hi.size)
    >>> eloc = local_value_kernel_jax_packed(
    ...     vstate._apply_fun, vstate.variables, σ, H, chunk_size=4096
    ... )
    >>> eloc.shape
    (64,)

    :func:`~netket.vqs.mc.kernels.local_value_kernel_jax` evaluates the network
    on all the configurations returned by
    :meth:`~netket.operator.DiscreteJaxOperator.get_conn_padded`, including the
    diagonal :math:`x' = x` and the padding. This kernel skips them, and
    evaluates the other ones with :func:`netket.jax.apply_packed`, which never
    recompiles when their number changes. This is faster for operators with
    many zero matrix elements, such as the Heisenberg and Hubbard models. The
    result agrees with :func:`~netket.vqs.mc.kernels.local_value_kernel_jax` up
    to rounding, and is deterministic.

    The kernel is differentiable.

    Args:
        logpsi: the log-amplitude function.
        pars: the variables of the model.
        σ: the samples, of shape ``(n_samples, hilbert.size)``.
        O: the operator.
        chunk_size: the number of connected configurations on which the
            network is evaluated at once (``n_local_samples × max_conn_size /
            16`` if None).
        min_chunk_size: the size of the smallest chunk (``chunk_size // 8``
            if None).
        conn_chunk_size: the number of connected configurations computed at
            once (as many as take 256 MiB if None), or the ``max_conn_size``
            padded connected configurations of a single sample if they are
            more.

    Returns:
        The local values, of shape ``(n_samples,)``.
    """
    kernel = nkjax.HashablePartial(
        _local_value_kernel_jax_packed,
        logpsi,
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        conn_chunk_size=conn_chunk_size,
    )
    # The kernel runs per device, inside of shard_map over the samples: the
    # packing is a global operation that GSPMD would otherwise replicate on
    # every device.
    # IMPORTANT: pars must be passed as explicit arg (not captured in the partial)
    # so that shard_map's pvary/pcast mechanism can give it Manual sharding.
    return sharding_decorator(
        kernel,
        sharded_args_tree=(False, True, False),
        pvary_args_tree=(True, False, False),
    )(pars, σ, O)


def _local_value_kernel_jax_packed(
    logpsi, pars, σ, O, *, chunk_size, min_chunk_size, conn_chunk_size
):
    # Runs on the samples of a single device.
    n_samples, N = σ.shape
    if n_samples == 0:
        # Nothing to evaluate: the empty local values, with the dtype of those
        # of a sample.
        sample = jax.ShapeDtypeStruct((1, N), σ.dtype)
        _, mels = jax.eval_shape(O.get_conn_padded, sample)
        logpsi_σ = jax.eval_shape(logpsi, pars, sample)
        return jnp.zeros((0,), jnp.result_type(mels.dtype, logpsi_σ.dtype))
    if conn_chunk_size is None:
        conn_chunk_size = _CONN_CHUNK_BYTES // (N * σ.dtype.itemsize)
    # The network is called in many places, on a few shapes of chunks: as a
    # jitted function, it is traced and lowered once per shape instead of once
    # per call, which takes most of the compilation time of large models.
    logpsi = jax.jit(logpsi)

    max_conn_size = O.max_conn_size
    # chunk_size counts the configurations evaluated at once, while
    # local_value_kernel_jax_chunked evaluates those of chunk_size //
    # max_conn_size samples, which can be as few as half as many.
    if chunk_size is None:
        chunk_size = -(-n_samples * max_conn_size // _N_CHUNKS_UNCHUNKED)
    chunk_size = max(1, chunk_size)
    logpsi_σ = nkjax.apply_chunked(
        logpsi, in_axes=(None, 0), chunk_size=chunk_size, axis_0_is_sharded=False
    )(pars, σ)

    def local_values(σ, logpsi_σ):
        σp, mels = O.get_conn_padded(σ)
        # The connected configurations equal to σ (the diagonal and the
        # padding) are found by comparison, so that we do not rely on the
        # operator placing them at a given position, and take logpsi(σ) without
        # evaluating the network. The selection does not depend on the matrix
        # elements: a configuration whose matrix element is zero for the current
        # coefficients of the operator is still evaluated, so that the
        # derivatives with respect to those coefficients are correct.
        is_new = jnp.any(σp != jnp.expand_dims(σ, -2), axis=-1)
        logpsi_σp = nkjax.apply_packed(
            logpsi,
            pars,
            σp.reshape(-1, N),
            is_new.reshape(-1),
            jnp.repeat(logpsi_σ, max_conn_size),
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            axis_0_is_sharded=False,
        ).reshape(is_new.shape)
        # The same summation as local_value_kernel_jax. No step accumulates
        # terms in an order that can change from call to call (unlike a
        # scatter-add on GPU), so that the result is deterministic.
        return jnp.sum(
            mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1
        )

    # Groups of samples whose padded connected configurations hold at most
    # conn_chunk_size configurations, all of the same size, each packed
    # separately in chunks of at most its size. This bounds the shapes of the
    # connected configurations that the kernel asks for, not the memory used by
    # XLA: the network usually takes most of it, and the operator can make
    # temporary copies of its configurations.
    group_size = max(1, min(n_samples, conn_chunk_size // max(1, max_conn_size)))
    n_groups = -(-n_samples // group_size)
    if n_groups == 1:
        return local_values(σ, logpsi_σ)
    group_size = -(-n_samples // n_groups)
    n_pad = n_groups * group_size - n_samples
    # The last group is padded with copies of the last sample.
    σ = jnp.concatenate([σ, jnp.repeat(σ[-1:], n_pad, axis=0)])
    logpsi_σ = jnp.concatenate([logpsi_σ, jnp.repeat(logpsi_σ[-1:], n_pad)])
    out = jax.lax.map(
        lambda args: local_values(*args),
        (σ.reshape(n_groups, group_size, N), logpsi_σ.reshape(n_groups, group_size)),
    )
    return out.reshape(-1)[:n_samples]
