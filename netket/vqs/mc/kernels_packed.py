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
from functools import partial

import jax
import jax.numpy as jnp

from netket.utils.types import PyTree, Array
import netket.jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket.jax.sharding import sharding_decorator
from netket.jax._apply_packed import _N_CHUNKS_UNCHUNKED

_CONN_CHUNK_BYTES = 2**28
"""Default size in bytes of the padded connected configurations computed at
once by :func:`local_value_kernel_jax_packed`, for the operators that cannot
compute them one at a time."""


def local_value_kernel_jax_packed(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: int | None = None,
    min_chunk_size: int | None = None,
    conn_chunk_size: int | None = None,
    generate: bool | None = None,
):
    r"""
    local_value kernel for MCState and jax-compatible operators that evaluates
    :math:`\log\psi` only on the connected configurations :math:`x' \neq x`.

    :class:`~netket.vqs.MCState` does not use it: it is called explicitly,
    with the same arguments as
    :func:`~netket.vqs.mc.kernels.local_value_kernel_jax`, for example

    .. code-block:: python

        σ = vstate.samples.reshape(-1, vstate.hilbert.size)
        eloc = local_value_kernel_jax_packed(
            vstate._apply_fun, vstate.variables, σ, H, chunk_size=4096
        )

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

    The connected configurations take at most ``conn_chunk_size``
    configurations (256 MiB if None) at once:

    - if those of all the samples take more, operators that compute their
      connected configurations one at a time
      (:meth:`~netket.operator.DiscreteJaxOperator.get_conn_at`, e.g.
      :class:`~netket.operator.LocalOperatorJax`,
      :class:`~netket.operator.IsingJax` and
      :class:`~netket.operator.PauliStringsJax`) give the matrix elements and
      which connected configurations differ from the sample
      (:meth:`~netket.operator.DiscreteJaxOperator.get_conn_mels`), and the
      configurations are computed inside of every chunk;
    - otherwise, the samples are processed in groups whose padded connected
      configurations, computed with
      :meth:`~netket.operator.DiscreteJaxOperator.get_conn_padded`, hold at
      most ``conn_chunk_size`` configurations. Every group is packed
      separately.

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
        conn_chunk_size: the number of padded connected configurations
            computed at once, when they are not computed one at a time.
        generate: whether to compute the connected configurations one at a
            time (default: if the operator supports it and those of all the
            samples do not fit in ``conn_chunk_size``).
    """
    kernel = nkjax.HashablePartial(
        _local_value_kernel_jax_packed,
        logpsi,
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        conn_chunk_size=conn_chunk_size,
        generate=generate,
    )
    # IMPORTANT: pars must be passed as explicit arg (not captured in the partial)
    # so that shard_map's pvary/pcast mechanism can give it Manual sharding.
    return sharding_decorator(
        kernel,
        sharded_args_tree=(False, True, False),
        pvary_args_tree=(True, False, False),
    )(pars, σ, O)


def generates_connected(O: DiscreteJaxOperator) -> bool:
    """
    Whether the operator computes its connected configurations one at a time,
    with :meth:`~netket.operator.DiscreteJaxOperator.get_conn_at` defined by
    the same class as its
    :meth:`~netket.operator.DiscreteJaxOperator.get_conn_padded`: a subclass
    that only changes ``get_conn_padded`` falls back to it.
    """

    def owner(name):
        return next(c for c in type(O).__mro__ if name in vars(c))

    defined_by = owner("get_conn_at")
    return defined_by is not DiscreteJaxOperator and defined_by is owner(
        "get_conn_padded"
    )


def _local_value_kernel_jax_packed(
    logpsi, pars, σ, O, *, chunk_size, min_chunk_size, conn_chunk_size, generate
):
    # Runs on the samples of a single device.
    n_samples, N = σ.shape
    if conn_chunk_size is None:
        conn_chunk_size = _CONN_CHUNK_BYTES // (N * σ.dtype.itemsize)
    if generate is None:
        # Computing the connected configurations in the chunks only pays off
        # when those of all the samples do not fit in a single group.
        generate = (
            generates_connected(O) and n_samples * O.max_conn_size > conn_chunk_size
        )
    # The network is called in many places, on a few shapes of chunks: as a
    # jitted function, it is traced and lowered once per shape instead of once
    # per call, which takes most of the compilation time of large models.
    logpsi = jax.jit(logpsi)

    if generate:
        # The matrix elements, and which connected configurations differ from
        # the samples. The configurations are computed in the chunks, from the
        # samples and the slots of the selected connected elements.
        mels, is_new = O.get_conn_mels(σ)
        max_conn_size = mels.shape[-1]
    else:
        max_conn_size = O.max_conn_size
    if chunk_size is None:
        chunk_size = -(-n_samples * max_conn_size // _N_CHUNKS_UNCHUNKED)
    chunk_size = max(1, chunk_size)
    logpsi_σ = nkjax.apply_chunked(
        logpsi, in_axes=(None, 0), chunk_size=chunk_size, axis_0_is_sharded=False
    )(pars, σ)

    if generate:
        logpsi_σp = nkjax.apply_packed(
            partial(_logpsi_conn_at, logpsi),
            (pars, jax.tree.map(jax.lax.stop_gradient, O)),
            σ,
            is_new,
            jnp.expand_dims(logpsi_σ, -1),
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            axis_0_is_sharded=False,
        )
        return jnp.sum(
            mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1
        )

    def local_values(σ, logpsi_σ):
        σp, mels = O.get_conn_padded(σ)
        # The connected configurations equal to σ (the diagonal and the
        # padding) are found by comparison, so that we do not rely on the
        # operator placing them at a given position.
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
        return jnp.sum(
            mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1
        )

    # Groups of samples whose padded connected configurations hold at most
    # conn_chunk_size configurations, all of the same size.
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


def _logpsi_conn_at(logpsi, args, x, slots):
    pars, O = args
    return logpsi(pars, O.get_conn_at(x, slots))
