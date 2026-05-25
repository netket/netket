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

"""LocalEstimators and LocalEstimatorsBatch: per-sample estimator containers."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from netket.utils import struct

# from netket.utils.deprecation import warn_deprecation
from netket.stats.mc_stats import Stats, statistics
from netket._src.stats.online_stats.operations import online_statistics
from netket._src.stats.online_stats.accumulator_batch import (
    OnlineStatsBatch,
    _delta_method_stats,
)


# Attributes that exist on JAX arrays but not on LocalEstimators/LocalEstimatorsBatch.
# Used in __getattr__ to give a more helpful error than the default AttributeError.
# fmt: off
_ARRAY_LIKE_ATTRS = frozenset({
    "reshape", "mean", "std", "sum", "min", "max", "flatten", "ravel",
    "T", "dtype", "real", "imag", "conj", "conjugate", "block_until_ready",
    "at", "astype", "tolist", "item", "nbytes", "strides", "addressable_shards",
    "size", "ndim", "shape",
})
# fmt: on


class LocalEstimators(struct.Pytree):
    """Per-sample scalar estimators returned by :meth:`~netket.vqs.MCState.local_estimators`.

    ``data`` has shape ``(n_chains, chain_len)``, one scalar local estimator value per
    sample.

    The typical workflow is::

        le = vstate.local_estimators(op)     # LocalEstimators, data: (n_chains, chain_len)
        stats = le.to_stats()               # one-shot Stats
        acc   = le.accumulate()             # start an OnlineStats accumulator

    For online estimation across multiple sampling steps::

        acc = None
        for _ in range(n_steps):
            vstate.sample(n_discard_per_chain=0)
            le  = vstate.local_estimators(op)
            acc = le.accumulate(acc)         # updates or creates the accumulator
        print(acc.get_stats())
    """

    data: jax.Array = struct.field(pytree_node=True)
    """Scalar local estimators with shape ``(n_chains, chain_len)``."""

    def __init__(self, data: jax.Array):
        self.data = data

    # ------------------------------------------------------------------
    # Backward compatibility: LocalEstimators used to be a raw JAX array
    # ------------------------------------------------------------------
    def __jax_array__(self):
        # TODO: May 2026 To decide if we should raise deprecation warning
        # probably yes...
        # warn_deprecation(
        #     "Implicit conversion of LocalEstimators to a JAX array is deprecated. "
        #     "Access the underlying array via .data instead."
        # )
        return self.data

    def __getattr__(self, name):
        if name in _ARRAY_LIKE_ATTRS:
            # TODO: May 2026 To decide if we should raise deprecation warning
            # probably yes...
            # warn_deprecation(
            #     f"LocalEstimators.{name} is deprecated: access .data.{name} directly."
            # )
            return getattr(self.data, name)
        raise AttributeError(f"LocalEstimators has no attribute {name!r}.")

    # ------------------------------------------------------------------

    def to_stats(self) -> Stats:
        """Compute summary statistics for this scalar local-estimator batch.

        Equivalent to calling :func:`~netket.stats.statistics` on ``self.data``.
        """
        return statistics(self.data)

    def to_online_stats(self, *, max_lag: int = 64):
        """Create an :class:`~netket.stats.OnlineStats` initialised with this batch.

        Subsequent batches are folded in via ``acc = acc.update(new_le.data)``.
        Prefer :meth:`accumulate` when writing the accumulation loop, as it
        handles both first and subsequent batches uniformly.
        """
        return online_statistics(self.data, None, max_lag=max_lag)

    def accumulate(self, old=None, *, max_lag: int = 64):
        """Fold this batch into an online accumulator.

        Args:
            old: existing :class:`~netket.stats.OnlineStats` returned by a
                previous call, or ``None`` to start a fresh accumulator.
            max_lag: maximum ACF lag (only used when creating a fresh
                accumulator on the first call).

        Returns:
            Updated :class:`~netket.stats.OnlineStats`.
        """
        if old is None:
            return self.to_online_stats(max_lag=max_lag)
        return old.update(self.data)


class LocalEstimatorsBatch(struct.Pytree):
    """Per-sample K-channel estimators for nonlinear observables.

    ``data`` has shape ``(n_chains, chain_len, K)``.
    ``combinator: (K,) -> scalar | array`` collapses the K channel means into a
    result of any shape using the **delta method** (first-order error propagation).

    This class is returned by :meth:`~netket.vqs.MCState.local_estimators` for
    operators such as :class:`~netket.experimental.observable.VarianceObservable`
    that require more than one local estimator channel to form the final quantity.

    :meth:`to_stats` uses :func:`jax.eval_shape` to inspect the combinator
    output shape and returns:

    - a :class:`~netket.stats.Stats` for scalar combinators;
    - a ``(mean, error_of_mean)`` tuple of JAX arrays for array-valued
      combinators, both with the same shape as ``combinator(X)``.

    :meth:`to_online_stats` returns an :class:`~netket.stats.OnlineStatsBatch`.

    Examples::

        # Variance: 2 channels, scalar combinator
        le = LocalEstimatorsBatch(
            data=jnp.stack([H_loc, H_loc**2], axis=-1),  # (n_chains, chain_len, 2)
            combinator=lambda mu: mu[1] - mu[0]**2,
        )
        stats = le.to_stats()    # Stats (scalar combinator → Stats)
        acc   = le.accumulate()  # OnlineStatsBatch for iterative estimation

        # Susceptibility matrix: p+p² channels, array combinator
        le = LocalEstimatorsBatch(
            data=channels,        # (n_chains, chain_len, p + p²)
            combinator=chi_matrix,  # (K,) -> (p, p)
        )
        mean, err = le.to_stats()       # both shape (p, p)
        mean, err = le.accumulate().get_stats()  # online version, same shapes
    """

    data: jax.Array = struct.field(pytree_node=True)
    """Estimator channels with shape ``(n_chains, chain_len, K)``."""
    combinator: Callable = struct.field(pytree_node=False)
    """JAX-traceable ``(K,) -> scalar | array`` map combining channel means."""

    def __init__(self, data: jax.Array, combinator: Callable):
        self.data = data
        self.combinator = combinator

    @property
    def n_channels(self) -> int:
        """Number of channels K in the data (last axis of ``data``)."""
        return self.data.shape[-1]

    def __getattr__(self, name):
        if name in _ARRAY_LIKE_ATTRS:
            raise AttributeError(
                f"LocalEstimatorsBatch has no attribute {name!r}. "
                f"The underlying JAX array is at .data; use le.data.{name} instead."
            )
        raise AttributeError(f"LocalEstimatorsBatch has no attribute {name!r}.")

    def to_stats(self):
        """One-shot delta-method statistics for this batch.

        Computes per-chain means, forms their sample covariance matrix, and
        applies the delta method via :func:`jax.eval_shape`-based dispatch:

        - Returns a :class:`~netket.stats.Stats` for scalar combinators.
        - Returns ``(mean, error_of_mean)`` as JAX arrays for array-valued
          combinators, both with the same shape as ``combinator(X)``.
        """
        n_chains = self.data.shape[0]
        chain_means = jnp.mean(self.data, axis=1)  # (n_chains, K)
        X = jnp.mean(chain_means, axis=0)  # (K,)

        if n_chains < 2:
            flat = self.data.reshape(-1, self.data.shape[-1])
            deviations = flat - X[None, :]
            Cov = (deviations.T @ deviations) / flat.shape[0] ** 2
        else:
            deviations = chain_means - X[None, :]  # (n_chains, K)
            Cov = (deviations.T @ deviations) / n_chains**2

        return _delta_method_stats(self.combinator, X, Cov)

    def to_online_stats(self, *, max_lag: int = 64) -> OnlineStatsBatch:
        """Create an :class:`~netket.stats.OnlineStatsBatch` initialised with this batch.

        Subsequent batches are folded in via ``acc = acc.update(new_le.data)``.
        Prefer :meth:`accumulate` when writing the accumulation loop, as it
        handles both first and subsequent batches uniformly.
        """
        return OnlineStatsBatch.from_data(
            self.data,
            self.combinator,
            max_lag=max_lag,
        )

    def accumulate(self, old=None, *, max_lag: int = 64) -> OnlineStatsBatch:
        """Fold this batch into an online accumulator.

        Args:
            old: existing :class:`~netket.stats.OnlineStatsBatch` returned by a
                previous call, or ``None`` to start a fresh accumulator.
            max_lag: maximum ACF lag (only used when creating a fresh
                accumulator on the first call).

        Returns:
            Updated :class:`~netket.stats.OnlineStatsBatch`.
        """
        if old is None:
            return self.to_online_stats(max_lag=max_lag)
        return old.update(self.data)
