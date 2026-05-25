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

"""OnlineStatsBatch: online accumulator for K-channel local estimators."""

from math import isnan
from collections.abc import Callable

import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.stats.mc_stats import Stats, StatsBatch
from netket._src.stats.online_stats.accumulator import OnlineStats


_NaN = float("nan")


def _delta_method_stats(
    f: Callable,
    X: jax.Array,
    Cov: jax.Array | None,
) -> Stats | StatsBatch:
    r"""Delta-method statistics for a scalar or array-valued functional ``f``.

    Linearises ``f`` around the mean ``X`` to propagate the covariance ``Cov``
    of the K channel means into an error estimate for ``f(X)``:

    .. math::

        \operatorname{Var}[f_i(X)] \approx J_i^\top \, \mathrm{Cov}[X] \, J_i

    where :math:`J_i = \nabla f_i(X)` is computed via ``jax.jacfwd``.

    Uses :func:`jax.eval_shape` to dispatch on the output shape of ``f``:

    - Returns a :class:`~netket.stats.Stats` when ``f`` returns a scalar.
    - Returns a :class:`~netket.stats.StatsBatch` when ``f`` returns an array.

    Args:
        f: JAX-traceable ``(K,) -> scalar | array``.
        X: ``(K,)`` channel means.
        Cov: ``(K, K)`` sample covariance of the chain means, or ``None`` when
            fewer than 2 chains are available (returns NaN errors).
    """
    out_shape = jax.eval_shape(f, X).shape
    mean = f(X)
    StatsClass = Stats if out_shape == () else StatsBatch

    if Cov is None:
        nan_err = _NaN if out_shape == () else jnp.full(out_shape, _NaN)
        return StatsClass(mean=mean, error_of_mean=nan_err)

    J = jax.jacfwd(f)(X)  # (*out_shape, K)
    err = jnp.sqrt(jnp.maximum(jnp.einsum("...k,kl,...l->...", J, Cov, J), 0.0))
    return StatsClass(mean=mean, error_of_mean=err)


class OnlineStatsBatch(struct.Pytree):
    """Batched accumulator for K OnlineStats estimators and a combining function.

    Wraps K separate :class:`~netket.stats.OnlineStats` instances and applies
    the *delta method* at :meth:`get_stats` time to compute statistics for a
    smooth functional of the K marginal means.

    The combinator ``f: (K,) -> scalar | array`` must be JAX-traceable so that
    ``jax.jacfwd`` can compute its Jacobian.  :meth:`get_stats` uses
    :func:`jax.eval_shape` to inspect the combinator output shape and returns:

    - a :class:`~netket.stats.Stats` object when ``f`` returns a scalar;
    - a ``(mean, error_of_mean)`` tuple of JAX arrays when ``f`` returns an
      array of any shape.

    Use :func:`online_statistics_batch` as the functional API, or
    :meth:`from_data` to construct directly from a first batch::

        acc = None
        for batch in batches:                       # batch: (n_chains, chain_len, K)
            acc = online_statistics_batch(batch, combinator, acc)
        stats = acc.get_stats()                     # Stats for scalar combinator

    For array-valued combinators (e.g. susceptibility matrix)::

        def chi_matrix(X):                          # combinator: (K,) -> (p, p)
            return X[p:].reshape(p, p) - X[:p, None] * X[None, :p]

        acc = online_statistics_batch(data, chi_matrix, acc)
        mean, err = acc.get_stats()                 # both shape (p, p)

    This class is used internally by :meth:`~netket.vqs.MCState.expect_to_precision`
    whenever the operator has a :class:`LocalEstimatorsBatch` dispatch.
    """

    estimators: tuple[OnlineStats, ...] = struct.field(pytree_node=True)
    """Per-channel :class:`~netket.stats.OnlineStats` accumulators."""

    combinator: Callable = struct.field(pytree_node=False)
    """JAX-traceable ``(K,) -> scalar | array`` map combining channel means."""

    def __init__(self, estimators: tuple, combinator: Callable):
        """
        Args:
            estimators: per-channel :class:`~netket.stats.OnlineStats` accumulators,
                one per channel K.
            combinator: JAX-traceable ``(K,) -> scalar | array`` map combining
                the K channel means into the final observable.
        """
        self.estimators = estimators
        self.combinator = combinator

    @classmethod
    def from_data(
        cls,
        data,
        combinator: Callable,
        *,
        max_lag: int = 64,
    ) -> "OnlineStatsBatch":
        """Construct an :class:`OnlineStatsBatch` from an initial batch.

        Equivalent to calling :func:`online_statistics_batch` with
        ``old_estimator=None``.

        Args:
            data: Array of shape ``(n_chains, n_samples_per_chain, K)``.
            combinator: ``f: (K,) -> scalar | array``, must be JAX-traceable.
            max_lag: Maximum lag for the per-channel online ACF estimator.

        Returns:
            A new :class:`OnlineStatsBatch` initialized from *data*.
        """
        data = jnp.asarray(data)
        if data.ndim != 3:
            raise ValueError(f"data must be 3D, got {data.ndim}D")

        K = data.shape[-1]
        n_chains = data.shape[0]
        estimators = tuple(
            OnlineStats(n_chains, dtype=jnp.float64, max_lag=max_lag) for _ in range(K)
        )
        estimator = cls(estimators=estimators, combinator=combinator)
        return estimator.update(data)

    def update(self, data) -> "OnlineStatsBatch":
        """Incorporate a new batch.

        Args:
            data: shape (n_chains, chain_len, K)
        """
        data = jnp.asarray(data)
        new_estimators = tuple(
            est.update(data[..., k]) for k, est in enumerate(self.estimators)
        )
        return OnlineStatsBatch(estimators=new_estimators, combinator=self.combinator)

    @property
    def n_samples(self) -> int:
        """Total samples accumulated."""
        return self.estimators[0].n_samples

    @property
    def n_chains(self) -> int:
        """Number of chains."""
        return self.estimators[0].n_chains

    @property
    def mean(self):
        """Current estimate of ``combinator(X)`` as a JAX array."""
        X = jnp.array([e.mean for e in self.estimators])
        return self.combinator(X)

    def get_stats(self):
        """Delta-method statistics for the combined functional.

        Uses :func:`jax.eval_shape` to inspect the combinator output shape and
        returns:

        - a :class:`~netket.stats.Stats` for scalar combinators;
        - a ``(mean, error_of_mean)`` tuple of JAX arrays for array-valued
          combinators, both with the same shape as ``combinator(X)``.
        """
        X = jnp.array([e.mean for e in self.estimators])  # (K,)

        if self.n_chains < 2 or any(isnan(float(jnp.real(m))) for m in X):
            return _delta_method_stats(self.combinator, X, None)

        chain_means = jnp.stack([e.chain_means for e in self.estimators])
        deviations = chain_means - X[:, None]
        Cov = (deviations @ deviations.T) / self.n_chains**2

        return _delta_method_stats(self.combinator, X, Cov)

    # Protocol compatibility with OnlineStats for expect_to_precision
    def to_dict(self):
        """Call :meth:`get_stats` and return its dictionary representation (scalar combinators only)."""
        return self.get_stats().to_dict()

    def to_compound(self):
        """Call :meth:`get_stats` and return its compound representation (scalar combinators only)."""
        return self.get_stats().to_compound()

    def __repr__(self):
        return repr(self.get_stats())


def online_statistics_batch(
    data,
    combinator: Callable,
    old_estimator: "OnlineStatsBatch | None" = None,
    *,
    max_lag: int = 64,
) -> OnlineStatsBatch:
    """Accumulate K-channel local estimators into an OnlineStatsBatch.

    Args:
        data: shape (n_chains, chain_len, K)
        combinator: f: (K,) -> scalar, JAX-traceable
        old_estimator: previous OnlineStatsBatch, or None to start fresh
        max_lag: maximum ACF lag (only used when creating a fresh accumulator)

    Returns:
        Updated OnlineStatsBatch.
    """
    data = jnp.asarray(data)
    if old_estimator is None:
        return OnlineStatsBatch.from_data(data, combinator, max_lag=max_lag)
    return old_estimator.update(data)
