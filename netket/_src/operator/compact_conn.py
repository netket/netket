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

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.operator import DiscreteJaxOperator
from netket.utils.types import Array


def split_diagonal(operator: DiscreteJaxOperator, x: Array):
    """
    Padded connected elements of a batch of configurations, split into the
    diagonal and the off-diagonal ones.

    The diagonal elements are those with :math:`x' = x`, found by comparison,
    so that we do not rely on the operator placing them at a given position.
    They include the padding, which NetKet's operators fill with :math:`x` and
    a zero matrix element. The off-diagonal elements are selected from the
    configurations only, not from the values of their matrix elements, so that
    an element that is zero for the current coefficients of the operator keeps
    its derivative with respect to them.

    Args:
        operator: a jax operator.
        x: configurations of shape `(n, hilbert.size)`.

    Returns:
        A tuple `(xp, mels, mels_diag, mask)` with the output of
        `get_conn_padded`, the sum of the diagonal matrix elements of every
        configuration and the mask of the off-diagonal elements.
    """
    xp, mels = operator.get_conn_padded(x)
    is_diag = jnp.all(xp == jnp.expand_dims(x, -2), axis=-1)
    mels_diag = jnp.sum(jnp.where(is_diag, mels, 0), axis=-1)
    return xp, mels, mels_diag, ~is_diag


def get_conn_compact(
    operator: DiscreteJaxOperator, x: Array, max_offdiag_conn_size: int
):
    """
    Connected elements of a batch of configurations, with the off-diagonal ones
    (`x' != x`) packed in a buffer of static size `max_offdiag_conn_size`.

    Args:
        operator: a jax operator.
        x: configurations of shape `(n, hilbert.size)`.
        max_offdiag_conn_size: the size of the buffer of off-diagonal elements.

    Returns:
        A tuple `(mels_diag, xp, mels, n_offdiag)` with the sum of the diagonal
        matrix elements of every configuration, of shape `(n,)`, the off-diagonal
        connected configurations and matrix elements, of shapes
        `(n, max_offdiag_conn_size, hilbert.size)` and
        `(n, max_offdiag_conn_size)`, and the number of off-diagonal
        elements of every configuration. If the latter is larger than
        `max_offdiag_conn_size`, the elements that do not fit are dropped.
        Unused entries contain the configuration itself and a zero matrix element.
    """
    K = max_offdiag_conn_size
    xp, mels, mels_diag, mask = split_diagonal(operator, x)
    n_offdiag = mask.sum(axis=-1)

    (idx,) = jax.vmap(lambda m: jnp.nonzero(m, size=K, fill_value=0))(mask)
    is_valid = jnp.arange(K) < n_offdiag[:, None]
    xp = jnp.take_along_axis(xp, idx[..., None], axis=1)
    xp = jnp.where(is_valid[..., None], xp, jnp.expand_dims(x, -2))
    mels = jnp.where(is_valid, jnp.take_along_axis(mels, idx, axis=1), 0)
    return mels_diag, xp, mels, n_offdiag


def _warn_overflow(n_offdiag_max, max_offdiag_conn_size):
    import warnings

    warnings.warn(
        f"A configuration has {int(n_offdiag_max)} off-diagonal "
        f"connected elements, more than max_offdiag_conn_size="
        f"{max_offdiag_conn_size}: its local estimator is NaN. Increase the "
        f"bound, or check it with CompactConnOperator.validate.",
        RuntimeWarning,
        stacklevel=1,
    )


def check_overflow(
    values: Array, n_offdiag: Array, max_offdiag_conn_size: int, max_conn_size: int
):
    """
    Sets to NaN the values of the configurations with more than
    `max_offdiag_conn_size` off-diagonal elements, and warns at
    runtime if there is any.

    Nothing is checked if the operator has no more than `max_offdiag_conn_size`
    padded connected elements (`max_conn_size`), so that no configuration can
    exceed the bound. This includes the operators without connected elements,
    for which the warning (a callback that does not depend on the samples)
    cannot be compiled when the samples are sharded.
    """
    if max_conn_size <= max_offdiag_conn_size:
        return values
    is_overflow = n_offdiag > max_offdiag_conn_size
    jax.lax.cond(
        jnp.any(is_overflow),
        lambda: jax.debug.callback(
            _warn_overflow, jnp.max(n_offdiag), max_offdiag_conn_size
        ),
        lambda: None,
    )
    return jnp.where(is_overflow, jnp.nan, values)


@register_pytree_node_class
class CompactConnOperator(DiscreteJaxOperator):
    r"""
    Wraps a jax operator, declaring a static bound on the number of
    off-diagonal connected elements (:math:`x' \neq x`) of every configuration.

    :meth:`~netket.operator.DiscreteJaxOperator.get_conn_padded` of most jax
    operators returns one entry per term of the operator, many of which have a
    zero matrix element for a given configuration. For example, the Heisenberg
    model only connects a configuration to the flips of its anti-parallel
    bonds, and on the triangular lattice at most 2 of the 3 bonds of every
    triangle can be anti-parallel: no configuration has more than :math:`2N`
    connected elements, instead of the :math:`3N + 1` padded ones.

    This operator returns instead the diagonal element first, followed by the
    off-diagonal elements packed in
    :attr:`max_offdiag_conn_size` entries, so that every consumer of
    :meth:`get_conn_padded` benefits from the tighter bound. The local
    estimators on a :class:`netket.vqs.MCState` are moreover computed with
    :func:`netket.vqs.mc.kernels.local_value_kernel_jax_compact`, which does
    not evaluate the network on the diagonal.

    The off-diagonal elements are the connected configurations
    :math:`x' \neq x`: the padding, which NetKet's operators fill with
    :math:`x` and a zero matrix element, is not counted, but an element whose
    matrix element vanishes only for the current coefficients of the operator
    is, so that it keeps its derivative with respect to them.

    The bound is taken from the argument, or else from the
    :attr:`~netket.operator.DiscreteJaxOperator.max_offdiag_conn_size` of the
    wrapped operator. It must hold for every configuration of the Hilbert
    space: a configuration with more elements has a NaN diagonal element (and
    therefore a NaN local estimator), and a :class:`RuntimeWarning` is issued.
    Use :meth:`validate` to check a bound on a set of configurations, or on
    the whole Hilbert space if it is small enough.

    Example:

        >>> import netket as nk
        >>> from netket.experimental.operator import CompactConnOperator
        >>> g = nk.graph.Triangular([3, 4], pbc=True)
        >>> hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
        >>> H = nk.operator.Heisenberg(hi, g)
        >>> H.max_conn_size
        37
        >>> Hc = CompactConnOperator(H, 2 * g.n_nodes)
        >>> Hc.max_conn_size
        25
        >>> Hc.validate()
        24
    """

    def __init__(
        self,
        operator: DiscreteJaxOperator,
        max_offdiag_conn_size: int | None = None,
    ):
        """
        Constructs the operator.

        Args:
            operator: the jax operator to wrap.
            max_offdiag_conn_size: the maximum number of off-diagonal
                connected elements of every configuration. Defaults to the
                :attr:`~netket.operator.DiscreteJaxOperator.max_offdiag_conn_size`
                of `operator`.
        """
        if not isinstance(operator, DiscreteJaxOperator):
            raise TypeError(
                "CompactConnOperator can only wrap a DiscreteJaxOperator, "
                f"but got {type(operator)}."
            )
        if max_offdiag_conn_size is None:
            max_offdiag_conn_size = operator.max_offdiag_conn_size
        if max_offdiag_conn_size is None:
            raise ValueError(
                f"The operator {type(operator).__name__} does not declare a "
                "max_offdiag_conn_size: you must specify it."
            )
        if max_offdiag_conn_size < 0:
            raise ValueError("max_offdiag_conn_size must be non-negative.")
        # Operators that are set up lazily must not be set up inside a trace.
        if hasattr(operator, "_setup"):
            operator._setup()

        super().__init__(operator.hilbert)
        self._operator = operator
        self._max_offdiag_conn_size = int(max_offdiag_conn_size)

    @property
    def operator(self) -> DiscreteJaxOperator:
        """The wrapped operator."""
        return self._operator

    @property
    def max_offdiag_conn_size(self) -> int:
        return self._max_offdiag_conn_size

    @property
    def max_conn_size(self) -> int:
        return self._max_offdiag_conn_size + 1

    @property
    def dtype(self):
        return self.operator.dtype

    @property
    def is_hermitian(self) -> bool:
        return self.operator.is_hermitian

    def get_conn_padded(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        mels_diag, xp, mels, n_offdiag = get_conn_compact(
            self.operator, x, self.max_offdiag_conn_size
        )
        mels_diag = check_overflow(
            mels_diag,
            n_offdiag,
            self.max_offdiag_conn_size,
            self.operator.max_conn_size,
        )
        xp = jnp.concatenate([x[:, None, :], xp], axis=1)
        mels = jnp.concatenate([mels_diag[:, None].astype(mels.dtype), mels], axis=1)
        return xp.reshape(*shape[:-1], *xp.shape[1:]), mels.reshape(
            *shape[:-1], mels.shape[-1]
        )

    def validate(self, x: Array | None = None, *, chunk_size: int = 4096) -> int:
        """
        Checks that the bound holds for a set of configurations.

        Args:
            x: the configurations to check, of shape `(..., hilbert.size)`.
                Defaults to all the states of the Hilbert space, which must
                be indexable.
            chunk_size: the number of configurations checked at once.

        Returns:
            The maximum number of off-diagonal connected elements.

        Raises:
            ValueError: if a configuration has more off-diagonal
                connected elements than :attr:`max_offdiag_conn_size`.
        """
        if x is None:
            x = self.hilbert.all_states()
        x = jnp.asarray(x).reshape(-1, self.hilbert.size)

        n_offdiag_max = 0
        for i in range(0, x.shape[0], chunk_size):
            n = _n_offdiag(self.operator, x[i : i + chunk_size])
            n_offdiag_max = max(n_offdiag_max, int(np.max(n)))

        if n_offdiag_max > self.max_offdiag_conn_size:
            raise ValueError(
                f"A configuration has {n_offdiag_max} off-diagonal "
                f"connected elements, more than max_offdiag_conn_size="
                f"{self.max_offdiag_conn_size}."
            )
        return n_offdiag_max

    def __repr__(self):
        return (
            f"CompactConnOperator({self.operator}, "
            f"max_offdiag_conn_size={self.max_offdiag_conn_size})"
        )

    def tree_flatten(self):
        return (self.operator,), {"max_offdiag_conn_size": self.max_offdiag_conn_size}

    @classmethod
    def tree_unflatten(cls, metadata, data):
        (operator,) = data
        res = cls.__new__(cls)
        DiscreteJaxOperator.__init__(res, operator.hilbert)
        res._operator = operator
        res._max_offdiag_conn_size = metadata["max_offdiag_conn_size"]
        return res


@jax.jit
def _n_offdiag(operator, x):
    return split_diagonal(operator, x)[3].sum(axis=-1)
