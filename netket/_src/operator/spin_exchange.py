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

from netket.graph import AbstractGraph
from netket.hilbert import Spin
from netket.hilbert.constraint import SumConstraint
from netket.operator import DiscreteJaxOperator
from netket.utils.types import Array, DType


def _edge_disjoint_triangles(edges: np.ndarray) -> int:
    """Number of edge-disjoint triangles found greedily in a graph."""
    neighbours = {}
    for i, j in edges:
        neighbours.setdefault(i, set()).add(j)
        neighbours.setdefault(j, set()).add(i)
    used = set()
    n_triangles = 0
    for i, j in sorted(tuple(sorted(e)) for e in edges):
        if (i, j) in used:
            continue
        for k in sorted(neighbours[i] & neighbours[j]):
            e1, e2 = tuple(sorted((i, k))), tuple(sorted((j, k)))
            if e1 not in used and e2 not in used:
                used.update({(i, j), e1, e2})
                n_triangles += 1
                break
    return n_triangles


def _max_antiparallel_bonds(hilbert: Spin, edges: np.ndarray) -> int:
    """
    An upper bound on the number of anti-parallel bonds of any configuration.

    It is the minimum of:

    - the number of bonds;
    - the number of bonds minus the number of edge-disjoint triangles: the 3
      spins of a triangle cannot be pairwise anti-parallel, so every triangle
      has at least one parallel bond;
    - with a fixed magnetisation, the sum of the degrees of the sites of the
      minority spin species, which are an endpoint of every anti-parallel bond.
    """
    if len(edges) == 0:
        return 0
    bound = len(edges) - _edge_disjoint_triangles(edges)

    constraint = hilbert.constraint
    if isinstance(constraint, SumConstraint):
        # The local states of a spin-1/2 are ±1.
        n_up = (hilbert.size + constraint.sum_value) // 2
        n_minority = int(min(n_up, hilbert.size - n_up))
        degrees = np.bincount(edges.reshape(-1), minlength=hilbert.size)
        bound = min(bound, int(np.sort(degrees)[::-1][:n_minority].sum()))
    return int(bound)


@register_pytree_node_class
class SpinExchangeOperator(DiscreteJaxOperator):
    r"""
    Jax operator for spin-1/2 Hamiltonians with two-body exchange
    interactions conserving the magnetisation, plus a longitudinal field

    .. math::

        \hat H = \sum_{\langle i,j\rangle} \left[ J_{ij}\left(
        \hat\sigma^x_i\hat\sigma^x_j + \hat\sigma^y_i\hat\sigma^y_j\right)
        + J^z_{ij}\, \hat\sigma^z_i\hat\sigma^z_j \right]
        + \sum_i h_i \hat\sigma^z_i,

    which includes the Heisenberg (:math:`J^z = J`) and XXZ models, with the
    same conventions as :func:`netket.operator.Heisenberg`.

    The exchange term only connects a configuration to the swaps of its
    anti-parallel bonds. Like the particle-number conserving fermionic
    operators, this operator only generates those:
    :meth:`get_conn_padded` returns the diagonal element first, followed by
    the swaps of the anti-parallel bonds packed in
    :attr:`max_offdiag_conn_size` entries. This bound is computed from the
    graph and the Hilbert space (see :attr:`max_offdiag_conn_size`), and is
    smaller than the number of bonds on frustrated lattices or at large
    magnetisation: for example :math:`2N` instead of :math:`3N` on the
    triangular lattice. The diagonal is computed directly from the
    :math:`\hat\sigma^z` values.

    Example:

        >>> import netket as nk
        >>> from netket.experimental.operator import SpinExchangeOperator
        >>> g = nk.graph.Triangular([3, 4], pbc=True)
        >>> hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
        >>> H = SpinExchangeOperator(hi, g)
        >>> H.max_conn_size, nk.operator.Heisenberg(hi, g).max_conn_size
        (25, 37)
    """

    def __init__(
        self,
        hilbert: Spin,
        graph: AbstractGraph | Array,
        J: float | Array = 1.0,
        Jz: float | Array | None = None,
        h: float | Array = 0.0,
        sign_rule: bool | None = None,
        max_offdiag_conn_size: int | None = None,
        dtype: DType | None = None,
    ):
        """
        Constructs the operator.

        Args:
            hilbert: a spin-1/2 Hilbert space, optionally with a fixed total
                magnetisation.
            graph: the graph whose edges are the bonds, or an array of shape
                `(n_bonds, 2)`.
            J: the exchange coupling, a scalar or one value per bond.
            Jz: the :math:`\\hat\\sigma^z\\hat\\sigma^z` coupling, a scalar or one
                value per bond. Defaults to `J` (Heisenberg model).
            h: the longitudinal field, a scalar or one value per site.
            sign_rule: if True, the sign of the exchange term is flipped, which
                for a bipartite graph is Marshall's sign rule. Defaults to True
                if `graph` is a bipartite graph, as in
                :func:`netket.operator.Heisenberg`, and to False otherwise.
            max_offdiag_conn_size: the maximum number of nonzero off-diagonal
                connected elements. Defaults to a bound computed from the
                graph and the Hilbert space, which holds for every configuration.
                A configuration exceeding a smaller bound has a NaN diagonal
                element.
            dtype: the dtype of the matrix elements.
        """
        if not isinstance(hilbert, Spin) or len(hilbert.local_states) != 2:
            raise TypeError("SpinExchangeOperator requires a spin-1/2 Hilbert space.")
        if hilbert.constrained and not isinstance(hilbert.constraint, SumConstraint):
            raise TypeError(
                "SpinExchangeOperator only supports a fixed total magnetisation "
                "as constraint."
            )
        super().__init__(hilbert)

        if isinstance(graph, AbstractGraph):
            if sign_rule is None:
                sign_rule = graph.is_bipartite()
            graph = graph.edges()
        edges = np.asarray(graph, dtype=np.int32).reshape(-1, 2)

        if Jz is None:
            Jz = J
        dtype = jnp.result_type(dtype or float, J, Jz, h)
        J = np.broadcast_to(np.asarray(J, dtype=dtype), (len(edges),))
        Jz = np.broadcast_to(np.asarray(Jz, dtype=dtype), (len(edges),))
        h = np.broadcast_to(np.asarray(h, dtype=dtype), (hilbert.size,))

        # Bonds without exchange do not generate connected elements.
        is_exchange = J != 0
        if max_offdiag_conn_size is None:
            max_offdiag_conn_size = _max_antiparallel_bonds(hilbert, edges[is_exchange])

        self._edges = jnp.asarray(edges)
        self._J = jnp.asarray(-J if sign_rule else J)
        self._Jz = jnp.asarray(Jz)
        self._h = jnp.asarray(h)
        self._max_offdiag_conn_size = int(max_offdiag_conn_size)

    @property
    def edges(self) -> Array:
        """The bonds, an array of shape `(n_bonds, 2)`."""
        return self._edges

    @property
    def J(self) -> Array:
        """The exchange coupling of every bond (including the sign rule)."""
        return self._J

    @property
    def Jz(self) -> Array:
        """The σᶻσᶻ coupling of every bond."""
        return self._Jz

    @property
    def h(self) -> Array:
        """The longitudinal field on every site."""
        return self._h

    @property
    def dtype(self) -> DType:
        return self._J.dtype

    @property
    def is_hermitian(self) -> bool:
        return not jnp.iscomplexobj(self._J)

    @property
    def max_offdiag_conn_size(self) -> int:
        """
        An upper bound on the number of nonzero off-diagonal connected
        elements of every configuration, the number of anti-parallel bonds.

        Unless specified, it is the minimum of the number of bonds, the number
        of bonds minus the number of edge-disjoint triangles (every triangle
        has at least one parallel bond), and, with a fixed magnetisation, the
        sum of the degrees of the sites of the minority spin species.
        """
        return self._max_offdiag_conn_size

    @property
    def max_conn_size(self) -> int:
        return self._max_offdiag_conn_size + 1

    @jax.jit
    def get_conn_padded(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        K = self._max_offdiag_conn_size

        # +1 for the first local state, -1 for the second one, as σᶻ.
        z = 1 - 2 * self.hilbert.states_to_local_indices(x).astype(self.dtype)
        zi = z[:, self._edges[:, 0]]
        zj = z[:, self._edges[:, 1]]
        mels_diag = (zi * zj) @ self._Jz + z @ self._h

        is_antiparallel = (zi != zj) & (self._J != 0)
        n_offdiag = is_antiparallel.sum(axis=-1)
        mels_diag = jnp.where(n_offdiag > K, jnp.nan, mels_diag)

        (bond,) = jax.vmap(lambda m: jnp.nonzero(m, size=K, fill_value=0))(
            is_antiparallel
        )
        is_valid = jnp.arange(K) < n_offdiag[:, None]
        i = self._edges[bond, 0]
        j = self._edges[bond, 1]
        # swap the two spins of every selected bond
        rows = jnp.arange(x.shape[0])[:, None]
        cols = jnp.arange(K)[None, :]
        xp = jnp.broadcast_to(x[:, None, :], (x.shape[0], K, x.shape[-1]))
        xi = x[rows, i]
        xj = x[rows, j]
        xp = xp.at[rows, cols, i].set(jnp.where(is_valid, xj, xi))
        xp = xp.at[rows, cols, j].set(jnp.where(is_valid, xi, xj))
        mels = jnp.where(is_valid, 2 * self._J[bond], 0)

        xp = jnp.concatenate([x[:, None, :], xp], axis=1)
        mels = jnp.concatenate([mels_diag[:, None], mels], axis=1)
        return (
            xp.reshape(*shape[:-1], K + 1, shape[-1]),
            mels.reshape(*shape[:-1], K + 1),
        )

    def to_local_operator(self):
        """Returns the equivalent :class:`netket.operator.LocalOperator`."""
        from netket.operator import LocalOperator, spin

        hi = self.hilbert
        exchange = np.array(
            [[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]], dtype=self.dtype
        )
        sz_sz = np.diag(np.array([1, -1, -1, 1], dtype=self.dtype))
        op = LocalOperator(hi, dtype=self.dtype)
        for (i, j), J, Jz in zip(
            np.asarray(self._edges), np.asarray(self._J), np.asarray(self._Jz)
        ):
            op += LocalOperator(hi, J * exchange + Jz * sz_sz, [int(i), int(j)])
        for i, h in enumerate(np.asarray(self._h)):
            if h != 0:
                op += h * spin.sigmaz(hi, i, dtype=self.dtype)
        return op

    def __repr__(self):
        return (
            f"SpinExchangeOperator(hilbert={self.hilbert}, "
            f"n_bonds={len(self._edges)}, "
            f"max_offdiag_conn_size={self.max_offdiag_conn_size})"
        )

    def tree_flatten(self):
        data = (self._edges, self._J, self._Jz, self._h)
        metadata = {
            "hilbert": self.hilbert,
            "max_offdiag_conn_size": self._max_offdiag_conn_size,
        }
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        res = cls.__new__(cls)
        DiscreteJaxOperator.__init__(res, metadata["hilbert"])
        res._edges, res._J, res._Jz, res._h = data
        res._max_offdiag_conn_size = metadata["max_offdiag_conn_size"]
        return res
