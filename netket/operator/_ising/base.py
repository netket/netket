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

from typing import Optional, Union

import numpy as np

from jax import numpy as jnp

from netket.graph import AbstractGraph
from netket.hilbert import AbstractHilbert
from netket.jax import canonicalize_dtypes
from netket.utils.numbers import dtype as _dtype
from netket.utils.types import Array, DType

from .. import spin
from .._hamiltonian import SpecialHamiltonian
from .._local_operator import LocalOperator


class IsingBase(SpecialHamiltonian):
    r"""
    The Transverse-Field Ising Hamiltonian :math:`-h\sum_i \sigma_i^{(x)} +J\sum_{\langle i,j\rangle} \sigma_i^{(z)}\sigma_j^{(z)}`.

    This implementation is considerably faster than the Ising hamiltonian constructed by summing :class:`~netket.operator.LocalOperator` s.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: Union[AbstractGraph, Array],
        h: float,
        J: float,
        dtype: Optional[DType],
    ):
        r"""
        Constructs the Ising Operator from an hilbert space and a
        graph specifying the connectivity.

        Args:
            hilbert: Hilbert space the operator acts on.
            h: The strength of the transverse field.
            J: The strength of the coupling. Default is 1.0.
            dtype: The dtype of the matrix elements.

        Examples:
            Constructs an ``Ising`` operator for a 1D system.

            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
            >>> op = nk.operator.Ising(h=1.321, hilbert=hi, J=0.5, graph=g)
            >>> print(op)
            Ising(J=0.5, h=1.321; dim=20)
        """
        super().__init__(hilbert)

        dtype = canonicalize_dtypes(float, h, J, dtype=dtype)

        if isinstance(graph, AbstractGraph):
            if graph.n_nodes != hilbert.size:
                raise ValueError(
                    """
                    The size of the graph must match the hilbert space.
                    """
                )
            # support also a matrix input in here.
            graph = graph.edges()

        if isinstance(graph, list):
            graph = np.asarray(
                [[u, v] for u, v in graph],
                dtype=np.intp,
            )

        if graph.ndim != 2 or graph.shape[1] != 2:
            raise ValueError(
                """
                Graph should be one of:
                    - NetKet graph type (nk.operator.AbstractGraph)
                    - List of tuples, describing the edges
                    - a (N,2) array of integers.
                """
            )

        self._h = h.astype(dtype=dtype)
        self._J = J.astype(dtype=dtype)
        self._edges = graph.astype(np.intp)

    @property
    def h(self) -> float:
        """The magnitude of the transverse field"""
        return self._h

    @property
    def J(self) -> float:
        """The magnitude of the hopping"""
        return self._J

    @property
    def edges(self) -> np.ndarray:
        """The (N_conns, 2) matrix of edges on which the interaction term
        is non-zero.
        """
        return self._edges

    @property
    def is_hermitian(self) -> bool:
        """A boolean stating whether this hamiltonian is hermitian."""
        return True

    @property
    def dtype(self) -> DType:
        """The dtype of the matrix elements."""
        return jnp.promote_types(_dtype(self.h), _dtype(self.J))

    def conjugate(self, *, concrete=True):
        # if real
        if isinstance(self.h, float) and isinstance(self.J, float):
            return self
        else:
            raise NotImplementedError

    def n_conn(self, x, out=None):  # pragma: no cover
        r"""Return the number of states connected to x.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            out (array): If None an output array is allocated.

        Returns:
            array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = np.empty(x.shape[0], dtype=np.int32)
        out.fill((self.h != 0) * x.shape[1] + 1)
        return out

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self.hilbert.size + 1

    def copy(self, *, dtype: Optional[DType] = None):
        if dtype is None:
            dtype = self.dtype

        return type(self)(
            hilbert=self.hilbert, graph=self.edges, J=self.J, h=self.h, dtype=dtype
        )

    def to_local_operator(self):
        # The hamiltonian
        ha = LocalOperator(self.hilbert, dtype=self.dtype)

        if self.h != 0:
            for i in range(self.hilbert.size):
                ha -= self.h * spin.sigmax(self.hilbert, int(i), dtype=self.dtype)

        if self.J != 0:
            for i, j in self.edges:
                ha += self.J * (
                    spin.sigmaz(self.hilbert, int(i), dtype=self.dtype)
                    * spin.sigmaz(self.hilbert, int(j), dtype=self.dtype)
                )

        return ha

    def _iadd_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._h += other.h
        self._J += other.J

    def _isub_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._h -= other.h
        self._J -= other.J

    def __repr__(self):
        return (
            f"{type(self).__name__}(J={self._J}, h={self._h}; dim={self.hilbert.size})"
        )
