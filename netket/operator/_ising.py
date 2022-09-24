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

from typing import Optional
from numba import jit

import numpy as np
import jax.numpy as jnp

from netket.graph import AbstractGraph, Graph
from netket.hilbert import AbstractHilbert
from netket.utils.numbers import dtype as _dtype
from netket.utils.types import DType

from . import spin
from ._local_operator import LocalOperator
from ._hamiltonian import SpecialHamiltonian


class Ising(SpecialHamiltonian):
    r"""
    The Transverse-Field Ising Hamiltonian :math:`-h\sum_i \sigma_i^{(x)} +J\sum_{\langle i,j\rangle} \sigma_i^{(z)}\sigma_j^{(z)}`.

    This implementation is considerably faster than the Ising hamiltonian constructed by summing :class:`~netket.operator.LocalOperator` s.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        h: float,
        J: float = 1.0,
        dtype: Optional[DType] = None,
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
        assert (
            graph.n_nodes == hilbert.size
        ), "The size of the graph must match the hilbert space"

        super().__init__(hilbert)

        if dtype is None:
            dtype = jnp.promote_types(_dtype(h), _dtype(J))
        dtype = np.empty((), dtype=dtype).dtype
        self._dtype = dtype

        self._h = np.array(h, dtype=dtype)
        self._J = np.array(J, dtype=dtype)
        self._edges = np.asarray(
            [[u, v] for u, v in graph.edges()],
            dtype=np.intp,
        )

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
        return self._edges

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self) -> DType:
        return self._dtype

    def conjugate(self, *, concrete=True):
        # if real
        if isinstance(self.h, float) and isinstance(self.J, float):
            return self
        else:
            raise NotImplementedError

    @staticmethod
    @jit(nopython=True)
    def n_conn(x, out):
        r"""Return the number of states connected to x.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            out (array): If None an output array is allocated.

        Returns:
            array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = np.empty(
                x.shape[0],
                dtype=np.int32,
            )

        out.fill(x.shape[1] + 1)

        return out

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self.hilbert.size + 1

    def copy(self, *, dtype: Optional[DType] = None):
        if dtype is None:
            dtype = self.dtype

        graph = Graph(edges=[list(edge) for edge in self.edges])
        return Ising(hilbert=self.hilbert, graph=graph, J=self.J, h=self.h, dtype=dtype)

    def to_local_operator(self):
        # The hamiltonian
        ha = LocalOperator(self.hilbert, dtype=self.dtype)

        if self.h != 0:
            for i in range(self.hilbert.size):
                ha -= self.h * spin.sigmax(self.hilbert, i, dtype=self.dtype)

        if self.J != 0:
            for (i, j) in self.edges:
                ha += self.J * (
                    spin.sigmaz(self.hilbert, i, dtype=self.dtype)
                    * spin.sigmaz(self.hilbert, j, dtype=self.dtype)
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

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(
        x,
        sections,
        edges,
        h,
        J,
    ):
        n_sites = x.shape[1]
        n_conn = n_sites + 1

        x_prime = np.empty(
            (
                x.shape[0] * n_conn,
                n_sites,
            ),
            dtype=x.dtype,
        )
        mels = np.empty(x.shape[0] * n_conn, dtype=h.dtype)

        diag_ind = 0

        for i in range(x.shape[0]):
            mels[diag_ind] = 0.0
            for k in range(edges.shape[0]):
                mels[diag_ind] += J * x[i, edges[k, 0]] * x[i, edges[k, 1]]

            odiag_ind = 1 + diag_ind

            mels[odiag_ind : (odiag_ind + n_sites)].fill(-h)

            x_prime[diag_ind : (diag_ind + n_conn)] = np.copy(x[i])

            for j in range(n_sites):
                x_prime[j + odiag_ind][j] *= -1.0

            diag_ind += n_conn

            sections[i] = diag_ind

        return x_prime, mels

    def get_conn_flattened(
        self,
        x,
        sections,
        pad=False,
    ):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of size (batch_size) useful to unflatten
                        the output of this function.
                        See numpy.split for the meaning of sections.
            pad (bool): no effect here

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self._flattened_kernel(
            np.asarray(x),
            sections,
            self._edges,
            self._h,
            self._J,
        )

    def _get_conn_flattened_closure(self):
        _edges = self._edges
        _h = self._h
        _J = self._J
        fun = self._flattened_kernel

        def gccf_fun(x, sections):
            return fun(x, sections, _edges, _h, _J)

        return jit(nopython=True)(gccf_fun)

    def __repr__(self):
        return f"Ising(J={self._J}, h={self._h}; dim={self.hilbert.size})"
