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
import math

from netket.graph import AbstractGraph, Graph
from netket.hilbert import Fock
from netket.jax import canonicalize_dtypes
from netket.utils.types import DType
from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError

from . import boson
from ._local_operator import LocalOperator
from ._hamiltonian import SpecialHamiltonian


class BoseHubbard(SpecialHamiltonian):
    r"""
    An extended Bose Hubbard model Hamiltonian operator, containing both
    on-site interactions and nearest-neighboring density-density interactions.
    """

    def __init__(
        self,
        hilbert: Fock,
        graph: AbstractGraph,
        U: float,
        V: float = 0.0,
        J: float = 1.0,
        mu: float = 0.0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Constructs a new BoseHubbard operator given a hilbert space, a graph
        specifying the connectivity and the interaction strength.
        The chemical potential and the density-density interaction strength
        can be specified as well.

        Args:
           hilbert: Hilbert space the operator acts on.
           U: The on-site interaction term.
           V: The strength of density-density interaction term.
           J: The hopping amplitude.
           mu: The chemical potential.
           dtype: The dtype of the matrix elements.

        Examples:
           Constructs a BoseHubbard operator for a 2D system.

           >>> import netket as nk
           >>> g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
           >>> hi = nk.hilbert.Fock(n_max=3, n_particles=6, N=g.n_nodes)
           >>> op = nk.operator.BoseHubbard(hi, U=4.0, graph=g)
           >>> print(op.hilbert.size)
           9
        """
        assert (
            graph.n_nodes == hilbert.size
        ), "The size of the graph must match the hilbert space."

        assert isinstance(hilbert, Fock)
        super().__init__(hilbert)

        dtype = canonicalize_dtypes(float, U, V, J, mu, dtype=dtype)
        self._dtype = dtype

        self._U = np.asarray(U, dtype=dtype)
        self._V = np.asarray(V, dtype=dtype)
        self._J = np.asarray(J, dtype=dtype)
        self._mu = np.asarray(mu, dtype=dtype)

        self._n_max = hilbert.n_max
        self._n_sites = hilbert.size
        self._edges = np.asarray(list(graph.edges()))
        self._max_conn = 1 + self._edges.shape[0] * 2
        self._max_mels = np.empty(self._max_conn, dtype=self.dtype)
        self._max_xprime = np.empty((self._max_conn, self._n_sites))

    @property
    def is_hermitian(self):
        return True

    @property
    def dtype(self):
        return self._dtype

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def U(self):
        """The strength of on-site interaction term."""
        return self._U

    @property
    def V(self):
        """The strength of density-density interaction term."""
        return self._V

    @property
    def J(self):
        """The hopping amplitude."""
        return self._J

    @property
    def mu(self):
        """The chemical potential."""
        return self._mu

    def copy(self):
        graph = Graph(edges=[list(edge) for edge in self.edges])
        return BoseHubbard(
            hilbert=self.hilbert,
            graph=graph,
            J=self.J,
            U=self.U,
            V=self.V,
            mu=self.mu,
            dtype=self.dtype,
        )

    def to_local_operator(self):
        # The hamiltonian
        ha = LocalOperator(self.hilbert, dtype=self.dtype)

        if self.U != 0 or self.mu != 0:
            for i in range(self.hilbert.size):
                n_i = boson.number(self.hilbert, i)
                ha += (self.U / 2) * n_i * (n_i - 1) - self.mu * n_i

        if self.J != 0:
            for i, j in self.edges:
                ha += self.V * (
                    boson.number(self.hilbert, i) * boson.number(self.hilbert, j)
                )
                ha -= self.J * (
                    boson.destroy(self.hilbert, i) * boson.create(self.hilbert, j)
                    + boson.create(self.hilbert, i) * boson.destroy(self.hilbert, j)
                )

        return ha

    def _iadd_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._mu += other.mu
        self._U += other.U
        self._J += other.J
        self._V += other.V

    def _isub_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._mu -= other.mu
        self._U -= other.U
        self._J -= other.J
        self._V -= other.V

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        # 1 diagonal element + 2 for every coupling
        return 1 + 2 * len(self._edges)

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(  # pragma: no cover
        x,
        sections,
        edges,
        U,
        V,
        J,
        mu,
        n_max,
        max_conn,
        mels=None,
        x_prime=None,
        pad=False,
    ):
        batch_size = x.shape[0]
        n_sites = x.shape[1]

        mels_allocated = False
        x_prime_allocated = False

        # When executed as a closure those must be allocated inside the numba jitted function
        if mels is None:
            mels_allocated = True
            mels = np.empty(batch_size * max_conn, dtype=U.dtype)

        if x_prime is None:
            x_prime_allocated = True
            x_prime = np.empty((batch_size * max_conn, n_sites), dtype=x.dtype)

        if pad:
            x_prime[:, :] = 0
            mels[:] = 0

        sqrt = math.sqrt
        Uh = 0.5 * U

        diag_ind = 0
        for b in range(batch_size):
            mels[diag_ind] = 0.0
            x_prime[diag_ind] = np.copy(x[b])

            for i in range(n_sites):
                # chemical potential
                mels[diag_ind] -= mu * x[b, i]
                # on-site interaction
                mels[diag_ind] += Uh * x[b, i] * (x[b, i] - 1.0)

            odiag_ind = 1 + diag_ind
            for e in range(edges.shape[0]):
                i, j = edges[e][0], edges[e][1]
                n_i = x[b, i]
                n_j = x[b, j]
                mels[diag_ind] += V * n_i * n_j

                # destroy on i create on j
                if n_i > 0 and n_j < n_max:
                    mels[odiag_ind] = -J * sqrt(n_i) * sqrt(n_j + 1)
                    x_prime[odiag_ind] = np.copy(x[b])
                    x_prime[odiag_ind, i] -= 1.0
                    x_prime[odiag_ind, j] += 1.0
                    odiag_ind += 1

                # destroy on j create on i
                if n_j > 0 and n_i < n_max:
                    mels[odiag_ind] = -J * sqrt(n_j) * sqrt(n_i + 1)
                    x_prime[odiag_ind] = np.copy(x[b])
                    x_prime[odiag_ind, j] -= 1.0
                    x_prime[odiag_ind, i] += 1.0
                    odiag_ind += 1

            if pad:
                odiag_ind = (b + 1) * max_conn

            diag_ind = odiag_ind

            sections[b] = odiag_ind

        x_prime = x_prime[:odiag_ind]
        mels = mels[:odiag_ind]

        # if not allocated return copies
        if not x_prime_allocated:
            x_prime = np.copy(x_prime)
        if not mels_allocated:
            mels = np.copy(mels)

        return x_prime, mels

    def get_conn_flattened(self, x, sections, pad=False):
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

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        # try to cache those temporary buffers with their max size
        total_size = x.shape[0] * self._max_conn
        if self._max_mels.size < total_size:
            self._max_mels = np.empty(total_size, dtype=self._max_mels.dtype)
            self._max_xprime = np.empty((total_size, x.shape[1]), dtype=x.dtype)
        elif x.dtype != self._max_xprime.dtype:
            self._max_xprime = self._max_xprime.astype(x.dtype)

        x = concrete_or_error(
            np.asarray,
            x,
            NumbaOperatorGetConnDuringTracingError,
            self,
        )

        return self._flattened_kernel(
            x,
            sections,
            self._edges,
            self._U,
            self._V,
            self._J,
            self._mu,
            self._n_max,
            self._max_conn,
            self._max_mels,
            self._max_xprime,
            pad,
        )

    def _get_conn_flattened_closure(self):
        _edges = self._edges
        _U = self._U
        _V = self._V
        _J = self._J
        _mu = self._mu
        _n_max = self._n_max
        _max_conn = self._max_conn
        fun = self._flattened_kernel

        # do not pass the preallocated self._max_mels and self._max_xprime because they are frozen in a closure
        # and become read only
        def gccf_fun(x, sections):  # pragma: no cover
            return fun(
                x,
                sections,
                _edges,
                _U,
                _V,
                _J,
                _mu,
                _n_max,
                _max_conn,
            )

        return jit(nopython=True)(gccf_fun)
