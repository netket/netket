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

from typing import List, Sequence, Union
from numba import jit

import numpy as np
import math

from netket.graph import AbstractGraph, Graph
from netket.hilbert import AbstractHilbert, Fock
from netket.utils.types import DType

from . import spin, boson
from ._local_operator import LocalOperator
from ._graph_operator import GraphOperator
from ._discrete_operator import DiscreteOperator


class SpecialHamiltonian(DiscreteOperator):
    def to_local_operator(self):
        raise NotImplementedError(
            "Must implemented to_local_operator for {}".format(type(self))
        )

    def conjugate(self, *, concrete: bool = True):
        return self.to_local_operator().conjugate(concrete=concrete)

    def __add__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res += other
            return res

        return self.to_local_operator().__add__(other)

    def __sub__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res -= other
            return res

        return self.to_local_operator().__sub__(other)

    def __radd__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res += other
            return res

        return self.to_local_operator().__radd__(other)

    def __rsub__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res -= other
            return res

        return self.to_local_operator().__rsub__(other)

    def __iadd__(self, other):
        if type(self) is type(other):
            self._iadd_same_hamiltonian(other)
            return self

        return NotImplemented

    def __isub__(self, other):
        if type(self) is type(other):
            self._isub_same_hamiltonian(other)
            return self

        return NotImplemented

    def __mul__(self, other):
        return self.to_local_operator().__mul__(other)

    def __rmul__(self, other):
        return self.to_local_operator().__rmul__(other)

    def _op__matmul__(self, other):
        if hasattr(other, "to_local_operator"):
            other = other.to_local_operator()
        return self.to_local_operator().__matmul__(other)

    def _op__rmatmul__(self, other):
        if hasattr(other, "to_local_operator"):
            other = other.to_local_operator()

        return self.to_local_operator().__matmul__(other)


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
        dtype: DType = float,
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

        self._h = dtype(h)
        self._J = dtype(J)
        self._edges = np.asarray(
            [[u, v] for u, v in graph.edges()],
            dtype=np.intp,
        )

        self._dtype = dtype

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

    def copy(self):
        graph = Graph(edges=[list(edge) for edge in self.edges])
        return Ising(hilbert=self.hilbert, graph=graph, J=self.J, h=self.h)

    def to_local_operator(self):
        # The hamiltonian
        ha = LocalOperator(self.hilbert, dtype=self.dtype)

        if self.h != 0:
            for i in range(self.hilbert.size):
                ha -= self.h * spin.sigmax(self.hilbert, i)

        if self.J != 0:
            for (i, j) in self.edges:
                ha += self.J * (
                    spin.sigmaz(self.hilbert, i) * spin.sigmaz(self.hilbert, j)
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
            )
        )
        mels = np.empty(x.shape[0] * n_conn, dtype=type(h))

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


class Heisenberg(GraphOperator):
    r"""
    The Heisenberg hamiltonian on a lattice.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        J: Union[float, Sequence[float]] = 1.0,
        sign_rule=None,
        *,
        acting_on_subspace: Union[List[int], int] = None,
    ):
        """
        Constructs an Heisenberg operator given a hilbert space and a graph providing the
        connectivity of the lattice.

        Args:
            hilbert: Hilbert space the operator acts on.
            graph: The graph upon which this hamiltonian is defined.
            J: The strength of the coupling. Default is 1.
               Can pass a sequence of coupling strengths with coloured graphs:
               edges of colour n will have coupling strength J[n]
            sign_rule: If True, Marshal's sign rule will be used. On a bipartite
                lattice, this corresponds to a basis change flipping the Sz direction
                at every odd site of the lattice. For non-bipartite lattices, the
                sign rule cannot be applied. Defaults to True if the lattice is
                bipartite, False otherwise.
                If a sequence of coupling strengths is passed, defaults to False
                and a matching sequence of sign_rule must be specified to override it
            acting_on_subspace: Specifies the mapping between nodes of the graph and
                Hilbert space sites, so that graph node :code:`i ∈ [0, ..., graph.n_nodes - 1]`,
                corresponds to :code:`acting_on_subspace[i] ∈ [0, ..., hilbert.n_sites]`.
                Must be a list of length `graph.n_nodes`. Passing a single integer :code:`start`
                is equivalent to :code:`[start, ..., start + graph.n_nodes - 1]`.

        Examples:
         Constructs a ``Heisenberg`` operator for a 1D system.

            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
            >>> op = nk.operator.Heisenberg(hilbert=hi, graph=g)
            >>> print(op)
            Heisenberg(J=1.0, sign_rule=True; dim=20)
        """
        if isinstance(J, Sequence):
            # check that the number of Js matches the number of colours
            assert len(J) == max(graph.edge_colors) + 1

            if sign_rule is None:
                sign_rule = [False] * len(J)
            else:
                assert len(sign_rule) == len(J)
                for i in range(len(J)):
                    subgraph = Graph(edges=graph.edges(filter_color=i))
                    if sign_rule[i] and not subgraph.is_bipartite():
                        raise ValueError(
                            "sign_rule=True specified for a non-bipartite lattice"
                        )
        else:
            if sign_rule is None:
                sign_rule = graph.is_bipartite()
            elif sign_rule and not graph.is_bipartite():
                raise ValueError("sign_rule=True specified for a non-bipartite lattice")

        self._J = J
        self._sign_rule = sign_rule

        sz_sz = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
        exchange = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        if isinstance(J, Sequence):
            bond_ops = [
                J[i] * (sz_sz - exchange if sign_rule[i] else sz_sz + exchange)
                for i in range(len(J))
            ]
            bond_ops_colors = list(range(len(J)))
        else:
            bond_ops = [J * (sz_sz - exchange if sign_rule else sz_sz + exchange)]
            bond_ops_colors = []

        super().__init__(
            hilbert,
            graph,
            bond_ops=bond_ops,
            bond_ops_colors=bond_ops_colors,
            acting_on_subspace=acting_on_subspace,
        )

    @property
    def J(self) -> float:
        """The coupling strength."""
        return self._J

    @property
    def uses_sign_rule(self):
        return self._sign_rule

    def __repr__(self):
        return f"Heisenberg(J={self._J}, sign_rule={self._sign_rule}; dim={self.hilbert.size})"


class BoseHubbard(SpecialHamiltonian):
    r"""
    An extended Bose Hubbard model Hamiltonian operator, containing both
    on-site interactions and nearest-neighboring density-density interactions.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        U: float,
        V: float = 0.0,
        J: float = 1.0,
        mu: float = 0.0,
        dtype: DType = float,
    ):
        r"""
        Constructs a new BoseHubbard operator given a hilbert space, a graph
        specifying the connectivity and the interaction strength.
        The chemical potential and the density-density interaction strenght
        can be specified as well.

        Args:
           hilbert: Hilbert space the operator acts on.
           U: The on-site interaction term.
           V: The strength of density-density interaction term.
           J: The hopping amplitude.
           mu: The chemical potential.
           dtype: The dtype of the matrix eleements.

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

        self._U = dtype(U)
        self._V = dtype(V)
        self._J = dtype(J)
        self._mu = dtype(mu)
        self._dtype = dtype

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
            for (i, j) in self.edges:
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

    def get_conn(self, x):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (array): An array of shape (hilbert.size) containing the quantum numbers x.

        Returns:
            matrix: The connected states x' of shape (N_connected,hilbert.size)
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        mels = self._max_mels
        x_prime = self._max_xprime

        mels[0] = 0.0
        x_prime[0] = np.copy(x)

        J = self._J
        V = self._V
        sqrt = math.sqrt
        n_max = self._n_max

        c = 1
        for e in self._edges:
            i, j = e
            n_i = x[i]
            n_j = x[j]
            mels[0] += V * n_i * n_j

            # destroy on i create on j
            if n_i > 0 and n_j < n_max:
                mels[c] = -J * sqrt(n_i) * sqrt(n_j + 1)
                x_prime[c] = np.copy(x)
                x_prime[c, i] -= 1.0
                x_prime[c, j] += 1.0
                c += 1

            # destroy on j create on i
            if n_j > 0 and n_i < n_max:
                mels[c] = -J * sqrt(n_j) * sqrt(n_i + 1)
                x_prime[c] = np.copy(x)
                x_prime[c, j] -= 1.0
                x_prime[c, i] += 1.0
                c += 1

        mu = self._mu
        Uh = 0.5 * self._U
        for i in range(self._n_sites):
            # chemical potential
            mels[0] -= mu * x[i]
            # on-site interaction
            mels[0] += Uh * x[i] * (x[i] - 1.0)

        return np.copy(x_prime[:c]), np.copy(mels[:c])

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(
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
            mels = np.empty(batch_size * max_conn, dtype=type(U))

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
        else:
            if x.dtype != self._max_xprime.dtype:
                self._max_xprime = self._max_xprime.astype(x.dtype)

        return self._flattened_kernel(
            np.asarray(x),
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

        # do not pass the preallocated self._max_mels andself._max_xprime because they are frozen in a closure
        # and become read only
        def gccf_fun(x, sections):
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
