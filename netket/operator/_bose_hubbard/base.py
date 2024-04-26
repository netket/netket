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
from functools import reduce

import numpy as np
from jax import numpy as jnp

from netket.graph import AbstractGraph
from netket.hilbert import Fock
from netket.jax import canonicalize_dtypes
from netket.utils.numbers import dtype as _dtype
from netket.utils.types import DType

from .. import boson
from .._hamiltonian import SpecialHamiltonian
from .._local_operator import LocalOperator


class BoseHubbardBase(SpecialHamiltonian):
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
        assert isinstance(hilbert, Fock)
        super().__init__(hilbert)

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

        dtype = canonicalize_dtypes(float, U, V, J, mu, dtype=dtype)
        self._U = U.astype(dtype=dtype)
        self._V = V.astype(dtype=dtype)
        self._J = J.astype(dtype=dtype)
        self._mu = mu.astype(dtype=dtype)

        self._n_max = hilbert.n_max
        self._n_sites = hilbert.size
        self._edges = graph.astype(np.intp)
        self._max_conn = 1 + self._edges.shape[0] * 2

    @property
    def is_hermitian(self):
        return True

    @property
    def dtype(self) -> DType:
        """The dtype of the matrix elements."""
        return reduce(
            jnp.promote_types,
            (_dtype(self.U), _dtype(self.V), _dtype(self._J), _dtype(self._mu)),
        )

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

    def conjugate(self, *, concrete=True):
        # if real
        if isinstance(self.dtype, float):
            return self
        else:
            raise NotImplementedError

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self._max_conn

    def copy(self, *, dtype: Optional[DType] = None):
        if dtype is None:
            dtype = self.dtype

        return type(self)(
            hilbert=self.hilbert,
            graph=self.edges,
            U=self.U,
            V=self.V,
            J=self.J,
            mu=self.mu,
            dtype=dtype,
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
                    boson.number(self.hilbert, int(i))
                    * boson.number(self.hilbert, int(j))
                )
                ha -= self.J * (
                    boson.destroy(self.hilbert, int(i))
                    * boson.create(self.hilbert, int(j))
                    + boson.create(self.hilbert, int(i))
                    * boson.destroy(self.hilbert, int(j))
                )

        return ha

    def _iadd_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._U += other.U
        self._V += other.V
        self._J += other.J
        self._mu += other.mu

    def _isub_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._U -= other.U
        self._V -= other.V
        self._J -= other.J
        self._mu -= other.mu

    def __repr__(self):
        return f"{type(self).__name__}(U={self._U}, V={self._V}, J={self._J}, mu={self._mu}; dim={self.hilbert.size})"
