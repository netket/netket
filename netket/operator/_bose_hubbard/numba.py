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

from functools import wraps
from typing import TYPE_CHECKING

import jax

import numpy as np
from numba import jit
import math

from netket.graph import AbstractGraph
from netket.hilbert import Fock
from netket.utils.types import DType
from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError

from .base import BoseHubbardBase

if TYPE_CHECKING:
    from .jax import BoseHubbardJax


class BoseHubbard(BoseHubbardBase):
    r"""
    An extended Bose Hubbard model Hamiltonian operator, containing both
    on-site interactions and nearest-neighboring density-density interactions.
    """

    @wraps(BoseHubbardBase.__init__)
    def __init__(
        self,
        hilbert: Fock,
        graph: AbstractGraph,
        U: float,
        V: float = 0.0,
        J: float = 1.0,
        mu: float = 0.0,
        dtype: DType | None = None,
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

        U = np.asarray(U)
        V = np.asarray(V)
        J = np.asarray(J)
        mu = np.asarray(mu)
        if isinstance(graph, jax.Array):
            graph = np.asarray(graph)
        super().__init__(hilbert, graph=graph, U=U, V=V, J=J, mu=mu, dtype=dtype)

        # caches for numba indexing methods
        self._max_mels = None  # np.zeros(self._max_conn, dtype=self.dtype)
        self._max_xprime = None  # np.zeros((self._max_conn, self._n_sites), dtype=)

    def to_jax_operator(self) -> "BoseHubbardJax":  # noqa: F821
        """
        Returns the jax-compatible version of this operator, which is an
        instance of :class:`netket.operator.IsingJax`.
        """
        from .jax import BoseHubbardJax

        return BoseHubbardJax(
            self.hilbert,
            graph=self.edges,
            U=self.U,
            V=self.V,
            J=self.J,
            mu=self.mu,
            dtype=self.dtype,
        )

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
            x_prime[:] = np.array(0, dtype=x_prime.dtype)
            mels[:] = np.array(0, dtype=x_prime.dtype)

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
                x_prime[odiag_ind : (b + 1) * max_conn] = np.copy(x[b])
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
        x_ids = self.hilbert.states_to_local_indices(x)
        x_ids = concrete_or_error(
            np.asarray,
            x_ids,
            NumbaOperatorGetConnDuringTracingError,
            self,
        )

        # try to cache those temporary buffers with their max size
        total_size = x.shape[0] * self._max_conn
        if self._max_xprime is None or (self._max_mels.size < total_size):
            self._max_mels = np.empty(total_size, dtype=self.dtype)
            self._max_xprime = np.empty((total_size, x.shape[-1]), dtype=x_ids.dtype)
        elif x.dtype != self._max_xprime.dtype:
            self._max_xprime = self._max_xprime.astype(x_ids.dtype)

        xp_ids, mels = self._flattened_kernel(
            x_ids,
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
        xp = self.hilbert.local_indices_to_states(xp_ids, dtype=x.dtype)
        return xp, mels
