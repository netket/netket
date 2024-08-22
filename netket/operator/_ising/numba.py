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

from netket.graph import AbstractGraph
from netket.hilbert import Spin
from netket.utils.types import DType
from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError

from .base import IsingBase

if TYPE_CHECKING:
    from .jax import IsingJax


class Ising(IsingBase):
    r"""
    The Transverse-Field Ising Hamiltonian :math:`-h\sum_i \sigma_i^{(x)} +J\sum_{\langle i,j\rangle} \sigma_i^{(z)}\sigma_j^{(z)}`.

    This implementation is considerably faster than the Ising hamiltonian constructed by summing :class:`~netket.operator.LocalOperator` s.
    """

    @wraps(IsingBase.__init__)
    def __init__(
        self,
        hilbert: Spin,
        graph: AbstractGraph,
        h: float,
        J: float = 1.0,
        dtype: DType | None = None,
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
        if not isinstance(hilbert, Spin):
            raise TypeError(
                """The Hilbert space used by Ising must be a `Spin-1/2` space.

                This limitation could be lifted by 'fixing' the method
                `_flattened_kernel` to work with arbitrary hilbert spaces, which
                should be relatively straightforward to do, but we have not done so
                yet.

                In the meantime, you can just use `nk.operator.IsingJax` as a
                workaround.
                """
            )
        if len(hilbert.local_states) != 2:
            raise ValueError("Ising only supports Spin-1/2 hilbert spaces.")

        h = np.array(h, dtype=dtype)
        J = np.array(J, dtype=dtype)
        if isinstance(graph, jax.Array):
            graph = np.asarray(graph)
        super().__init__(hilbert, graph=graph, h=h, J=J, dtype=dtype)

    def to_jax_operator(self) -> "IsingJax":  # noqa: F821
        """
        Returns the jax-compatible version of this operator, which is an
        instance of :class:`netket.operator.IsingJax`.
        """
        from .jax import IsingJax

        return IsingJax(
            self.hilbert, graph=self.edges, h=self.h, J=self.J, dtype=self.dtype
        )

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(x, sections, edges, h, J):  # pragma: no cover
        n_sites = x.shape[1]
        n_conn = n_sites + 1

        x_prime = np.empty((x.shape[0] * n_conn, n_sites), dtype=x.dtype)
        mels = np.empty(x.shape[0] * n_conn, dtype=h.dtype)

        diag_ind = 0

        for i in range(x.shape[0]):
            mels[diag_ind] = 0.0
            for k in range(edges.shape[0]):
                mels[diag_ind] += (
                    J * (2 * x[i, edges[k, 0]] - 1) * (2 * x[i, edges[k, 1]] - 1)
                )

            odiag_ind = 1 + diag_ind

            mels[odiag_ind : (odiag_ind + n_sites)].fill(-h)

            x_prime[diag_ind : (diag_ind + n_conn)] = np.copy(x[i])

            for j in range(n_sites):
                x_prime[j + odiag_ind][j] = np.mod(x_prime[j + odiag_ind][j] + 1, 2)

            diag_ind += n_conn

            sections[i] = diag_ind

        return x_prime, mels

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for
        :math:`k=0,1...N_{\mathrm{connected}}`.

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
        x = concrete_or_error(
            np.asarray,
            x,
            NumbaOperatorGetConnDuringTracingError,
            self,
        )
        x_ids = self.hilbert.states_to_local_indices(x)

        xp_ids, mels = self._flattened_kernel(
            x_ids, sections, self.edges, self._h, self._J
        )
        xp = self.hilbert.local_indices_to_states(xp_ids, dtype=x.dtype)
        return xp, mels
