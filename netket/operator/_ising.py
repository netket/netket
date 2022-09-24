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

from functools import partial, wraps
from typing import Optional

import numpy as np
from numba import jit

import jax
from jax import numpy as jnp

from netket.graph import AbstractGraph, Graph
from netket.hilbert import AbstractHilbert
from netket.utils.numbers import dtype as _dtype
from netket.utils.types import DType

from . import spin
from ._hamiltonian import SpecialHamiltonian
from ._local_operator import LocalOperator


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
        # Fallback to float32 when float64 is disabled in JAX
        dtype = jnp.empty((), dtype=dtype).dtype
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

        graph = Graph(edges=[list(edge) for edge in self.edges])
        return type(self)(
            hilbert=self.hilbert, graph=graph, J=self.J, h=self.h, dtype=dtype
        )

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
    def _flattened_kernel(x, sections, edges, h, J):  # pragma: no cover
        n_sites = x.shape[1]
        n_conn = n_sites + 1

        x_prime = np.empty((x.shape[0] * n_conn, n_sites), dtype=x.dtype)
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
        return self._flattened_kernel(
            np.asarray(x), sections, self._edges, self._h, self._J
        )

    def _get_conn_flattened_closure(self):
        _edges = self._edges
        _h = self._h
        _J = self._J
        fun = self._flattened_kernel

        def gccf_fun(x, sections):  # pragma: no cover
            return fun(x, sections, _edges, _h, _J)

        return jit(nopython=True)(gccf_fun)

    def __repr__(self):
        return (
            f"{type(self).__name__}(J={self._J}, h={self._h}; dim={self.hilbert.size})"
        )


@partial(jax.vmap, in_axes=(0, None, None, None))
def _ising_mels_jax(x, edges, h, J):
    if h == 0:
        max_conn_size = 1
    else:
        max_conn_size = x.size + 1

    same_spins = x[edges[:, 0]] == x[edges[:, 1]]
    mels = jnp.empty((max_conn_size,), dtype=J.dtype)
    mels = mels.at[0].set(J * (2 * same_spins - 1).sum())
    if h != 0:
        mels = mels.at[1:].set(-h)
    return mels


def _flip_if(cond, x, local_states):
    # TODO here we could special-case for qubit / ising
    # by taking -x / 1 - x
    # i.e
    # if local_states[0] + local_states[1] == 0:
    #     return jnp.where(cond, -x, x)
    # elif local_states[0] == 0:
    #     return jnp.where(cond, local_states[1] - x, x)
    # elif local_states[1] == 0:
    #     return jnp.where(cond, local_states[0] - x, x)
    # else:
    #     ...
    was_state_0 = x == local_states[0]
    state_0 = jnp.asarray(local_states[0], dtype=x.dtype)
    state_1 = jnp.asarray(local_states[1], dtype=x.dtype)
    return jnp.where(cond ^ was_state_0, state_0, state_1)


@partial(jax.vmap, in_axes=(0, None, None))
def _ising_conn_states_jax(x, flip, local_states):
    return _flip_if(flip, x, local_states)


@partial(jax.jit, static_argnames=("h", "local_states"))
def _ising_kernel_jax(x, edges, flip, h, J, local_states):
    batch_shape = x.shape[:-1]
    x = x.reshape((-1, x.shape[-1]))

    mels = _ising_mels_jax(x, edges, h, J)
    mels = mels.reshape(batch_shape + mels.shape[1:])

    if h == 0:
        x_prime = jnp.expand_dims(x, axis=1)
    else:
        x_prime = _ising_conn_states_jax(x, flip, local_states)
    x_prime = x_prime.reshape(batch_shape + x_prime.shape[1:])

    return x_prime, mels


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None))
def _ising_n_conn_jax(x, edges, h, J):
    n_sites = x.size
    n_conn_X = jnp.asarray(h != 0, dtype=jnp.int32) * n_sites
    same_spins = x[edges[:, 0]] == x[edges[:, 1]]
    # TODO duplicated with _ising_mels_jax
    mels_ZZ = J * (2 * same_spins - 1).sum()
    n_conn_ZZ = mels_ZZ != 0
    return n_conn_X + n_conn_ZZ


class IsingJax(Ising):
    @wraps(Ising.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._edges_jax = jnp.asarray(self.edges, dtype=jnp.int32)
        if self.h == 0:
            self._flip = None
        else:
            self._flip = jnp.eye(
                self.max_conn_size, self.hilbert.size, k=-1, dtype=bool
            )

        if len(self.hilbert.local_states) != 2:
            raise ValueError(
                "IsingJax only supports Hamiltonians with two local states"
            )
        self._hi_local_states = tuple(self.hilbert.local_states)

    def n_conn(self, x):
        return _ising_n_conn_jax(x, self._edges_jax, self.h, self.J)

    def get_conn_padded(self, x):
        return _ising_kernel_jax(
            x, self._edges_jax, self._flip, self.h.item(), self.J, self._hi_local_states
        )
