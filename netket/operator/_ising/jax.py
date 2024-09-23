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
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.graph import AbstractGraph
from netket.hilbert import AbstractHilbert
from netket.utils.numbers import StaticZero
from netket.utils.types import DType

from .._discrete_operator_jax import DiscreteJaxOperator

from .base import IsingBase

if TYPE_CHECKING:
    from .numba import Ising


@register_pytree_node_class
class IsingJax(IsingBase, DiscreteJaxOperator):
    """
    Jax-compatible version of :class:`netket.operator.Ising`.
    """

    @wraps(IsingBase.__init__)
    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        h: float,
        J: float = 1.0,
        dtype: DType | None = None,
    ):
        if len(hilbert.local_states) != 2:
            raise ValueError(
                "IsingJax only supports Hamiltonians with two local states"
            )

        if not isinstance(h, jax.Array) and (h == 0 or h is None):
            h = StaticZero()

        J = jnp.array(J, dtype=dtype)
        if not isinstance(h, StaticZero):
            h = jnp.array(h, dtype=dtype)

        super().__init__(hilbert, graph=graph, h=h, J=J, dtype=dtype)

        self._edges = jnp.asarray(self.edges, dtype=jnp.int32)

    @jax.jit
    @wraps(IsingBase.n_conn)
    def n_conn(self, x):
        x_ids = self.hilbert.states_to_local_indices(x)
        return _ising_n_conn_jax(x_ids, self._edges, self.h, self.J)

    @jax.jit
    @wraps(IsingBase.get_conn_padded)
    def get_conn_padded(self, x):
        x_ids = self.hilbert.states_to_local_indices(x)
        xp_ids, mels = _ising_kernel_jax(x_ids, self._edges, self.h, self.J)
        xp = self.hilbert.local_indices_to_states(xp_ids, dtype=x.dtype)
        return xp, mels

    def to_numba_operator(self) -> "Ising":  # noqa: F821
        """
        Returns the standard (numba) version of this operator, which is an
        instance of {class}`nk.operator.Ising`.
        """

        from .numba import Ising

        return Ising(
            self.hilbert, graph=self.edges, h=self.h, J=self.J, dtype=self.dtype
        )

    def to_local_operator(self):
        # The hamiltonian
        ha = super().to_local_operator()

        return ha.to_jax_operator()

    def tree_flatten(self):
        data = (self.h, self.J, self.edges)
        metadata = {"hilbert": self.hilbert}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        h, J, edges = data
        hi = metadata["hilbert"]

        res = cls(hi, h=1.0, graph=[(0, 0)])
        res._h = h
        res._J = J
        res._edges = edges
        return res


def _ising_mels_jax(x, edges, h, J):
    batch_dims = x.shape[:-1]
    if isinstance(h, StaticZero):
        max_conn_size = 1
    else:
        max_conn_size = x.shape[-1] + 1
    mels = jnp.zeros((*batch_dims, max_conn_size), dtype=J.dtype)

    same_spins = x[..., edges[:, 0]] == x[..., edges[:, 1]]
    mels = mels.at[..., 0].set(J * (2 * same_spins - 1).sum(axis=-1))
    if not isinstance(h, StaticZero):
        mels = mels.at[..., 1:].set(-h)
    return mels


def _ising_conn_states_jax(x, cond):
    was_state_0 = x == 0
    state_0 = jnp.asarray(0, dtype=x.dtype)
    state_1 = jnp.asarray(1, dtype=x.dtype)
    return jnp.where(cond ^ was_state_0, state_0, state_1)


@partial(jax.jit)
def _ising_kernel_jax(x, edges, h, J):
    hilb_size = x.shape[-1]
    batch_shape = x.shape[:-1]
    x = x.reshape((-1, hilb_size))

    mels = _ising_mels_jax(x, edges, h, J)
    mels = mels.reshape(batch_shape + mels.shape[1:])

    if isinstance(h, StaticZero):
        x_prime = jnp.expand_dims(x, axis=1)
    else:
        max_conn_size = hilb_size + 1
        flip = jnp.eye(max_conn_size, hilb_size, k=-1, dtype=bool)
        x_prime = _ising_conn_states_jax(x[..., None, :], flip)

    x_prime = x_prime.reshape(batch_shape + x_prime.shape[1:])

    return x_prime, mels


@jax.jit
def _ising_n_conn_jax(x, edges, h, J):
    n_conn_X = 0 if isinstance(h, StaticZero) else x.shape[-1]
    same_spins = x[..., edges[:, 0]] == x[..., edges[:, 1]]
    # TODO duplicated with _ising_mels_jax
    mels_ZZ = J * (2 * same_spins - 1).sum(axis=-1)
    n_conn_ZZ = jnp.asarray(mels_ZZ != 0, dtype=jnp.int32)
    return n_conn_X + n_conn_ZZ
