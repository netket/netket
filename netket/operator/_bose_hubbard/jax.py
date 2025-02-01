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
from netket.hilbert import Fock
from netket.utils.types import DType

from .._discrete_operator_jax import DiscreteJaxOperator

from .base import BoseHubbardBase

if TYPE_CHECKING:
    from .numba import BoseHubbard


@register_pytree_node_class
class BoseHubbardJax(BoseHubbardBase, DiscreteJaxOperator):
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
        U, V, J, mu = jax.tree_util.tree_map(jnp.asarray, (U, V, J, mu))
        super().__init__(hilbert, graph=graph, U=U, V=V, J=J, mu=mu, dtype=dtype)
        self._edges = jnp.asarray(self.edges, dtype=jnp.int32)

        self._n_max = self.hilbert.n_max

    @jax.jit
    @wraps(BoseHubbardBase.get_conn_padded)
    def get_conn_padded(self, x):
        x_ids = self.hilbert.states_to_local_indices(x)

        # Note that the calculation of the matrix elements will be done
        # in double precision, and we cast to the dtype of the oeprator
        # only at the end. Otherwise jnp.sqrt might return 'bad' results.
        xp_ids, mels = _bh_kernel_jax(
            x_ids, self._edges, self.U, self.V, self.J, self.mu, self._n_max
        )
        xp = self.hilbert.local_indices_to_states(xp_ids, dtype=x.dtype)
        return xp, mels.astype(self.dtype)

    def to_numba_operator(self) -> "BoseHubbard":  # noqa: F821
        """
        Returns the standard (numba) version of this operator, which is an
        instance of {class}`nk.operator.BoseHubbard`.
        """

        from .numba import BoseHubbard

        return BoseHubbard(
            self.hilbert,
            graph=self.edges,
            U=self.U,
            V=self.V,
            J=self.J,
            mu=self.mu,
            dtype=self.dtype,
        )

    def to_local_operator(self):
        # The hamiltonian
        ha = super().to_local_operator()

        return ha.to_jax_operator()

    def tree_flatten(self):
        data = (self.U, self.V, self.J, self.mu, self.edges)
        metadata = {"hilbert": self.hilbert}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        U, V, J, mu, edges = data
        hi = metadata["hilbert"]
        res = cls(hi, graph=[(0, 0)], U=1.0)
        res._U = U
        res._V = V
        res._J = J
        res._mu = mu
        res._edges = edges
        return res


@partial(jax.jit, static_argnames="n_max")
def _bh_kernel_jax(x, edges, U, V, J, mu, n_max):
    i = edges[:, 0]
    j = edges[:, 1]
    n_i = jnp.vectorize(lambda x: x[i], signature="(n)->(m)")(x)
    n_j = jnp.vectorize(lambda x: x[j], signature="(n)->(m)")(x)

    Uh = 0.5 * U

    _x = jnp.expand_dims(x, axis=-2)
    xp0 = _x
    mels0 = 0
    mels0 -= mu * x.sum(axis=-1, keepdims=True)
    mels0 += Uh * (x * (x - 1)).sum(axis=-1, keepdims=True)
    mels0 += V * (n_i * n_j).sum(axis=-1, keepdims=True)
    mask0 = jnp.full((*x.shape[:-1], 1), True)

    add_at = jax.vmap(
        lambda x, idx, addend: x.at[..., idx].add(addend), (-2, 0, None), -2
    )

    # x might be stored in lower (int8, int16...) precision, which will
    # cause jnp.sqrt(n) to be computed in float16, float32 precision, which
    # we have seen to lead to some numerical precision errors.
    # so in order to enforce 64 bit precision we need to case it to 64 bits
    h_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    n_i_64 = n_i.astype(h_dtype)
    n_j_64 = n_j.astype(h_dtype)

    # destroy on i create on j
    mask1 = (n_i > 0) & (n_j < n_max)
    mels1 = mask1 * (-J * jnp.sqrt(n_i_64) * jnp.sqrt(n_j_64 + 1))
    xp1 = jnp.repeat(_x, mask1.shape[-1], axis=-2)
    xp1 = add_at(xp1, i, -1)
    xp1 = add_at(xp1, j, +1)

    # destroy on j create on i
    mask2 = (n_j > 0) & (n_i < n_max)
    mels2 = mask2 * (-J * jnp.sqrt(n_j_64) * jnp.sqrt(n_i_64 + 1))
    xp2 = jnp.repeat(_x, mask2.shape[-1], axis=-2)
    xp2 = add_at(xp2, j, -1)
    xp2 = add_at(xp2, i, +1)

    mask = jnp.concatenate([mask0, mask1, mask2], axis=-1)
    mels = jnp.concatenate([mels0, mels1, mels2], axis=-1)
    xp = jnp.concatenate([xp0, xp1, xp2], axis=-2)

    xp = jnp.vectorize(jax.lax.select, signature="(),(n),(n)->(n)")(mask, xp, _x)

    return xp, mels
