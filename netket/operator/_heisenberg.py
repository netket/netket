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
from typing import List, Optional, Sequence, Union

import numpy as np

import jax
from jax import numpy as jnp

from netket.graph import AbstractGraph, Graph
from netket.hilbert import AbstractHilbert

from ._graph_operator import GraphOperator
from ._ising import _ising_conn_states_jax


class Heisenberg(GraphOperator):
    r"""
    The Heisenberg hamiltonian on a lattice.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        J: Union[float, Sequence[float]] = 1.0,
        sign_rule: Optional[Union[bool, Sequence[bool]]] = None,
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
        return f"{type(self).__name__}(J={self._J}, sign_rule={self._sign_rule}; dim={self.hilbert.size})"


@partial(jax.vmap, in_axes=(0, None, None, None))
def _heisenberg_mels_jax(x, edges, J, signs):
    max_conn_size = edges.shape[0] + 1

    same_spins = x[edges[:, 0]] == x[edges[:, 1]]
    mels = jnp.empty((max_conn_size,), dtype=J.dtype)
    mels = mels.at[0].set((J * (2 * same_spins - 1)).sum())
    mels = mels.at[1:].set(2 * (-2 * signs + 1) * J * (1 - same_spins))
    return mels


@partial(jax.jit, static_argnames=("local_states"))
def _heisenberg_kernel_jax(x, edges, flip, J, signs, local_states):
    batch_shape = x.shape[:-1]
    x = x.reshape((-1, x.shape[-1]))

    mels = _heisenberg_mels_jax(x, edges, J, signs)
    mels = mels.reshape(batch_shape + mels.shape[1:])

    # Same function as Ising
    x_prime = _ising_conn_states_jax(x, flip, local_states)
    x_prime = x_prime.reshape(batch_shape + x_prime.shape[1:])

    return x_prime, mels


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None))
def _heisenberg_n_conn_jax(x, edges, J):
    same_spins = x[edges[:, 0]] == x[edges[:, 1]]
    n_conn_XY = (~same_spins).sum(dtype=jnp.int32)
    # TODO duplicated with _heisenberg_mels_jax
    mels_ZZ = (J * (2 * same_spins - 1)).sum()
    n_conn_ZZ = mels_ZZ != 0
    return n_conn_XY + n_conn_ZZ


class HeisenbergJax(Heisenberg):
    @wraps(Heisenberg.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._edges_jax = jnp.asarray(self.acting_on, dtype=jnp.int32)

        flip = np.zeros((self.max_conn_size, self.hilbert.size), dtype=bool)
        for k, (i, j) in enumerate(self.acting_on, 1):
            flip[k, i] = True
            flip[k, j] = True
        self._flip = jnp.asarray(flip)

        if isinstance(self.J, Sequence):
            J = [self.J[color] for _, _, color in self.graph.edges(return_color=True)]
        else:
            J = [self.J] * self.graph.n_edges
        self._J_jax = jnp.asarray(J, dtype=self.dtype)

        if isinstance(self.uses_sign_rule, Sequence):
            signs = [
                self.uses_sign_rule[color]
                for _, _, color in self.graph.edges(return_color=True)
            ]
        else:
            signs = [self.uses_sign_rule] * self.graph.n_edges
        self._signs_jax = jnp.asarray(signs, dtype=bool)

        if len(self.hilbert.local_states) != 2:
            raise ValueError(
                "HeisenbergJax only supports Hamiltonians with two local states"
            )
        # self._local_states is assigned in LocalOperator
        self._hi_local_states = tuple(self.hilbert.local_states)

    def n_conn(self, x):
        return _heisenberg_n_conn_jax(x, self._edges_jax, self._J_jax)

    def get_conn_padded(self, x):
        return _heisenberg_kernel_jax(
            x,
            self._edges_jax,
            self._flip,
            self._J_jax,
            self._signs_jax,
            self._hi_local_states,
        )
