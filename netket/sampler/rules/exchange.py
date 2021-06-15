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

from typing import Optional, List, Any

import jax
import numpy as np

from jax import numpy as jnp
from flax import struct

from netket.graph import AbstractGraph
from ..metropolis import MetropolisRule


@struct.dataclass
class ExchangeRule_(MetropolisRule):
    r"""
    A Rule exchanging the state on a random couple of sites, chosen from a list of
    possible couples (clusters).

    This rule acts on two local degree of freedom :math:`s_i` and :math:`s_j`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N`,
    where in general :math:`s^\prime_i \neq s_i` and :math:`s^\prime_j \neq s_j`.
    The sites :math:`i` and :math:`j` are also chosen to be within a maximum graph
    distance of :math:`d_{\mathrm{max}}`.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A pair of indices :math:`i,j = 1\dots N`, and such
       that :math:`\mathrm{dist}(i,j) \leq d_{\mathrm{max}}`,
       is chosen with uniform probability.
    2. The sites are exchanged, i.e. :math:`s^\prime_i = s_j` and :math:`s^\prime_j = s_i`.

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum numbers
    are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where :math:`\sum_i s_i = \mathrm{constant}` is needed,
    otherwise the sampling would be strongly not ergodic.
    """

    clusters: Any

    def transition(rule, sampler, machine, parameters, state, key, σ):
        n_chains = σ.shape[0]

        # pick a random cluster
        cluster_id = jax.random.randint(
            key, shape=(n_chains,), minval=0, maxval=rule.clusters.shape[0]
        )

        def scalar_update_fun(σ, cluster):
            # sites to be exchanged,
            si = rule.clusters[cluster, 0]
            sj = rule.clusters[cluster, 1]

            σp = jax.ops.index_update(σ, si, σ[sj])
            return jax.ops.index_update(σp, sj, σ[si])

        return (
            jax.vmap(scalar_update_fun, in_axes=(0, 0), out_axes=0)(σ, cluster_id),
            None,
        )

    def __repr__(self):
        return f"ExchangeRule(# of clusters: {len(self.clusters)})"


def compute_clusters(graph, d_max):
    clusters = []
    distances = np.asarray(graph.distances())
    size = distances.shape[0]
    for i in range(size):
        for j in range(i + 1, size):
            if distances[i][j] <= d_max:
                clusters.append((i, j))

    res_clusters = np.empty((len(clusters), 2), dtype=np.int64)

    for i, cluster in enumerate(clusters):
        res_clusters[i] = np.asarray(cluster)

    return res_clusters


def ExchangeRule(
    *,
    clusters: Optional[List[List[int]]] = None,
    graph: Optional[AbstractGraph] = None,
    d_max: int = 1,
):
    r"""
    A Rule exchanging the state on a random couple of sites, chosen from a list of
    possible couples (clusters).

    This rule acts on two local degree of freedom :math:`s_i` and :math:`s_j`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N`,
    where in general :math:`s^\prime_i \neq s_i` and :math:`s^\prime_j \neq s_j`.
    The sites :math:`i` and :math:`j` are also chosen to be within a maximum graph
    distance of :math:`d_{\mathrm{max}}`.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A pair of indices :math:`i,j = 1\dots N`, and such
       that :math:`\mathrm{dist}(i,j) \leq d_{\mathrm{max}}`,
       is chosen with uniform probability.
    2. The sites are exchanged, i.e. :math:`s^\prime_i = s_j` and :math:`s^\prime_j = s_i`.

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum numbers
    are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where :math:`\sum_i s_i = \mathrm{constant}` is needed,
    otherwise the sampling would be strongly not ergodic.

    Args:
        clusters: The list of clusters that can be exchanged.
        graph: A graph, from which the edges determine the clusters that can be exchanged.
        d_max: Only valid if a graph is passed in. The maximum distance between two sites
    """
    if clusters is None and graph is not None:
        clusters = compute_clusters(graph, d_max)
    elif not (clusters is not None and graph is None):
        raise ValueError(
            """You must either provide the list of exchange-clusters or a netket graph, from
                          which clusters will be computed using the maximum distance d_max. """
        )

    return ExchangeRule_(jnp.array(clusters))
