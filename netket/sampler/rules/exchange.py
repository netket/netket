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

import jax
import numpy as np

from jax import numpy as jnp

from netket.graph import AbstractGraph

from .base import MetropolisRule


class ExchangeRule(MetropolisRule):
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
       is chosen with uniform probability, excluding all the sites where
       :math:`s_i = s_j`;
    2. The sites are exchanged, i.e. :math:`s^\prime_i = s_j` and :math:`s^\prime_j = s_i`;

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum numbers
    are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where :math:`\sum_i s_i = \mathrm{constant}` is needed,
    otherwise the sampling would be strongly not ergodic.

    .. warning::

        If you are working with systems where the number of nodes in the physical lattice
        does not match the number of degrees of freedom, you must be careful!

        A typical example is a system of Spin-1/2 fermions on a lattice with N sites, where the
        first N degrees of freedom correspond to the spin down degrees of freedom and the
        next N degrees of freedom correspond to the spin up degrees of freedom.

        In this case, you tipically want to exchange only degrees of freedom of the same type.
        A simple way to achieve this is to double the graph:

        .. code-block:: python

            import netket as nk
            g = nk.graph.Square(5)
            hi = nk.hilbert.SpinOrbitalFermions(g.n_nodes, s=0.5)

            exchange_graph = nk.graph.disjoint_union(g, g)
            print("Exchange graph size:", exchange_graph.n_nodes)

            sa = nk.sampler.MetropolisExchange(hi, graph=exchange_graph, d_max=1)

    """

    clusters: jax.Array
    r"""2-Dimensional tensor :math:`T_{i,j}` of shape
    :math:`N_\text{clusters}\times 2` where the first dimension
    runs over the list of 2-site clusters and the second dimension
    runs over the 2 sites of those clusters.

    The Exchange rule will swap the two sites of a random row of this
    matrix at every Metropolis step.
    """
    probabilities: jax.Array
    r"""1-Dimensional array :math:`p_{i}` of length :math:`N_\text{clusters}`
    containing the relative probability of choosing each particular cluster
    for the exchange update.
    """

    def __init__(
        self,
        *,
        clusters: list[tuple[int, int]] | None = None,
        graph: AbstractGraph | None = None,
        d_max: int = 1,
        probabilities: list[float] | None = None,
    ):
        r"""
        Constructs the Exchange Rule.

        You can pass either a list of clusters or a netket graph object to
        determine the clusters to exchange.

        Args:
            clusters: The list of clusters that can be exchanged. This should be
                a list of 2-tuples containing two integers. Every tuple is an edge,
                or cluster of sites to be exchanged.
            graph: The graph for determining the clusters that can be exchanged.
                Do not specify together with `clusters`.
            d_max: The maximum graph distance allowed for exchanges.
                Only used if `graph` is specified.
            probabilities: Relative probability of trying each cluster
                (if `clusters` is given) or clusters with each graph distance
                (if `graph` is given). Defaults to equal probability for all clusters.
        """
        if probabilities is not None:
            probabilities = np.atleast_1d(probabilities)
            if not np.all(probabilities > 0):
                raise ValueError("Probabilities must be positive")
        if clusters is None and graph is not None:
            clusters, probabilities = compute_clusters(graph, d_max, probabilities)
        elif not (clusters is not None and graph is None):
            raise ValueError(
                """You must either provide the list of exchange-clusters or a netket graph, from
                              which clusters will be computed using the maximum distance d_max. """
            )

        self.clusters = jnp.array(clusters)

        if probabilities is None:
            self.probabilities = jnp.ones(len(clusters))
        else:
            if len(probabilities) != len(clusters):
                # shouldn't get here if both are computed from graphs
                raise TypeError(
                    "Number of clusters and probabilities don't match: "
                    f"{len(clusters)} != {len(probabilities)}"
                )
            self.probabilities = jnp.asarray(probabilities)

    def transition(rule, sampler, machine, parameters, state, key, σ):
        n_chains = σ.shape[0]

        # compute a mask for the clusters that can be hopped
        hoppable_clusters = _compute_different_clusters_mask(rule.clusters, σ)

        keys = jnp.asarray(jax.random.split(key, n_chains))

        @jax.vmap
        def _update_samples(key, σ, hoppable_clusters):
            # pick a random cluster, taking into account the mask
            n_conn = hoppable_clusters @ rule.probabilities
            cluster = jax.random.choice(
                key,
                a=jnp.arange(rule.clusters.shape[0]),
                p=hoppable_clusters * rule.probabilities,
                replace=True,
            )

            # sites to be exchanged
            si = rule.clusters[cluster, 0]
            sj = rule.clusters[cluster, 1]

            σp = σ.at[si].set(σ[sj])
            σp = σp.at[sj].set(σ[si])

            # compute the number of connected sites
            hoppable_clusters_proposed = _compute_different_clusters_mask(
                rule.clusters, σp
            )
            n_conn_proposed = hoppable_clusters_proposed @ rule.probabilities
            log_prob_corr = jnp.log(n_conn) - jnp.log(n_conn_proposed)
            return σp, log_prob_corr

        return _update_samples(keys, σ, hoppable_clusters)

    def __repr__(self):
        return f"ExchangeRule(# of clusters: {len(self.clusters)})"


def compute_clusters(graph: AbstractGraph, d_max: int, prob: np.ndarray):
    """
    Given a netket graph and a maximum distance, computes all clusters.
    If `d_max = 1` this is equivalent to taking the edges of the graph.
    Then adds next-nearest neighbors and so on.
    """
    distances = np.asarray(graph.distances())
    clusters = np.argwhere(distances <= d_max)
    # keep one copy of i != j clusters
    clusters = clusters[clusters[:, 0] < clusters[:, 1]]

    if prob is not None:
        assert prob.shape == (d_max,), f"Expected {d_max = } probabilities, got {prob}"
        prob = prob[distances[clusters[:, 0], clusters[:, 1]] - 1]

    return clusters, prob


@jax.jit
def _compute_different_clusters_mask(clusters, σ):
    # mask the clusters to include only moves
    # where the dof changes
    if jnp.issubdtype(σ, jnp.bool) or jnp.issubdtype(σ, jnp.integer):
        hoppable_clusters_mask = σ[..., clusters[:, 0]] != σ[..., clusters[:, 1]]
    else:
        hoppable_clusters_mask = ~jnp.isclose(
            σ[..., clusters[:, 0]], σ[..., clusters[:, 1]]
        )
    return hoppable_clusters_mask
