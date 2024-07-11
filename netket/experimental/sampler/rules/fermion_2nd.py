from typing import Optional
import numpy as np

from netket.sampler.rules import ExchangeRule
from netket.graph import AbstractGraph
from netket.graph import disjoint_union
from netket.experimental.hilbert import SpinOrbitalFermions


class ParticleExchangeRule(ExchangeRule):
    """Exchange rule for particles on a lattice.

    Works similarly to :class:`netket.sampler.rules.ExchangeRule`, but
    takes into account that only occupied orbitals
    can be exchanged with unoccupied ones.

    This sampler conserves the number of particles.
    """

    def __init__(
        self,
        hilbert,
        *,
        clusters: Optional[list[tuple[int, int]]] = None,
        graph: Optional[AbstractGraph] = None,
        d_max: int = 1,
        exchange_spins: bool = False,
    ):
        r"""
        Constructs the ParticleExchange Rule.

        Particles are only exchanged between modes where the particle number is different.
        For fermions, only occupied orbitals can be exchanged with unoccupied ones.

        You can pass either a list of clusters or a netket graph object to
        determine the clusters to exchange.

        Args:
            hilbert: The hilbert space to be sampled.
            clusters: The list of clusters that can be exchanged. This should be
                a list of 2-tuples containing two integers. Every tuple is an edge,
                or cluster of sites to be exchanged.
            graph: A graph, from which the edges determine the clusters
                that can be exchanged.
            d_max: Only valid if a graph is passed in. The maximum distance
                between two sites
            exchange_spins: (default False) If exchange_spins, the graph must encode the
                connectivity  between the first N physical sites having same spin, and
                it is replicated using :func:`netket.graph.disjoint_union` other every
                spin subsector. This option conserves the number of fermions per
                spin subsector. If the graph does not have a number of sites equal
                to the number of orbitals in the hilbert space, this flag has no effect.
        """
        if not isinstance(hilbert, SpinOrbitalFermions):
            raise ValueError(
                "This sampler rule currently only works with SpinOrbitalFermions hilbert spaces."
            )
        if not exchange_spins and hilbert.n_spin_subsectors > 1:
            if graph is not None and graph.n_nodes == hilbert.n_orbitals:
                graph = disjoint_union(*[graph] * hilbert.n_spin_subsectors)
            if clusters is not None and np.max(clusters) < hilbert.n_orbitals:
                clusters = np.concatenate(
                    [
                        clusters + i * hilbert.n_orbitals
                        for i in range(hilbert.n_spin_subsectors)
                    ]
                )
        super().__init__(clusters=clusters, graph=graph, d_max=d_max)

    def __repr__(self):
        return f"ParticleExchangeRule(# of clusters: {len(self.clusters)})"
