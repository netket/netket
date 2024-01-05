from netket.sampler import MetropolisSampler
from netket.experimental.hilbert import SpinOrbitalFermions


def MetropolisFermionExchange(
    hilbert, *, clusters=None, graph=None, d_max=1, **kwargs
) -> MetropolisSampler:
    r"""
    This sampler moves (or hops) a random fermion to a different but random empty mode.
    It works similar to MetropolisExchange, but only allows exchanges between occupied and unoccupied modes.

    Args:
        hilbert: The Hilbert space to sample.
        d_max: The maximum graph distance allowed for exchanges.
        n_chains: The total number of independent Markov chains across all MPI ranks. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    from .rules import FermionExchangeRule

    if not isinstance(hilbert, SpinOrbitalFermions):
        raise ValueError(
            "This sampler only works with SpinOrbitalFermions hilbert spaces."
        )

    rule = FermionExchangeRule(clusters=clusters, graph=graph, d_max=d_max)
    return MetropolisSampler(hilbert, rule, **kwargs)
