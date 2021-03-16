from .metropolis_hastings import *
from ._kernels import _LocalKernel


def MetropolisLocal(machine, n_chains=16, sweep_size=None, **kwargs):
    r"""
    Sampler acting on one local degree of freedom.

    This sampler acts locally only on one local degree of freedom :math:`s_i`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s_N`,
    where :math:`s^\prime_i \\neq s_i`.

    The transition probability associated to this
    sampler can be decomposed into two steps:

    1. One of the site indices :math:`i = 1\dots N` is chosen
    with uniform probability.
    2. Among all the possible (:math:`m`) values that :math:`s_i` can take,
    one of them is chosen with uniform probability.

    For example, in the case of spin :math:`1/2` particles, :math:`m=2`
    and the possible local values are :math:`s_i = -1,+1`.
    In this case then :class:`MetropolisLocal` is equivalent to flipping a random spin.

    In the case of bosons, with occupation numbers
    :math:`s_i = 0, 1, \dots n_{\mathrm{max}}`, :class:`MetropolisLocal`
    would pick a random local occupation number uniformly between :math:`0`
    and :math:`n_{\mathrm{max}}`.

    Args:
        machine: A machine :math:`\Psi(s)` used for the sampling.
                 The probability distribution being sampled
                 from is :math:`F(\Psi(s))`, where the function
                 :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.

        n_chains:   The number of Markov Chain to be run in parallel on a single process.
        sweep_size: The number of exchanges that compose a single sweep.
                    If None, sweep_size is equal to the number of degrees of freedom (n_visible).


    Examples:
        Sampling from a RBM machine in a 1D lattice of spin 1/2

        >>> import netket as nk
        >>>
        >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
        >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
        >>>
        >>> # RBM Spin Machine
        >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
        >>>
        >>> # Construct a MetropolisLocal Sampler
        >>> sa = nk.sampler.MetropolisLocal(machine=ma)
        >>> print(sa.machine.hilbert.size)
        100
    """

    return MetropolisHastings(
        machine, _LocalKernel(machine), n_chains, sweep_size, **kwargs
    )


def MetropolisLocalPt(machine, n_replicas=16, sweep_size=None, **kwargs):
    r"""
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `MetropolisLocal`.
    The number of replicas can be chosen by the user.

    Args:
         machine: A machine :math:`\Psi(s)` used for the sampling.
                  The probability distribution being sampled
                  from is :math:`F(\Psi(s))`, where the function
                  :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
         n_replicas: The number of replicas used for parallel tempering.
         sweep_size: The number of exchanges that compose a single sweep.
                     If None, sweep_size is equal to the number of degrees of freedom (n_visible).

    """
    return MetropolisHastingsPt(
        machine, _LocalKernel(machine), n_replicas, sweep_size, **kwargs
    )
