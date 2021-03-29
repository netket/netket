from .metropolis_hastings import *
from ._kernels import _CustomKernel


def CustomSampler(
    machine, move_operators, move_weights=None, n_chains=16, sweep_size=None, **kwargs
):
    r"""
     Custom Sampler, where transition operators are specified by the user.
     For the moment, this functionality is limited to transition operators which
     are sums of :math:`k`-local operators:

     .. math::
        \mathcal{M}= \sum_i M_i


     where the move operators :math:`M_i` act on an (arbitrary) subset of sites.

     The operators :math:`M_i` are specified giving their matrix elements, and a list
     of sites on which they act. Each operator :math:`M_i` must be real,
     symmetric, positive definite and stochastic (i.e. sum of each column and line is 1).

     The transition probability associated to a custom sampler can be decomposed into two steps:

     1. One of the move operators :math:`M_i` is chosen with a weight given by the
       user (or uniform probability by default). If the weights are provided,
       they do not need to sum to unity.

     2. Starting from state
       :math:`|n \rangle`, the probability to transition to state
       :math:`|m\rangle` is given by
       :math:`\langle n|  M_i | m \rangle`.

     Args:
        machine: A machine :math:`\Psi(s)` used for the sampling.
               The probability distribution being sampled
               from is :math:`F(\Psi(s))`, where the function
               :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
        move_operators: The stochastic `LocalOperator`
             :math:`\mathcal{M}= \sum_i M_i` used for transitions.
        move_weights: For each :math:`i`, the probability to pick one of
             the move operators (must sum to one).
        n_chains: The number of Markov Chains to be run in parallel on a single process.
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
        >>> # Construct a Custom Sampler
        >>> # Using random local spin flips (Pauli X operator)
        >>> X = [[0, 1],[1, 0]]
        >>> move_op = nk.operator.LocalOperator(hilbert=hi,operators=[X] * g.n_nodes, acting_on=[[i] for i in range(g.n_nodes)])
        >>> sa = nk.sampler.CustomSampler(machine=ma, move_operators=move_op)
    """
    return MetropolisHastings(
        machine,
        _CustomKernel(machine, move_operators, move_weights),
        n_chains,
        sweep_size,
        **kwargs,
    )


def CustomSamplerPt(
    machine, move_operators, move_weights=None, n_replicas=16, sweep_size=None, **kwargs
):
    r"""
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `CustomSampler`.
    The number of replicas can be chosen by the user.

    Args:
      machine: A machine :math:`\Psi(s)` used for the sampling.
               The probability distribution being sampled
               from is :math:`F(\Psi(s))`, where the function
               :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
      move_operators: The stochastic `LocalOperator`
               :math:`\mathcal{M}= \sum_i M_i` used for transitions.
      move_weights: For each :math:`i`, the probability to pick one of
               the move operators (must sum to one).
      n_replicas: The number of replicas used for parallel tempering.
      sweep_size: The number of exchanges that compose a single sweep.
                  If None, sweep_size is equal to the number of degrees of freedom (n_visible).
    """
    return MetropolisHastingsPt(
        machine,
        _CustomKernel(machine, move_operators, move_weights),
        n_replicas,
        sweep_size,
        **kwargs,
    )
