from functools import singledispatch
from . import numpy


@singledispatch
def MetropolisHastings(machine, kernel, n_chains=16, sweep_size=None):
    r"""
    ``MetropolisHastings`` is a generic Metropolis-Hastings sampler using
    a transition kernel to perform moves in the Markov Chain.
    The transition kernel is used to generate
    a proposed state :math:`s^\prime`, starting from the current state :math:`s`.
    The move is accepted with probability

    .. math::
    A(s\rightarrow s^\prime) = \mathrm{min}\left (1,\frac{P(s^\prime)}{P(s)} F(e^{L(s,s^\prime)})\right),

    where the probability being sampled from is :math:`P(s)=|M(s)|^p. Here ::math::`M(s)` is a
    user-provided function (the machine), :math:`p` is also user-provided with default value :math:`p=2`,
    and :math:`L(s,s^\prime)` is a suitable correcting factor computed by the transition kernel.


    Args:
        machine: A machine :math:`M(s)` used for the sampling.
                The probability distribution being sampled
                from is :math:`P(s)=|M(s)|^p`, where the power :math:`p`
                is arbitrary, and by default :math:`p=2`.
        kernel: A kernel to generate random transitions from a given state as
                well as uniform random states.
                This must have two methods, `random_state` and `transition`.
                `random_state` takes an input state (in batches) and
                changes it in-place to a valid random state.
                `transition` takes an input state (in batches) and
                returns a batch of random states obtained transitioning from the initial state.
                `transition` must also return an array containing the
                `log_prob_corrections` :math:`L(s,s^\prime)`.
        n_chains: The number of Markov Chain to be run in parallel on a single process.
        sweep_size: The number of exchanges that compose a single sweep.
                If None, sweep_size is equal to the number of degrees of freedom being sampled
                (the size of the input vector s to the machine).

    """

    return numpy.MetropolisHastings(machine, kernel, n_chains, sweep_size)


@singledispatch
def MetropolisHastingsPt(machine, kernel, n_replicas=32, sweep_size=None):
    r"""
    ``MetropolisHastingsPt`` is a generic Metropolis-Hastings sampler using
    a local transition kernel to perform moves in the Markov Chain and replica-exchange moves
    to increase mixing times.
    The transition kernel is used to generate a proposed state :math:`s^\prime`,
    starting from the current state :math:`s`, as in ``MetropolisHastings``.
    Each replica is at an inverse temperature :math:`\beta` linearly spaced in the interval [0,1],
    where :math:`beta=1` corresponds to the target probability distribution,
    and :math:`beta=0` is the flat distribution.

    Args:
        machine: A machine :math:`M(s)` used for the sampling.
                The probability distribution being sampled
                from is :math:`P(s)=|M(s)|^p`, where the power :math:`p`
                is arbitrary, and by default :math:`p=2`.
        kernel: A kernel to generate random transitions from a given state as
                well as uniform random states.
                This must have two methods, `random_state` and `transition`.
                `random_state` takes an input state (in batches) and
                changes it in-place to a valid random state.
                `transition` takes an input state (in batches) and
                returns a batch of random states obtained transitioning from the initial state.
                `transition` must also return an array containing the
                `log_prob_corrections` :math:`L(s,s^\prime)`.
        n_replicas (int): The number of replicas used for replica-exchange moves. Each replica samples
        sweep_size (int): The number of exchanges that compose a single sweep.
                    If None, sweep_size is equal to the number of degrees of freedom (the input size of the machine).

    """
    return numpy.MetropolisHastingsPt(machine, kernel, n_replicas, sweep_size)
