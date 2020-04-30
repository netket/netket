import numpy as _np
from netket import random as _random

from .abstract_sampler import AbstractSampler
from .metropolis_hastings import *
from .._C_netket import sampler as c_sampler

from numba import jit, int64, float64
from .._jitclass import jitclass


@jitclass([("local_states", float64[:]), ("size", int64), ("n_states", int64)])
class _local_kernel:
    def __init__(self, local_states, size):
        self.local_states = _np.sort(
            _np.asarray(local_states, dtype=_np.float64))
        self.size = size
        self.n_states = self.local_states.size

    def apply(self, state, state_1, log_prob_corr):

        for i in range(state.shape[0]):
            state_1[i] = state[i]

            si = _random.randint(0, self.size)

            rs = _random.randint(0, self.n_states - 1)

            state_1[i, si] = self.local_states[
                rs + (self.local_states[rs] >= state[i, si])
            ]

        log_prob_corr[:] = 0.0


class MetropolisLocal(AbstractSampler):
    """
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
    """

    def __init__(
        self, machine, n_chains=16, sweep_size=None, batch_size=None, backend=None
    ):
        """

         Constructs a new :class:`MetropolisLocal` sampler given a machine.

         Args:
            machine: A machine :math:`\Psi(s)` used for the sampling.
                     The probability distribution being sampled
                     from is :math:`F(\Psi(s))`, where the function
                     :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.

            n_chains:   The number of Markov Chain to be run in parallel on a single process.
            sweep_size: The number of exchanges that compose a single sweep.
                        If None, sweep_size is equal to the number of degrees of freedom (n_visible).
            batch_size: The batch size to be used when calling log_val on the given Machine.
                        If None, batch_size is equal to the number Markov chains (n_chains).


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
        if "_C_netket.machine" in str(type(machine)) and backend != "py":
            self.sampler = c_sampler.MetropolisLocal(
                machine=machine,
                n_chains=n_chains,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                _local_kernel(
                    _np.asarray(
                        machine.hilbert.local_states), machine.hilbert.size
                ),
                n_chains,
                sweep_size,
                batch_size,
            )
        super().__init__(machine, n_chains)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_pow(self):
        return self.sampler.machine_pow

    @machine_pow.setter
    def machine_pow(self, m_pow):
        self.sampler.machine_pow = m_pow

    @property
    def acceptance(self):
        """The measured acceptance probability."""
        return self.sampler.acceptance


class MetropolisLocalPt(AbstractSampler):
    """
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `MetropolisLocal`.
    The number of replicas can be chosen by the user.
    """

    def __init__(self, machine, n_replicas=16, sweep_size=None, batch_size=None):
        """
        Args:
             machine: A machine :math:`\Psi(s)` used for the sampling.
                      The probability distribution being sampled
                      from is :math:`F(\Psi(s))`, where the function
                      :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
             n_replicas: The number of replicas used for parallel tempering.
             sweep_size: The number of exchanges that compose a single sweep.
                         If None, sweep_size is equal to the number of degrees of freedom (n_visible).

        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisLocalPt(
                machine=machine, n_replicas=n_replicas, sweep_size=sweep_size
            )
        else:
            raise ValueError(
                """Parallel Tempering samplers are not yet implemented
                for pure python machines"""
            )
        super().__init__(machine, 1)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_pow(self):
        return self.sampler.machine_pow

    @machine_pow.setter
    def machine_pow(self, m_pow):
        self.sampler.machine_pow = m_pow

    @property
    def acceptance(self):
        """The measured acceptance probability."""
        return self.sampler.acceptance
