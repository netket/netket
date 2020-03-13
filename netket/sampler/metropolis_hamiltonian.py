import numpy as _np
from netket import random as _random

from .abstract_sampler import AbstractSampler
from .metropolis_hastings import *
from .._C_netket import sampler as c_sampler

from numba import jit


class _hamiltonian_kernel:
    def __init__(self, hamiltonian):
        self._hamiltonian = hamiltonian
        self._get_conn = hamiltonian.get_conn

    def apply(self, state, state_1, log_prob_corr):
        hamconn = self._get_conn
        vprimes = hamconn(state)[0]
        self._choose(tuple(vprimes), state_1, log_prob_corr)
        vprimes = hamconn(state_1)[0]
        # TODO avoid casting to tuple here using numba's List
        self._corr(tuple(vprimes), log_prob_corr)

    @staticmethod
    @jit(nopython=True)
    def _choose(states, out, w):
        for i, state in enumerate(states):
            n = state.shape[0]
            n_rand = _random.randint(0, n)
            out[i] = state[n_rand]
            w[i] = _np.log(n)

    @staticmethod
    @jit(nopython=True)
    def _corr(states, w):
        for i, state in enumerate(states):
            n = state.shape[0]
            w[i] -= _np.log(n)


class MetropolisHamiltonian(AbstractSampler):
    """
    Sampling based on the off-diagonal elements of a Hamiltonian (or a generic Operator).
    In this case, the transition matrix is taken to be:

    .. math::
       T( \mathbf{s} \rightarrow \mathbf{s}^\prime) = \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),

    where :math:`\theta(x)` is the Heaviside step function, and :math:`\mathcal{N}(\mathbf{s})`
    is a state-dependent normalization.
    The effect of this transition probability is then to connect (with uniform probability)
    a given state :math:`\mathbf{s}` to all those states :math:`\mathbf{s}^\prime` for which the Hamiltonian has
    finite matrix elements.
    Notice that this sampler preserves by construction all the symmetries
    of the Hamiltonian. This is in generally not true for the local samplers instead.
    """

    def __init__(
        self, machine, hamiltonian, n_chains=16, sweep_size=None, batch_size=None
    ):
        """
        Args:
           machine: A machine :math:`\Psi(s)` used for the sampling.
                    The probability distribution being sampled
                    from is :math:`F(\Psi(s))`, where the function
                    :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
           hamiltonian: The operator used to perform off-diagonal transition.
           n_chains: The number of Markov Chain to be run in parallel on a single process.
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
           >>> # Transverse-field Ising Hamiltonian
           >>> ha = nk.operator.Ising(hilbert=hi, h=1.0)
           >>>
           >>> # Construct a MetropolisHamiltonian Sampler
           >>> sa = nk.sampler.MetropolisHamiltonian(machine=ma,hamiltonian=ha)
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisHamiltonian(
                machine=machine,
                hamiltonian=hamiltonian,
                n_chains=n_chains,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                _hamiltonian_kernel(hamiltonian),
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


class MetropolisHamiltonianPt(AbstractSampler):
    """
     This sampler performs parallel-tempering
     moves in addition to the local moves implemented in `MetropolisLocal`.
     The number of replicas can be :math:`N_{\mathrm{rep}}` chosen by the user.
    """

    def __init__(self, machine, hamiltonian, n_replicas=16, sweep_size=None):
        """
        Args:
            machine: A machine :math:`\Psi(s)` used for the sampling.
                      The probability distribution being sampled
                      from is :math:`F(\Psi(s))`, where the function
                      :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
            hamiltonian: The operator used to perform off-diagonal transition.
            n_replicas: The number of replicas used for parallel tempering.
            sweep_size: The number of exchanges that compose a single sweep.
                         If None, sweep_size is equal to the number of degrees of freedom (n_visible).
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisHamiltonianPt(
                machine=machine,
                hamiltonian=hamiltonian,
                n_replicas=n_replicas,
                sweep_size=sweep_size,
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
