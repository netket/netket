import numpy as _np
from netket import random as _random

from .metropolis_hastings import MetropolisHastings
from .metropolis_hastings_pt import MetropolisHastingsPt

from numba import jit
import math


class _hamiltonian_kernel:
    def __init__(self, hamiltonian):
        self._hamiltonian = hamiltonian
        self._sections = _np.empty(1, dtype=_np.int32)
        self._hamconn = self._hamiltonian.get_conn_flattened
        self._n_conn = self._hamiltonian.n_conn
        self._hilbert = hamiltonian.hilbert

    def transition(self, state, state_1, log_prob_corr):

        sections = self._sections
        sections = _np.empty(state.shape[0], dtype=_np.int32)
        vprimes = self._hamconn(state, sections)[0]

        self._choose(vprimes, sections, state_1, log_prob_corr)

        self._n_conn(state_1, sections)

        log_prob_corr -= _np.log(sections)

    def random_state(self, state):

        for i in range(state.shape[0]):
            self._hilbert.random_vals(out=state[i])

    @staticmethod
    @jit(nopython=True)
    def _choose(states, sections, out, w):
        low_range = 0
        for i, s in enumerate(sections):
            n_rand = _random.randint(low_range, s)
            out[i] = states[n_rand]
            w[i] = math.log(s - low_range)
            low_range = s


class MetropolisHamiltonian(MetropolisHastings):
    r"""
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
        r"""
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
        super().__init__(
            machine, _hamiltonian_kernel(hamiltonian), n_chains, sweep_size, batch_size
        )


class MetropolisHamiltonianPt(MetropolisHastingsPt):
    r"""
     This sampler performs parallel-tempering
     moves in addition to the local moves implemented in `MetropolisLocal`.
     The number of replicas can be :math:`N_{\mathrm{rep}}` chosen by the user.
    """

    def __init__(
        self, machine, hamiltonian, n_replicas=16, sweep_size=None, batch_size=None
    ):
        r"""
        Args:
            machine: A machine :math:`\Psi(s)` used for the sampling.
                      The probability distribution being sampled
                      from is :math:`F(\Psi(s))`, where the function
                      :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
            hamiltonian: The operator used to perform off-diagonal transition.
            n_replicas: The number of replicas used for parallel tempering.
            sweep_size: The number of exchanges that compose a single sweep.
                         If None, sweep_size is equal to the number of degrees of freedom (n_visible).
            batch_size: The batch size to be used when calling log_val on the given Machine.
                        If None, batch_size is equal to the number of replicas (n_replicas).
        """
        super().__init__(
            machine,
            _hamiltonian_kernel(hamiltonian),
            n_replicas,
            sweep_size,
            batch_size,
        )
