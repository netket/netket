import numpy as _np
from netket import random as _random

from .metropolis_hastings import MetropolisHastings
from .metropolis_hastings_pt import MetropolisHastingsPt

from numba import jit


class _custom_local_kernel:
    def __init__(self, move_operators, move_weights=None):

        self._rand_op_n = _np.empty(1, dtype=_np.intp)
        self._sections = _np.empty(1, dtype=_np.intp)
        self._x_prime = _np.empty(1)
        self._mels = _np.empty(1)
        self._get_conn = move_operators.get_conn_filtered
        self._n_operators = move_operators.n_operators

        if move_weights is None:
            self._move_weights = _np.ones(self._n_operators, dtype=_np.float64)
        else:
            self._move_weights = _np.asarray(move_weights, dtype=_np.float64)

        self._check_operators(move_operators.operators)

        # Check move weights
        if self._move_weights.shape != (self._n_operators,):
            raise ValueError("move_weights have the wrong shape")
        if self._move_weights.min() < 0:
            raise ValueError("move_weights must be positive")

        # normalize the probabilities and compute the cumulative
        self._move_weights /= self._move_weights.sum()
        self._move_cumulative = _np.cumsum(self._move_weights)

    def _check_operators(self, operators):
        for op in operators:
            assert op.imag.max() < 1.0e-10
            assert op.min() >= 0
            assert _np.allclose(op.sum(axis=0), 1.0)
            assert _np.allclose(op.sum(axis=1), 1.0)
            assert _np.allclose(op, op.T)

    def apply(self, state, state_1, log_prob_corr):

        self._rand_op_n, self._sections = self._pick_random_and_init(
            state.shape[0], self._move_cumulative, self._rand_op_n, self._sections
        )

        self._x_prime, self._mels = self._get_conn(
            state, self._sections, self._rand_op_n
        )

        self._choose_and_return(
            state_1, self._x_prime, self._mels, self._sections, log_prob_corr
        )

    @staticmethod
    @jit(nopython=True)
    def _pick_random_and_init(batch_size, move_cumulative, out, sections):

        if out.size != batch_size:
            out = _np.empty(batch_size, dtype=out.dtype)
            sections = _np.empty(batch_size, dtype=out.dtype)

        for i in range(batch_size):
            p = _random.uniform()
            out[i] = _np.searchsorted(move_cumulative, p)
        return out, sections

    @staticmethod
    @jit(nopython=True)
    def _choose_and_return(state_1, x_prime, mels, sections, log_prob_corr):
        low = 0
        for i in range(state_1.shape[0]):
            p = _random.uniform()
            exit_state = 0
            cumulative_prob = mels[low].real
            while p > cumulative_prob:
                exit_state += 1
                cumulative_prob += mels[low + exit_state].real
            state_1[i] = x_prime[low + exit_state]
            low = sections[i]

        log_prob_corr.fill(0.0)


class CustomSampler(MetropolisHastings):
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
    """

    def __init__(
        self,
        machine,
        move_operators,
        move_weights=None,
        n_chains=16,
        sweep_size=None,
        batch_size=None,
    ):
        r"""
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
           >>> move_op = nk.operator.LocalOperator(hilbert=hi,operators=[X] * g.n_sites,acting_on=[[i] for i in range(g.n_sites)])
           >>> sa = nk.sampler.CustomSampler(machine=ma, move_operators=move_op)
        """
        super().__init__(
            machine,
            _custom_local_kernel(move_operators, move_weights),
            n_chains,
            sweep_size,
            batch_size,
        )


class CustomSamplerPt(MetropolisHastingsPt):
    """
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `CustomSampler`.
    The number of replicas can be chosen by the user.
    """

    def __init__(
        self,
        machine,
        move_operators,
        move_weights=None,
        n_replicas=16,
        sweep_size=None,
        batch_size=None,
    ):
        r"""
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
        super().__init__(
            machine,
            _custom_local_kernel(move_operators, move_weights),
            n_replicas,
            sweep_size,
            batch_size,
        )
