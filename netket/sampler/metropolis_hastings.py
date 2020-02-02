import numpy as _np
from .abstract_sampler import AbstractSampler

from .._C_netket import sampler as c_sampler
from .._C_netket.utils import random_engine, rand_uniform_real


class PyMetropolisHastings(AbstractSampler):
    """
    ``MetropolisHastings`` is a generic Metropolis-Hastings sampler using
    a local transition kernel to perform moves in the Markov Chain.
    The transition kernel is used to generate
    a proposed state :math:`s^\prime`, starting from the current state :math:`s`.
    The move is accepted with probability

    .. math::
    A(s\rightarrow s^\prime) = \mathrm{min}\left (1,\frac{P(s^\prime)}{P(s)} F(e^{L(s,s^\prime)})\right),

    where the probability being sampled is :math:`F(\Psi(s))` (by default :math:`F(x)=|x|^2`)
    and :math:`L(s,s^\prime)` is a correcting factor computed by the transition kernel.
    """

    def __init__(
        self, machine, transition_kernel, n_chains=16, sweep_size=None, batch_size=None
    ):
        """
        Constructs a new ``MetropolisHastings`` sampler given a machine and
        a transition kernel.

        Args:
            machine: A machine :math:`\Psi(s)` used for the sampling.
                          The probability distribution being sampled
                          from is :math:`F(\Psi(s))`, where the function
                          $$F(X)$$, is arbitrary, by default :math:`F(X)=|X|^2`.
            transition_kernel: A function to generate a transition.
                          This should take as an input the current state (in batches)
                          and return a modified state (also in batches).
                          This function must also return an array containing the
                          `log_prob_corrections` :math:`L(s,s^\prime)`.
            n_chains: The number of Markov Chain to be run in parallel on a single process.
            sweep_size: The number of exchanges that compose a single sweep.
                        If None, sweep_size is equal to the number of degrees of freedom (n_visible).
            batch_size: The batch size to be used when calling log_val on the given Machine.
                        If None, batch_size is equal to the number Markov chains (n_chains).

        """

        self.machine = machine
        self.n_chains = n_chains

        self.sweep_size = sweep_size

        self._kernel = transition_kernel

        self.machine_func = lambda x, out=None: _np.square(
            _np.absolute(x), out)

        super().__init__(machine, n_chains)

    @property
    def n_chains(self):
        return self._n_chains

    @n_chains.setter
    def n_chains(self, n_chains):
        if n_chains < 0:
            raise ValueError("Expected a positive integer for n_chains ")

        self._n_chains = n_chains

        self._state = _np.zeros((n_chains, self._n_visible))
        self._state1 = _np.copy(self._state)

        self._log_values = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_values_1 = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_prob_corr = _np.zeros(n_chains)

    @property
    def machine_func(self):
        return self._machine_func

    @machine_func.setter
    def machine_func(self, machine_fun):
        self._machine_func = machine_fun

    @property
    def sweep_size(self):
        return self._sweep_size

    @sweep_size.setter
    def sweep_size(self, sweep_size):
        self._sweep_size = sweep_size if sweep_size != None else self._n_visible
        if self._sweep_size < 0:
            raise ValueError("Expected a positive integer for sweep_size ")
        self._rand_for_acceptance = _np.zeros(
            self._sweep_size * self.n_chains, dtype=float
        )

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, machine):
        self._machine = machine
        self._n_visible = machine.hilbert.size
        self._hilbert = machine.hilbert

    def reset(self, init_random=False):
        if init_random:
            for state in self._state:
                self._hilbert.random_vals(state, random_engine())
        self.machine.log_val(self._state, out=self._log_values)

    def _log_val_batched(self, v, out=None):
        return self.machine.log_val(v, out)

    def __next__(self):

        rand_uniform_real(self._rand_for_acceptance)

        for sweep in range(self.sweep_size):

            # Propose a new state using the transition kernel
            self._kernel(self._state, self._state1, self._log_prob_corr)

            self._log_val_batched(self._state1, out=self._log_values_1)

            # Acceptance probability
            self._prob = self.machine_func(
                _np.exp(self._log_values_1 -
                        self._log_values + self._log_prob_corr)
            )

            # Acceptance test
            accept = self._prob > self._rand_for_acceptance[sweep]

            # Update of the state
            self._log_values = _np.where(
                accept, self._log_values_1, self._log_values)
            self._state = _np.where(
                accept.reshape(-1, 1), self._state1, self._state)
        return self._state


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
                c_sampler.LocalKernel(machine.hilbert),
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
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


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
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class MetropolisExchange(AbstractSampler):
    """
    This sampler acts locally only on two local degree of freedom :math:`s_i` and :math:`s_j`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N`,
    where in general :math:`s^\prime_i \neq s_i` and :math:`s^\prime_j \neq s_j`.
    The sites :math:`i` and :math:`j` are also chosen to be within a maximum graph
    distance of :math:`d_{\mathrm{max}}`.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A pair of indices :math:`i,j = 1\dots N`, and such
       that :math:`\mathrm{dist}(i,j) \leq d_{\mathrm{max}}`,
       is chosen with uniform probability.
    2. The sites are exchanged, i.e. :math:`s^\prime_i = s_j` and :math:`s^\prime_j = s_i`.

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum numbers
    are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where :math:`\sum_i s_i = \mathrm{constant}` is needed,
    otherwise the sampling would be strongly not ergodic.
    """

    def __init__(self, machine, d_max=1, n_chains=16, sweep_size=None, batch_size=None):
        """
        Args:
              machine: A machine :math:`\Psi(s)` used for the sampling.
                       The probability distribution being sampled
                       from is :math:`F(\Psi(s))`, where the function
                       :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.

              d_max: The maximum graph distance allowed for exchanges.
              n_chains: The number of Markov Chain to be run in parallel on a single process.
              sweep_size: The number of exchanges that compose a single sweep.
                          If None, sweep_size is equal to the number of degrees of freedom (n_visible).
              batch_size: The batch size to be used when calling log_val on the given Machine.
                          If None, batch_size is equal to the number Markov chains (n_chains).


        Examples:
              Sampling from a RBM machine in a 1D lattice of spin 1/2, using
              nearest-neighbours exchanges.

              >>> import netket as nk
              >>>
              >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
              >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
              >>>
              >>> # RBM Spin Machine
              >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
              >>>
              >>> # Construct a MetropolisExchange Sampler
              >>> sa = nk.sampler.MetropolisExchange(machine=ma)
              >>> print(sa.machine.hilbert.size)
              100
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisExchange(
                machine=machine,
                n_chains=n_chains,
                d_max=d_max,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                c_sampler.ExchangeKernel(machine.hilbert, d_max),
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
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class MetropolisExchangePt(AbstractSampler):
    """
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `MetropolisExchange`.
    The number of replicas can be chosen by the user.
    """

    def __init__(
        self, machine, d_max=1, n_replicas=16, sweep_size=None, batch_size=None
    ):
        """
        Args:
            machine: A machine :math:`\Psi(s)` used for the sampling.
                     The probability distribution being sampled
                     from is :math:`F(\Psi(s))`, where the function
                     :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
            d_max: The maximum graph distance allowed for exchanges.
            n_replicas: The number of replicas used for parallel tempering.
            sweep_size: The number of exchanges that compose a single sweep.
                        If None, sweep_size is equal to the number of degrees of freedom (n_visible).

        Examples:
            Sampling from a RBM machine in a 1D lattice of spin 1/2, using
            nearest-neighbours exchanges.

            >>> import netket as nk
            >>>
            >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
            >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
            >>>
            >>> # RBM Spin Machine
            >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
            >>>
            >>> # Construct a MetropolisExchange Sampler with parallel tempering
            >>> sa = nk.sampler.MetropolisExchangePt(machine=ma,n_replicas=24)
            >>> print(sa.machine.hilbert.size)
            100
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisExchangePt(
                machine=machine,
                n_replicas=n_replicas,
                d_max=d_max,
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
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


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
                c_sampler.HamiltonianKernel(hamiltonian),
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
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


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
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class CustomSampler(AbstractSampler):
    """
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
        """
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
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.CustomSampler(
                machine=machine,
                move_operators=move_operators,
                move_weights=move_weights,
                n_chains=n_chains,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                c_sampler.CustomLocalKernel(move_operators, move_weights),
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
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class CustomSamplerPt(AbstractSampler):
    """
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `CustomSampler`.
    The number of replicas can be chosen by the user.
    """

    def __init__(
        self, machine, move_operators, move_weights=None, n_replicas=16, sweep_size=None
    ):
        """
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
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.CustomSamplerPt(
                machine=machine,
                move_operators=move_operators,
                move_weights=move_weights,
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
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func
