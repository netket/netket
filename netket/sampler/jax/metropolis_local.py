import jax
from functools import partial
from .metropolis_hastings import MetropolisHastings


class _local_kernel:
    def __init__(self, local_states, size):
        self.local_states = jax.numpy.sort(jax.numpy.array(local_states))
        self.size = size
        self.n_states = self.local_states.size

    @partial(jax.jit, static_argnums=(0))
    def transition(self, key, state):

        keys = jax.random.split(key, 2)
        si = jax.random.randint(keys[0], shape=(1,), minval=0, maxval=self.size)
        rs = jax.random.randint(keys[1], shape=(1,), minval=0, maxval=self.n_states - 1)

        return jax.ops.index_update(
            state, si, self.local_states[rs + (self.local_states[rs] >= state[si])]
        )

    @partial(jax.jit, static_argnums=(0))
    def random_state(self, key, state):
        keys = jax.random.split(key, 2)

        rs = jax.random.randint(
            keys[1], shape=(self.size,), minval=0, maxval=self.n_states
        )

        return keys[0], self.local_states[rs]


class MetropolisLocal(MetropolisHastings):
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
    """

    def __init__(self, machine, n_chains=16, sweep_size=None, batch_size=None):
        r"""

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
             >>> ma = nk.machine.JaxRbm(alpha=1, hilbert=hi)
             >>>
             >>> # Construct a MetropolisLocal Sampler
             >>> sa = nk.sampler.MetropolisLocal(machine=ma)
             >>> print(sa.machine.hilbert.size)
             100
        """
        kernel = _local_kernel(machine.hilbert.local_states, machine.input_size)

        super().__init__(
            machine, kernel, n_chains, sweep_size,
        )
