import abc


class AbstractHilbert(abc.ABC):
    """Abstract class for NetKet hilbert objects"""

    @property
    @abc.abstractmethod
    def size(self):
        r"""int: The total number number of spins."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def is_discrete(self):
        r"""bool: Whether the hilbert space is discrete."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def local_size(self):
        r"""int: Size of the local degrees of freedom that make the total hilbert space."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def local_states(self):
        r"""list[float]: A list of discreet local quantum numbers."""
        return NotImplementedError

    @abc.abstractmethod
    def random_vals(self, rgen=None):
        r"""Member function generating uniformely distributed local random states.

        Args:
            state: A reference to a visible configuration, in output this
                  contains the random state.
            rgen: The random number generator. If None, the global
                 NetKet random number generator is used.

        Examples:
           Test that a new random state is a possible state for the hilbert
           space.

           >>> import netket as nk
           >>> import numpy as np
           >>> hi = nk.hilbert.Boson(n_max=3, graph=nk.graph.Hypercube(length=5, n_dim=1))
           >>> rstate = np.zeros(hi.size)
           >>> rg = nk.utils.RandomEngine(seed=1234)
           >>> hi.random_vals(rstate, rg)
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True
           """
        return NotImplementedError
