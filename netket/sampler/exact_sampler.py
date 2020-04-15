import numpy as _np
from .abstract_sampler import AbstractSampler

from .._C_netket import sampler as c_sampler
from .._C_netket.utils import random_engine, rand_uniform_real


class PyExactSampler(AbstractSampler):
    """
    Pure Python version of ExactSampler. See ExactSampler for more details.
    """

    def __init__(self, machine, sample_size=16):
        """
         Constructs a new ``PyExactSampler`` given a machine.

         Args:
             machine: A machine $$\Psi(s)$$ used for the sampling.
                      The probability distribution being sampled
                      from is $$F(\Psi(s))$$, where the function
                      $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.

             sample_size: The number of independent samples to be generated at each invocation of __next__.
        """
        self.hilbert = machine.hilbert
        self.machine_pow = 2.0
        super().__init__(machine, sample_size)

    def reset(self, init_random=False):
        self._prob = _np.exp(self.machine_pow * self.machine.to_array().real())
        self._prob /= self._prob.sum()

    def __next__(self):
        numbers = _np.random.choice(
            self._prob.size, size=self.sample_shape[0], replace=True, p=self._prob
        )
        return self.hilbert.number_to_state(numbers)

    @property
    def machine_pow(self):
        return self._machine_pow

    @machine_pow.setter
    def machine_pow(self, m_power):
        self._machine_pow = m_power
        self.reset()


class ExactSampler(AbstractSampler):
    """
    This sampler generates i.i.d. samples from $$|\Psi(s)|^2$$.
    In order to perform exact sampling, $$|\Psi(s)|^2$$ is precomputed an all
    the possible values of the quantum numbers $$s$$. This sampler has thus an
    exponential cost with the number of degrees of freedom, and cannot be used
    for large systems, where Metropolis-based sampling are instead a viable
    option.
    """

    def __init__(self, machine, sample_size=16):
        """
         Constructs a new ``ExactSampler`` given a machine.

         Args:
             machine: A machine $$\Psi(s)$$ used for the sampling.
                      The probability distribution being sampled
                      from is $$F(\Psi(s))$$, where the function
                      $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.

             sample_size: The number of independent samples to be generated at each invocation of __next__.

         Examples:
             Exact sampling from a RBM machine in a 1D lattice of spin 1/2

             ```python
             >>> import netket as nk
             >>>
             >>> g=nk.graph.Hypercube(length=8,n_dim=1,pbc=True)
             >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
             >>>
             >>> # RBM Spin Machine
             >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
             >>>
             >>> sa = nk.sampler.ExactSampler(machine=ma)

             ```
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.ExactSampler(
                machine=machine, sample_size=sample_size
            )
        else:
            self.sampler = PyExactSampler(
                machine=machine, sample_size=sample_size)
        super().__init__(machine, sample_size)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_pow(self):
        return self.sampler.machine_pow

    @machine_pow.setter
    def machine_pow(self, m_power):
        self.sampler.machine_pow = m_power
