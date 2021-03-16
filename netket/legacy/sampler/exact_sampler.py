import numpy as _np
from .abstract_sampler import AbstractSampler
from netket.legacy.machine.density_matrix import AbstractDensityMatrix
from netket.hilbert import DoubledHilbert
import netket.legacy.random


class ExactSampler(AbstractSampler):
    r"""
    This sampler generates i.i.d. samples from $$|\Psi(s)|^2$$.
    In order to perform exact sampling, $$|\Psi(s)|^2$$ is precomputed an all
    the possible values of the quantum numbers $$s$$. This sampler has thus an
    exponential cost with the number of degrees of freedom, and cannot be used
    for large systems, where Metropolis-based sampling are instead a viable
    option.
    """

    def __init__(self, machine, sample_size=16):
        r"""
        Constructs a new ``ExactSampler`` given a machine.

        Args:
            machine: A machine $$\Psi(s)$$ used for the sampling.
                     The probability distribution being sampled
                     from is $$F(\Psi(s))$$, where the function
                     $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.

            sample_size: The number of independent samples to be generated at each invocation of __next__.
        """
        super().__init__(machine, sample_size)
        if isinstance(machine, AbstractDensityMatrix):
            self.hilbert = DoubledHilbert(machine.hilbert)
        else:
            self.hilbert = machine.hilbert
        self._machine_pow = 2.0
        self.reset()

    def reset(self, init_random=False):
        self._prob = _np.absolute(self.machine.to_array()) ** self.machine_pow
        self._prob /= self._prob.sum()

    def __next__(self):
        numbers = netket.legacy.random.choice(
            self._prob.size, size=self.sample_shape[0], replace=True, p=self._prob
        )
        return self.hilbert.numbers_to_states(numbers)

    def generate_samples(self, n_samples, init_random=False, samples=None):

        if samples is None:
            samples = _np.zeros((n_samples, self.sample_shape[0], self.sample_shape[1]))

        numbers = netket.legacy.random.choice(
            self._prob.size,
            size=self.sample_shape[0] * n_samples,
            replace=True,
            p=self._prob,
        )
        samples[:] = self.hilbert.numbers_to_states(numbers).reshape(samples.shape)

        return samples

    @property
    def machine_pow(self):
        return self._machine_pow

    @machine_pow.setter
    def machine_pow(self, m_power):
        self._machine_pow = m_power
        self.reset()
