import abc
import numpy as _np


class AbstractSampler(abc.ABC):
    """Abstract class for NetKet samplers"""

    def __init__(self, machine, sample_size=1):
        self.sample_size = sample_size
        self.machine = machine

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        pass

    @abc.abstractmethod
    def reset(self, init_random=False):
        pass

    @property
    def machine_pow(self):
        return 2.0

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, machine):
        self._machine = machine
        self._input_size = machine.input_size

        self.sample_shape = (self.sample_size, self._input_size)

    @machine_pow.setter
    def machine_pow(self, m_power):
        raise NotImplementedError

    def samples(self, n_max, init_random=False):

        self.reset(init_random)

        n = 0
        while n < n_max:
            yield self.__next__()
            n += 1

    def generate_samples(self, n_samples, init_random=False, samples=None):
        self.reset(init_random)

        if samples is None:
            samples = _np.empty((n_samples, self.sample_shape[0], self.sample_shape[1]))

        for i in range(n_samples):
            samples[i] = self.__next__()
        return samples
