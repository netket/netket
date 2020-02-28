import abc
import numpy as _np


class AbstractSampler(abc.ABC):
    """Abstract class for NetKet samplers"""

    def __init__(self, machine, sample_size=None):
        super().__init__()
        self.machine = machine

        self.sample_size = sample_size if sample_size != None else 1

        self.sample_shape = (sample_size, machine.hilbert.size)

        self.reset(True)

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
            samples = _np.zeros(
                (n_samples, self.sample_shape[0], self.sample_shape[1]))

        for k, sample in enumerate(self.samples(n_samples)):
            samples[k] = sample
        return samples
