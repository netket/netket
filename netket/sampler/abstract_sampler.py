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
    def machine_func(self):
        return lambda x, out=None: _np.square(_np.absolute(x), out)

    @machine_func.setter
    def machine_func(self, func):
        raise NotImplementedError

    def samples(self, n_max, init_random=False):

        self.reset(init_random)

        n = 0
        while n < n_max:
            yield self.__next__()
            n += 1
