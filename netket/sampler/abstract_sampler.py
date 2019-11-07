import abc


class AbstractSampler(abc.ABC):
    """Abstract class for NetKet samplers"""

    def __init__(self, machine, sample_shape=None):
        super().__init__()
        self.machine = machine

        self.sample_shape = (
            sample_shape if sample_shape != None else (1, machine.hilbert.size)
        )

        self.reset(True)

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        pass

    @abc.abstractmethod
    def reset(self, init_random=False):
        pass

    def samples(self, n_max, init_random=False):

        self.reset(init_random)

        n = 0
        while n < n_max:
            yield self.__next__()
            n += 1
