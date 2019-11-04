from .._C_netket import sampler as c_sampler
from .abstract_sampler import AbstractSampler
from .py_metropolis_local import PyMetropolisLocal


class MetropolisLocal(AbstractSampler):
    def __init__(self, machine, n_chains, sweep_size=None, batch_size=None):
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisLocal(
                machine=machine,
                n_chains=n_chains,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisLocal(
                machine=machine, n_chains=n_chains, sweep_size=sweep_size
            )
        super().__init__(machine, (n_chains, machine.hilbert.size))

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()
