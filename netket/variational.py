from ._C_netket.variational import *

import itertools


def _Vmc_iter(self, n_iter=None, step_size=1):
    """
    iter(self: Vmc, n_iter: int=None, step_size: int=1) -> int

    Returns a generator which advances the VMC optimization, yielding
    after every step_size steps up to n_iter.

    Args:
        n_iter (int=None): The number of steps or None, for no limit.
        step_size (int=1): The number of steps the simulation is advanced.

    Yields:
        int: The current step.
    """
    self.reset()
    for i in itertools.count(step=step_size):
        if n_iter and i >= n_iter:
            return
        self.advance(step_size)
        yield i


Vmc.iter = _Vmc_iter
