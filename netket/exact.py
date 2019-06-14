from ._C_netket.exact import *

import itertools


def _ImagTimePropagation_iter(self, dt, n_iter=None):
    """
    iter(self: ImagTimePropagation, dt: float, n_iter: int=None) -> int

    Returns a generator which advances the time evolution by dt,
    yielding after every step.

    Args:
        dt (float): The size of the time step.
        n_iter (int=None): The number of steps or None, for no limit.

    Yields:
        int: The current step.
    """
    for i in itertools.count():
        if n_iter and i >= n_iter:
            return
        self.advance(dt)
        yield i


ImagTimePropagation.iter = _ImagTimePropagation_iter
