import numpy as _np
from numba import jit, objmode
from mpi4py import MPI


@jit
def seed(seed=None):

    with objmode(derived_seed="int64"):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        if rank == 0:
            _np.random.seed(seed)
            derived_seed = _np.random.randint(0, 1 << 32, size=size)
        else:
            derived_seed = None

        derived_seed = comm.scatter(derived_seed, root=0)

    _np.random.seed(derived_seed)


@jit
def uniform(low=0.0, high=1.0):
    return _np.random.uniform(low, high)


@jit
def randint(low, high):
    ''' Generate random integers from low (inclusive) to high (exclusive).

    Args:
        low (int): Lowest (signed) integer to be drawn from the distribution.
        high (int): One above the largest (signed) integer to be drawn from the distribution.

    Returns:
        int: A random integer uniformely distributed in [low,high).

    '''
    return _np.random.randint(low, high)

# TODO use numba version when argument p is made available


def choice(a, size=None, replace=True, p=None):
    '''Generates a random sample from a given 1-D array
    Args:
        a (1-D array-like or int): If an ndarray, a random sample is generated from its elements.
                    If an int, the random sample is generated as if a were np.arange(a)
        size (int or tuple of ints, optional): Output shape.
                    If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
                    Default is None, in which case a single value is returned.
        replace (boolean, optional): Whether the sample is with or without replacement.
        p (1-D array-like, optional): The probabilities associated with each entry in a.
                    If not given the sample assumes a uniform distribution over all entries in a.

    Returns:
        single item or ndarry: The generated random samples
    '''
    return _np.random.choice(a, size, replace, p)


# By default, the generator is initialized with a random seed (on node 0)
# and then propagated correctly to the other nodes
seed(None)
