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
    return _np.random.randint(low, high)


# By default, the generator is initialized with a random seed (on node 0)
# and then propagated correctly to the other nodes
seed(None)
