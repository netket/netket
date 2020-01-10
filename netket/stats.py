from ._C_netket.stats import *
from ._C_netket.stats import _subtract_mean


def subtract_mean(x):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.
    """
    return _subtract_mean(x.reshape(-1, x.shape[-1]))


from mpi4py import MPI
import numpy as _np

_MPI_comm = MPI.COMM_WORLD

_n_nodes = _MPI_comm.Get_size()


def mean(a, axis=None, dtype=None, out=None):
    """
    Compute the arithmetic mean along the specified axis and over MPI processes.

    Returns the average of the array elements. The average is taken over the flattened array by default,
    otherwise over the specified axis. float64 intermediate and return values are used for integer inputs.
    """

    out = _np.mean(a, axis=axis, dtype=None, out=out)

    _MPI_comm.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.SUM)

    out /= float(_n_nodes)

    return out
