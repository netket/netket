import numpy as _np
from mpi4py import MPI


def subtract_mean(x, axis=None):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.

    Args:
        axis: Axis or axes along which the means are computed. The default is to
              compute the mean of the flattened array.
    """
    x_mean = mean(x, axis=axis)
    x -= x_mean

    return x


_MPI_comm = MPI.COMM_WORLD
_n_nodes = _MPI_comm.Get_size()


def mean(a, axis=None, out=None):
    """
    Compute the arithmetic mean along the specified axis and over MPI processes.

    Returns the average of the array elements. The average is taken over the flattened array by default,
    otherwise over the specified axis. float64 intermediate and return values are used for integer inputs.
    """
    # asarray is necessary for the axis=None case to work, as the MPI call requires a NumPy array
    out = _np.asarray(_np.mean(a, axis=axis, out=out))

    _MPI_comm.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.SUM)
    out /= _n_nodes

    return out


def sum(a, axis=None, out=None):
    """
    Compute the arithmetic mean along the specified axis and over MPI processes.
    """
    # asarray is necessary for the axis=None case to work, as the MPI call requires a NumPy array
    out = _np.asarray(_np.sum(a, axis=axis, out=out))
    _MPI_comm.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.SUM)
    return out


def var(a, axis=None, out=None, ddof=0):
    """
    Compute the variance mean along the specified axis and over MPI processes.
    """
    m = mean(a, axis=axis)

    if axis is None:
        ssq = _np.abs(a - m) ** 2.0
    else:
        ssq = _np.abs(a - _np.expand_dims(m, axis)) ** 2.0

    out = sum(ssq, axis=axis, out=out)

    n_all = total_size(a, axis=axis)
    out /= n_all - ddof

    return out


def total_size(a, axis=None):
    if axis is None:
        l_size = a.size
    else:
        l_size = a.shape[axis]

    l_size = _MPI_comm.allreduce(l_size, op=MPI.SUM)
    return l_size
