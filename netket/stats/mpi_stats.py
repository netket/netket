import numpy as _np
from mpi4py import MPI


def subtract_mean(x, axis=None, dtype=None, mean_out=None):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.

    Args:
        axis: Axis or axes along which the means are computed. The default is to
              compute the mean of the flattened array.
        dtype: Type to use in computing the mean
        mean_out: pre-allocated array to store the mean
    """
    x_mean = mean(x, axis=axis, dtype=dtype, out=mean_out)

    x -= x_mean

    return x


_MPI_comm = MPI.COMM_WORLD

_n_nodes = _MPI_comm.Get_size()


def mean(a, axis=None, dtype=None, out=None):
    """
    Compute the arithmetic mean along the specified axis and over MPI processes.

    Returns the average of the array elements. The average is taken over the flattened array by default,
    otherwise over the specified axis. float64 intermediate and return values are used for integer inputs.
    """

    out = _np.mean(a, axis=axis, dtype=None, out=out)

    if _np.isscalar(out):
        out = _MPI_comm.allreduce(out, op=MPI.SUM) / float(_n_nodes)
        return out

    old_shape = out.shape
    out = out.reshape(-1)
    _MPI_comm.Allreduce(MPI.IN_PLACE, out, op=MPI.SUM)
    out /= float(_n_nodes)

    return out.reshape(old_shape)


def sum_on_nodes(a, out=None):
    """
    Computes the sum of a numpy array over MPI processes.
    """
    if out is None:
        _MPI_comm.Allreduce(MPI.IN_PLACE, a.reshape(-1), op=MPI.SUM)
        return a
    else:
        out = _np.copy(a)
        _MPI_comm.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.SUM)
        return out


def var(a, axis=None, dtype=None, out=None):
    """
    Compute the variance mean along the specified axis and over MPI processes.


    """

    m = mean(a, axis=axis, dtype=dtype, out=out)
    if axis is None or axis == 0:
        out = mean(_np.abs(a - m) ** 2.0, axis=axis, dtype=dtype, out=out)
    elif axis == 1:
        out = mean(
            _np.abs(a - m[:, _np.newaxis]) ** 2.0, axis=axis, dtype=dtype, out=out
        )
    elif axis == 2:
        out = mean(
            _np.abs(a - m[:, :, _np.newaxis]) ** 2.0, axis=axis, dtype=dtype, out=out
        )
    else:
        raise RuntimeError("var implemented only for ndim<=3.")

    return out


def total_size(a, axis=None):
    if axis is None:
        l_size = a.size
    else:
        l_size = a.shape[axis]

    l_size = _MPI_comm.allreduce(l_size, op=MPI.SUM)
    return l_size
