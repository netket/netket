import numpy as _np
from ._sum_inplace import sum_inplace as _sum_inplace

from netket.utils.mpi import (
    mpi_available as _mpi_available,
    n_nodes as _n_nodes,
    MPI_py_comm as MPI_comm,
)

if _mpi_available:
    from netket.utils.mpi import MPI


def subtract_mean(x, axis=None):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.

    Args:
        x: Input array
        axis: Axis or axes along which the means are computed. The default (None) is to
              compute the mean of the flattened array.

    Returns:
        The resulting array.

    """
    x_mean = mean(x, axis=axis)
    x -= x_mean

    return x


def mean(a, axis=None, keepdims: bool = False):
    """
    Compute the arithmetic mean along the specified axis and over MPI processes.

    Returns the average of the array elements. The average is taken over the flattened array by default,
    otherwise over the specified axis. float64 intermediate and return values are used for integer inputs.

    Args:
        a: The input array
        axis: Axis or axes along which the means are computed. The default (None) is to
              compute the mean of the flattened array.
        keepdims: If True the output array will have the same number of dimensions as the input,
              with the reduced axes having length 1. (default=False)

    Returns:
        The array with reduced dimensions defined by axis.

    """
    out = a.mean(axis=axis, keepdims=keepdims)

    out = _sum_inplace(out)
    out /= _n_nodes

    return out


def sum(a, axis=None, out=None, keepdims: bool = False):
    """
    Compute the sum along the specified axis and over MPI processes.

    Args:
        a: The input array
        axis: Axis or axes along which the mean is computed. The default (None) is to
              compute the mean of the flattened array.
        out: An optional pre-allocated array to fill with the result.
        keepdims: If True the output array will have the same number of dimensions as the input,
              with the reduced axes having length 1. (default=False)

    Returns:
        The array with reduced dimensions defined by axis. If out is not none, returns out.

    """
    # asarray is necessary for the axis=None case to work, as the MPI call requires a NumPy array
    out = _np.asarray(_np.sum(a, axis=axis, out=out, keepdims=keepdims))

    if _n_nodes > 1:
        MPI_comm.Allreduce(MPI.IN_PLACE, out.reshape(-1), op=MPI.SUM)

    return out


def var(a, axis=None, out=None, ddof: int = 0):
    """
    Compute the variance mean along the specified axis and over MPI processes.

    Args:
        a: The input array
        axis: Axis or axes along which the variance is computed. The default (None) is to
              compute the variance of the whole flattened array.
        out: An optional pre-allocated array to fill with the result.
        ddof: “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
              where N represents the number of elements. By default ddof is zero.

    Returns:
        The array with reduced dimensions defined by axis. If out is not none, returns out.

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
    """
    Compute the total number of elements stored in the input array among all MPI processes.

    This function essentially returns MPI_sum_among_processes(a.size).

    Args:
        a: The input array.
        axis: If specified, only considers the total size of that axis.

    Returns:
        a.size or a.shape[axis], reduced among all MPI processes.
    """
    if axis is None:
        l_size = a.size
    else:
        l_size = a.shape[axis]

    if _n_nodes > 1:
        l_size = MPI_comm.allreduce(l_size, op=MPI.SUM)

    return l_size
