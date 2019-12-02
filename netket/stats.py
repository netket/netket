from ._C_netket.stats import *
from ._C_netket.stats import _subtract_mean, _compute_mean


def subtract_mean(x):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.
    """
    return _subtract_mean(x.reshape(-1, x.shape[-1]))


def compute_mean(x):
    """
    Computes the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.
    """
    return _compute_mean(x.reshape(-1, x.shape[-1]))
