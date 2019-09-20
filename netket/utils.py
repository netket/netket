from ._C_netket.utils import *
from ._C_netket.utils import _subtract_mean


def subtract_mean(x):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.
    """
    return _subtract_mean(x.reshape(-1, x.shape[-1]))
