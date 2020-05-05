# Note: Think of this as a function to be called as sum_inplace(x)
# The fact that it's a metaclass is just an implementation detail to
# allow extensible dispatching
class sum_inplace:
    """
    Performs a distributed-sum reduction across several processes. This is
    a wrapper dispatching to a specific method depending on the type of the
    input.

    It is semantically equivalent to MPI.Allreduce(MPI_IN_PLACE, x, MPI_SUM).

    Methods for specific types can be defined with the @define_sum_inplace(atype=type)
    decorator.

    Args:
        x : A N-dimensional array from numpy, jax's DeviceArray or other packages.

    Returns:
        x: The array itself
    """

    # Dictionary holding the Type <-> method dispatch table
    funcs = {}

    def __new__(mcls, arr):
        t = type(arr)
        return mcls.funcs[t](arr)


def define_sum_inplace(atype):
    """
    Defines a method implementing sum_inplace for a specific array type
    atype.

    To be used as a decorator.

    Args:
        atype: type to use for dispatch
    Returns:
        Decorator
    """

    def _define_sum_inplace(func):
        sum_inplace.funcs[atype] = func
        return func

    return _define_sum_inplace


#######
# MPI
from netket.stats.mpi_stats import _MPI_comm, _n_nodes
from netket.stats.mpi_stats import MPI as _MPI
import numpy as _np


@define_sum_inplace(atype=_np.ndarray)
def sum_inplace_MPI(a):
    """
    Computes the elementwise sum of a numpy array over all MPI processes.

    Args:
        a (numpy.ndarray): The input array, which will be overwritten in place.
    """
    _MPI_comm.Allreduce(_MPI.IN_PLACE, a.reshape(-1), op=_MPI.SUM)
    return a


#######
# Jax
from netket.utils import jax_available

if jax_available:
    import numpy as _np
    import jax

    @define_sum_inplace(atype=jax.interpreters.xla.DeviceArray)
    def sum_inplacememaybe_jax(x):
        # if not isinstance(x, jax.interpreters.xla.DeviceArray):
        #    raise TypeError("Argument to sum_inplace_jax must be a DeviceArray, got {}"
        #            .format(type(x)))

        if _n_nodes == 1:
            return x
        # This below only works on cpus...
        # we should make this work for gpus too..
        # TODO: unsafe_buffer_pointer is considered not yet definitive interface
        ptr = x.block_until_ready().device_buffer.unsafe_buffer_pointer()

        # The above is faster.
        # This below should work more often, but might copy.
        # Depending on future changes in jaxlib, we might have to switch to
        # this below.
        # _x = jax.xla._force(x.block_until_ready())
        # ptr = _x.device_buffer.unsafe_buffer_pointer()

        # see Google/jax#2123 and #1009
        data_pointer = _np.ctypeslib.ndpointer(x.dtype, shape=x.shape)

        # wrap jax data into a standard numpy array which is handled by MPI
        arr = data_pointer(ptr).contents
        _MPI_comm.Allreduce(_MPI.IN_PLACE, arr.reshape(-1), op=_MPI.SUM)

        return x
