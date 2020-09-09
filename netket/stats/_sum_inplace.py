from functools import singledispatch
import numpy as _np

from netket.utils import mpi_available as _mpi_available, n_nodes as _n_nodes

if _mpi_available:
    from netket.utils import MPI_comm as _MPI_comm
    from netket.utils import MPI as _MPI


@singledispatch
def sum_inplace(x):
    """
    Computes the elementwie sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy 
    might be returned.

    Args:
        a: The input array, which will usually be overwritten in place.
    Returns:
        out: The reduced array.
    """
    raise TypeError("Unknown type to perform dispatch upon: {}".format(type(x)))


#######
# Scalar
@sum_inplace.register(complex)
@sum_inplace.register(_np.float64)
@sum_inplace.register(_np.float32)
@sum_inplace.register(_np.complex64)
@sum_inplace.register(_np.complex128)
@sum_inplace.register(float)
def sum_inplace_scalar(a):
    ar = _np.asarray(a)

    if _n_nodes > 1:
        _MPI_comm.Allreduce(_MPI.IN_PLACE, ar.reshape(-1), op=_MPI.SUM)

    return ar


##############
# Numpy Array
#
@sum_inplace.register(_np.ndarray)
def sum_inplace_MPI(a):
    """
    Computes the elementwise sum of a numpy array over all MPI processes.

    Args:
        a (numpy.ndarray): The input array, which will be overwritten in place.
    """
    if _n_nodes > 1:
        _MPI_comm.Allreduce(_MPI.IN_PLACE, a.reshape(-1), op=_MPI.SUM)

    return a


##############
# Jax
#
from netket.utils import jax_available

if jax_available:
    import numpy as _np
    import jax

    @sum_inplace.register(jax.interpreters.xla.DeviceArray)
    def sum_inplace_jax(x):
        if not isinstance(x, jax.interpreters.xla.DeviceArray):
            raise TypeError(
                "Argument to sum_inplace_jax must be a DeviceArray, got {}".format(
                    type(x)
                )
            )

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
        # see Google/jax #2123 and #1009
        # _x = jax.xla._force(x.block_until_ready())
        # ptr = _x.device_buffer.unsafe_buffer_pointer()

        # using native numpy because jax's numpy does not have ctypeslib
        data_pointer = _np.ctypeslib.ndpointer(x.dtype, shape=x.shape)

        # wrap jax data into a standard numpy array which is handled by MPI
        arr = data_pointer(ptr).contents
        _MPI_comm.Allreduce(_MPI.IN_PLACE, arr.reshape(-1), op=_MPI.SUM)

        return x

    @sum_inplace.register(jax.interpreters.partial_eval.JaxprTracer)
    @sum_inplace.register(jax.interpreters.ad.JVPTracer)
    def sum_inplace_jax_jittracer(x):
        if _n_nodes == 1:
            return x
        else:
            raise RuntimError(
                "Cannot jit through sum_inplace when running with multiple MPI processes."
            )
