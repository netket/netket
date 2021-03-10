from functools import singledispatch
import numpy as _np

from netket.utils import mpi_available as _mpi_available, n_nodes as _n_nodes

if _mpi_available:
    from netket.utils import MPI_py_comm as _MPI_py_comm
    from netket.utils import MPI_jax_comm as _MPI_jax_comm
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
@sum_inplace.register(int)
def sum_inplace_scalar(a):
    ar = _np.asarray(a)

    if _n_nodes > 1:
        _MPI_py_comm.Allreduce(_MPI.IN_PLACE, ar.reshape(-1), op=_MPI.SUM)

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
        _MPI_py_comm.Allreduce(_MPI.IN_PLACE, a.reshape(-1), op=_MPI.SUM)

    return a


##############
# Jax
#
from netket.utils import jax_available, mpi4jax_available

if jax_available:
    import numpy as _np
    import jax

    if mpi4jax_available:
        import mpi4jax

        @sum_inplace.register(jax.interpreters.xla.DeviceArray)
        @sum_inplace.register(jax.interpreters.partial_eval.JaxprTracer)
        @sum_inplace.register(jax.interpreters.partial_eval.DynamicJaxprTracer)
        @sum_inplace.register(jax.interpreters.ad.JVPTracer)
        def sum_inplace_jax(x):
            if _n_nodes == 1:
                return x
            else:
                # Note: We must supply a token because we can't transpose `create_token`.
                # The token can't depend on x for the same reason
                # This token depends on a constant and will be eliminated by DCE
                token = jax.lax.create_token(0)
                res, _ = mpi4jax.allreduce(
                    x, op=_MPI.SUM, comm=_MPI_jax_comm, token=token
                )
                return res

    else:

        @sum_inplace.register(jax.interpreters.xla.DeviceArray)
        @sum_inplace.register(jax.interpreters.partial_eval.JaxprTracer)
        @sum_inplace.register(jax.interpreters.partial_eval.DynamicJaxprTracer)
        @sum_inplace.register(jax.interpreters.ad.JVPTracer)
        def sum_inplace_jax(x):
            if _n_nodes == 1:
                return x
            else:
                raise RuntimeError(
                    "The package mpi4jax is required in order to use jax machines with SR and\
                    more than one MPI process. To solve this issue, run `pip install mpi4jax` and restart netket."
                )
