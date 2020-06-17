try:
    from mpi4py import MPI

    mpi_available = True
    MPI_comm = MPI.COMM_WORLD
    n_nodes = MPI_comm.Get_size()
    node_number = MPI_comm.Get_rank()

except ImportError:
    mpi_available = False
    MPI_comm = None
    n_nodes = 1
    node_number = 0

try:
    import os

    os.environ["JAX_ENABLE_X64"] = "1"
    # os.environ["JAX_LOG_COMPILES"] = "1"
    import jax

    jax_available = True

    def jit_if_singleproc(f, *args, **kwargs):
        if n_nodes == 1:
            return jax.jit(f, *args, **kwargs)
        else:
            return f


except ImportError:
    jax_available = False


try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


try:
    import tensorboardX

    tensorboard_available = True
except ImportError:
    tensorboard_available = False


try:
    import backpack

    backpack_available = True
except ImportError:
    backpack_available = False
