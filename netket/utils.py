from mpi4py import MPI


MPI_comm = MPI.COMM_WORLD
n_nodes = MPI_comm.Get_size()
node_number = MPI_comm.Get_rank()

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
