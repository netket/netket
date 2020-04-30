from ._C_netket.utils import *
from mpi4py import MPI


MPI_comm = MPI.COMM_WORLD
n_nodes = MPI_comm.Get_size()
node_number = MPI_comm.Get_rank()


try:
    import os

    os.environ["JAX_ENABLE_X64"] = "1"

    import jax

    jax_available = True
except ImportError:
    jax_available = False


try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False
