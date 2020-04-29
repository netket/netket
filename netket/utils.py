from ._C_netket.utils import *
from mpi4py import MPI


MPI_comm = MPI.COMM_WORLD
n_nodes = MPI_comm.Get_size()
node_number = MPI_comm.Get_rank()
