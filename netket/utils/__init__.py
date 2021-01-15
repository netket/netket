from .jax import jit_if_singleproc, get_afun_if_module
from .mpi import mpi_available, MPI, MPI_comm, n_nodes, node_number, rank
from .optional_deps import torch_available, tensorboard_available, backpack_available
from .seed import random_seed

from .deprecation import warn_deprecation, deprecated, wraps_legacy
from .moduletools import _hide_submodules, rename

jax_available = True
flax_available = True
mpi4jax_available = mpi_available

_hide_submodules(__name__, remove_self=False)

from . import flax
