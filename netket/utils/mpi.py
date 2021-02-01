# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils.version import LooseVersion as _LooseVersion

try:
    from mpi4py import MPI

    mpi_available = True
    MPI_comm = MPI.COMM_WORLD
    n_nodes = MPI_comm.Get_size()
    node_number = MPI_comm.Get_rank()
    rank = MPI_comm.Get_rank()

    import jax

    if not jax.config.omnistaging_enabled:
        raise RuntimeError(
            "MPI requires jax omnistaging to be enabled."
            + "check jax documentation to enable it or uninstall mpi."
        )

    import mpi4jax


except ImportError:
    mpi_available = False
    MPI_comm = None
    n_nodes = 1
    node_number = 0
    rank = 0

    class FakeMPI:
        COMM_WORLD = None

    MPI = FakeMPI()

if mpi_available:
    _min_mpi4jax_version = "0.2.7"
    if not _LooseVersion(mpi4jax.__version__) >= _LooseVersion(_min_mpi4jax_version):
        raise ImportError(
            "Netket is only compatible with mpi4jax >= {}. Please update it (`pip install -U mpi4jax`).".format(
                _min_mpi4jax_version
            )
        )
