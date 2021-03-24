# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils.version import LooseVersion as _LooseVersion

try:
    from mpi4py import MPI

    mpi_available = True

    #  We don't use the standard communicator because Jax and
    # np/python must use different ones to avoid desync issues
    # so we insulate also from the user-space.
    MPI_py_comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
    MPI_jax_comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())

    n_nodes = MPI_py_comm.Get_size()
    node_number = MPI_py_comm.Get_rank()
    rank = MPI_py_comm.Get_rank()

    import jax

    if not jax.config.omnistaging_enabled:
        raise RuntimeError(
            "MPI requires jax omnistaging to be enabled."
            + "check jax documentation to enable it or uninstall mpi."
        )

    import mpi4jax


except ImportError:
    mpi_available = False
    MPI_py_comm = None
    MPI_jax_comm = None
    n_nodes = 1
    node_number = 0
    rank = 0

    class FakeMPI:
        COMM_WORLD = None

    MPI = FakeMPI()

if mpi_available:
    _min_mpi4jax_version = "0.2.11"
    if not _LooseVersion(mpi4jax.__version__) >= _LooseVersion(_min_mpi4jax_version):
        raise ImportError(
            "Netket is only compatible with mpi4jax >= {}. Please update it (`pip install -U mpi4jax`).".format(
                _min_mpi4jax_version
            )
        )
