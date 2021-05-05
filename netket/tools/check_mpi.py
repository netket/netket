# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from netket.utils.mpi import mpi_available, mpi4jax_available, rank, n_nodes

from ._cpu_info import available_cpus


def check_mpi():
    """
    When called via::

        # python3 -m netket.tools.check_mpi
        mpi_available                : True
        mpi4jax_available            : True
        avalable_cpus (rank 0)       : 12
        n_nodes                      : 1
        mpi4py | MPI version         : (3, 1)
        mpi4py | MPI library_version : Open MPI v4.1.0, ...

    this will print out basic MPI information to allow users to check whether
    the environment has been set up correctly.
    """
    if rank > 0:
        return

    info = {
        "mpi_available": mpi_available,
        "mpi4jax_available": mpi4jax_available,
        "avalable_cpus (rank 0)": available_cpus(),
    }
    if mpi_available:
        from mpi4py import MPI

        info.update(
            {
                "n_nodes": n_nodes,
                "mpi4py | MPI version": MPI.Get_version(),
                "mpi4py | MPI library_version": MPI.Get_library_version(),
            }
        )

    maxkeylen = max(len(k) for k in info.keys())

    for k, v in info.items():
        print(f"{k:{maxkeylen}} : {v}")


check_mpi()
