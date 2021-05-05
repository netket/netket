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

from .mpi import (
    mpi_available,
    mpi4jax_available,
    MPI,
    MPI_py_comm,
    MPI_jax_comm,
    n_nodes,
    node_number,
    rank,
)

from .primitives import mpi_sum, mpi_mean, mpi_any, mpi_bcast, mpi_allgather, mpi_max
from .primitives import (
    mpi_sum_jax,
    mpi_mean_jax,
    mpi_any_jax,
    mpi_bcast_jax,
    mpi_allgather_jax,
    mpi_max_jax,
)
