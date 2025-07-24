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

from netket.utils.mpi.mpi import (
    mpi4jax_available as _depr_available,
    MPI as _depr_MPI,
    MPI_py_comm as _depr_MPI_py_comm,
    MPI_jax_comm as _depr_MPI_jax_comm,
    n_nodes as _depr_n_nodes,
    node_number as _depr_node_number,
    rank as _depr_rank,
)

from netket.utils.mpi.primitives import (
    mpi_all as _depr_mpi_all,
    mpi_allgather as _depr_mpi_allgather,
    mpi_any as _depr_mpi_any,
    mpi_bcast as _depr_mpi_bcast,
    mpi_max as _depr_mpi_max,
    mpi_mean as _depr_mpi_mean,
    mpi_sum as _depr_mpi_sum,
    mpi_all_jax as _depr_mpi_all_jax,
    mpi_allgather_jax as _depr_mpi_allgather_jax,
    mpi_any_jax as _depr_mpi_any_jax,
    mpi_bcast_jax as _depr_mpi_bcast_jax,
    mpi_max_jax as _depr_mpi_max_jax,
    mpi_mean_jax as _depr_mpi_mean_jax,
    mpi_sum_jax as _depr_mpi_sum_jax,
    mpi_gather as _depr_mpi_gather,
    mpi_gather_jax as _depr_mpi_gather_jax,
    mpi_alltoall_jax as _depr_mpi_alltoall_jax,
    mpi_reduce_sum_jax as _depr_mpi_reduce_sum_jax,
    mpi_allreduce_sum_jax as _depr_mpi_allreduce_sum_jax,
    mpi_scatter_jax as _depr_mpi_scatter_jax,
)

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

msg = (
    "netket.utils.mpi is deprecated, as MPI support has been removed. "
    "Refer to `jax.process_count()` and `jax.process_index()` for process count and rank. "
    "And to other jax functions for distributed operations."
)
_deprecations = {
    # July 2025, NetKet 3.19
    "available": (msg, _depr_available),
    "MPI": (msg, _depr_MPI),
    "MPI_py_comm": (msg, _depr_MPI_py_comm),
    "MPI_jax_comm": (msg, _depr_MPI_jax_comm),
    "n_nodes": (msg, _depr_n_nodes),
    "node_number": (msg, _depr_node_number),
    "rank": (msg, _depr_rank),
    "mpi_all": (msg, _depr_mpi_all),
    "mpi_allgather": (msg, _depr_mpi_allgather),
    "mpi_any": (msg, _depr_mpi_any),
    "mpi_bcast": (msg, _depr_mpi_bcast),
    "mpi_max": (msg, _depr_mpi_max),
    "mpi_mean": (msg, _depr_mpi_mean),
    "mpi_sum": (msg, _depr_mpi_sum),
    "mpi_all_jax": (msg, _depr_mpi_all_jax),
    "mpi_allgather_jax": (msg, _depr_mpi_allgather_jax),
    "mpi_any_jax": (msg, _depr_mpi_any_jax),
    "mpi_bcast_jax": (msg, _depr_mpi_bcast_jax),
    "mpi_max_jax": (msg, _depr_mpi_max_jax),
    "mpi_mean_jax": (msg, _depr_mpi_mean_jax),
    "mpi_sum_jax": (msg, _depr_mpi_sum_jax),
    "mpi_gather": (msg, _depr_mpi_gather),
    "mpi_gather_jax": (msg, _depr_mpi_gather_jax),
    "mpi_alltoall_jax": (msg, _depr_mpi_alltoall_jax),
    "mpi_reduce_sum_jax": (msg, _depr_mpi_reduce_sum_jax),
    "mpi_allreduce_sum_jax": (msg, _depr_mpi_allreduce_sum_jax),
    "mpi_scatter_jax": (msg, _depr_mpi_scatter_jax),
}

__getattr__ = _deprecation_getattr(__name__, _deprecations)

del _deprecation_getattr
