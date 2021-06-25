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

import jax.numpy as jnp

import netket as nk

from .. import common


@common.onlyif_mpi
def test_key_split(_mpi_size, _mpi_comm):
    from mpi4py import MPI

    size = _mpi_size
    comm = _mpi_comm

    key = nk.jax.PRNGKey(1256)

    keys = comm.allgather(key)
    assert all([jnp.all(k == key) for k in keys])

    key, _ = nk.jax.mpi_split(key)
    keys = MPI.COMM_WORLD.allgather(key)
    assert all([not jnp.all(k == keys) for k in keys])
    assert len(keys) == size
