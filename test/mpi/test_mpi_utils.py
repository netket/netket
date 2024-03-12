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

import jax
import numpy as np
import jax.numpy as jnp

import netket as nk
from netket.utils import mpi

from .. import common


@common.onlyif_mpi
def test_key_split(_mpi_size, _mpi_comm, _mpi_rank):
    from mpi4py import MPI

    size = _mpi_size
    comm = _mpi_comm

    key = nk.jax.PRNGKey(1256)

    keys = comm.allgather(key)
    assert all([jnp.all(k == key) for k in keys])

    key, _ = nk.jax.mpi_split(key)
    keys = MPI.COMM_WORLD.allgather(key)

    # print(f"{comm.Get_rank()} : k{key}, {type(key)}")
    # print(f"{comm.Get_rank()} : ks{keys}, {type(keys)}, {type(keys[0])}")

    for r, ki in enumerate(keys):
        if _mpi_rank == r:
            assert np.array(key) == np.array(ki)
        else:
            assert not np.array(key) == np.array(ki)

    assert len(keys) == size


# @common.onlyif_mpi
@common.named_parametrize("complex_", [False, True])
def test_mpi_logsumexp(_mpi_size, _mpi_comm, _mpi_rank, complex_):
    s = (16 * _mpi_size, 3, 2)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(123), 4)
    a = jax.random.uniform(k1, shape=s)
    b = jax.random.uniform(k2, shape=s)
    if complex_:
        a = jax.lax.complex(a, jax.random.uniform(k3, shape=s))
        b = jax.lax.complex(b, jax.random.uniform(k4, shape=s))

    a_ = a.reshape(
        (
            _mpi_size,
            -1,
        )
        + a.shape[1:]
    )[_mpi_rank]
    b_ = b.reshape((_mpi_size, -1) + a.shape[1:])[_mpi_rank]

    y1 = jax.scipy.special.logsumexp(a, axis=0)
    y2, _ = mpi.mpi_logsumexp(a_, axis=0)
    np.testing.assert_allclose(y1, y2)

    y1 = jax.scipy.special.logsumexp(a)
    y2, _ = mpi.mpi_logsumexp(a_)
    np.testing.assert_allclose(y1, y2)

    y1 = jax.scipy.special.logsumexp(a, axis=1)
    y1 = y1.reshape(
        (
            _mpi_size,
            -1,
        )
        + y1.shape[1:]
    )[_mpi_rank]
    y2, _ = mpi.mpi_logsumexp(a_, axis=1)
    np.testing.assert_allclose(y1, y2)

    y1 = jax.scipy.special.logsumexp(a, axis=0, b=b)
    y2, _ = mpi.mpi_logsumexp(a_, axis=0, b=b_)
    np.testing.assert_allclose(y1, y2)

    y1 = jax.scipy.special.logsumexp(a, b=b)
    y2, _ = mpi.mpi_logsumexp(a_, b=b_)
    np.testing.assert_allclose(y1, y2)

    y1 = jax.scipy.special.logsumexp(a, axis=1, b=b)
    y1 = y1.reshape(
        (
            _mpi_size,
            -1,
        )
        + y1.shape[1:]
    )[_mpi_rank]
    y2, _ = mpi.mpi_logsumexp(a_, axis=1, b=b_)
    np.testing.assert_allclose(y1, y2)

    y1 = jax.scipy.special.logsumexp(a, b=b, axis=(0, 1))
    y2, _ = mpi.mpi_logsumexp(a_, b=b_, axis=(0, 1))
    np.testing.assert_allclose(y1, y2)


@common.named_parametrize("complex_", [False, True])
def test_mpi_logsumexp_scalar(_mpi_size, _mpi_comm, _mpi_rank, complex_):
    s = (_mpi_size,)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(123), 4)
    a = jax.random.uniform(k1, shape=s)
    b = jax.random.uniform(k2, shape=s)
    if complex_:
        a = jax.lax.complex(a, jax.random.uniform(k3, shape=s))
        b = jax.lax.complex(b, jax.random.uniform(k4, shape=s))

    print(_mpi_rank, "a", a[_mpi_rank])
    print(_mpi_rank, "a_", a)

    # y1 = jax.scipy.special.logsumexp(a, b=b)
    # y2, _ = mpi.mpi_logsumexp(a[_mpi_rank], b=b[_mpi_rank])
    y1 = jax.scipy.special.logsumexp(a)
    y2, _ = mpi.mpi_logsumexp(a[_mpi_rank])
    print(_mpi_rank, "y1", y1.shape, y1)
    print(_mpi_rank, "y2", y2.shape, y2)
    np.testing.assert_allclose(y1, y2)
