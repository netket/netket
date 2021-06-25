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

import pytest

import jax
import jax.flatten_util
import numpy as np

import tarfile
from io import BytesIO

from flax import serialization

import netket as nk

from .. import common

pytestmark = common.skipif_mpi

SEED = 111


@pytest.fixture()
def vstate(request):
    N = 8
    hi = nk.hilbert.Spin(1 / 2, N)

    ma = nk.models.RBM(
        alpha=1,
        dtype=float,
        hidden_bias_init=nk.nn.initializers.normal(),
        visible_bias_init=nk.nn.initializers.normal(),
    )

    return nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )


def test_variables_from_file(vstate, tmp_path):
    fname = str(tmp_path) + "/file.mpack"

    with open(fname, "wb") as f:
        f.write(serialization.to_bytes(vstate.variables))

    for name in [fname, fname[:-6]]:
        vstate2 = nk.vqs.MCState(
            vstate.sampler, vstate.model, n_samples=10, seed=SEED + 100
        )

        vstate2.variables = nk.vqs.experimental.variables_from_file(
            name, vstate2.variables
        )

        # check
        jax.tree_multimap(
            np.testing.assert_allclose, vstate.parameters, vstate2.parameters
        )


def test_variables_from_tar(vstate, tmp_path):
    fname = str(tmp_path) + "/file.tar"

    with tarfile.TarFile(fname, "w") as f:
        for i in range(10):
            save_binary_to_tar(
                f, serialization.to_bytes(vstate.variables), f"{i}.mpack"
            )

    for name in [fname, fname[:-4]]:
        vstate2 = nk.vqs.MCState(
            vstate.sampler, vstate.model, n_samples=10, seed=SEED + 100
        )

        for j in [0, 3, 8]:
            vstate2.variables = nk.vqs.experimental.variables_from_tar(
                name, vstate2.variables, j
            )

            # check
            jax.tree_multimap(
                np.testing.assert_allclose, vstate.parameters, vstate2.parameters
            )

        with pytest.raises(KeyError):
            nk.vqs.experimental.variables_from_tar(name, vstate2.variables, 15)


def save_binary_to_tar(tar_file, byte_data, name):
    abuf = BytesIO(byte_data)

    # Contruct the info object with the correct length
    info = tarfile.TarInfo(name=name)
    info.size = len(abuf.getbuffer())

    # actually save the data to the tar file
    tar_file.addfile(tarinfo=info, fileobj=abuf)
