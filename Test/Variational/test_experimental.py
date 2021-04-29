import pytest

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from functools import partial
import itertools

import tarfile
import glob
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
    g = nk.graph.Chain(N)

    ma = nk.models.RBM(
        alpha=1,
        dtype=float,
        hidden_bias_init=nk.nn.initializers.normal(),
        visible_bias_init=nk.nn.initializers.normal(),
    )

    return nk.variational.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )


def test_variables_from_file(vstate, tmp_path):
    fname = str(tmp_path) + "/file.mpack"

    with open(fname, "wb") as f:
        f.write(serialization.to_bytes(vstate.variables))

    for name in [fname, fname[:-6]]:
        vstate2 = nk.variational.MCState(
            vstate.sampler, vstate.model, n_samples=10, seed=SEED + 100
        )

        vstate2.variables = nk.variational.experimental.variables_from_file(
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
        vstate2 = nk.variational.MCState(
            vstate.sampler, vstate.model, n_samples=10, seed=SEED + 100
        )

        for j in [0, 3, 8]:
            vstate2.variables = nk.variational.experimental.variables_from_tar(
                name, vstate2.variables, j
            )

            # check
            jax.tree_multimap(
                np.testing.assert_allclose, vstate.parameters, vstate2.parameters
            )

        with pytest.raises(KeyError):
            nk.variational.experimental.variables_from_tar(name, vstate2.variables, 15)


def save_binary_to_tar(tar_file, byte_data, name):
    abuf = BytesIO(byte_data)

    # Contruct the info object with the correct length
    info = tarfile.TarInfo(name=name)
    info.size = len(abuf.getbuffer())

    # actually save the data to the tar file
    tar_file.addfile(tarinfo=info, fileobj=abuf)
