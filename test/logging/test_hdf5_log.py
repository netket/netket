import pytest

import os
import glob

import netket as nk
from jax.nn.initializers import normal
from jax import numpy as jnp

from .. import common

pytestmark = common.skipif_mpi


@pytest.fixture()
def vstate(request):
    N = 8
    hi = nk.hilbert.Spin(1 / 2, N)

    ma = nk.models.RBM(
        alpha=1,
        dtype=float,
        hidden_bias_init=normal(),
        visible_bias_init=normal(),
    )

    return nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )


def test_hdf5log(vstate, tmp_path):
    path = str(tmp_path) + "/dir1/dir2"

    log = nk.logging.HDF5Log(path+'/output', save_params_every=10)

    for i in range(30):
        log(i, {"Energy": jnp.array(1.0), "complex": jnp.array(1.0 + 1j)}, vstate)

    log.flush()
    del log

    files = glob.glob(path + "/*")
    assert len(files) >= 1


def test_lazy_init(tmp_path):
    path = str(tmp_path) + "/dir1"

    log = nk.logging.HDF5Log(path)

    files = glob.glob(path + "/*")
    assert len(files) == 0
