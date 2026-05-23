import pytest

import glob
from collections import namedtuple

import numpy as np
import jax
from jax.nn.initializers import normal
from jax import numpy as jnp

import netket as nk
import netket.experimental as nkx

from .. import common


@pytest.fixture()
def vstate(request):
    N = 8
    hi = nk.hilbert.Spin(1 / 2, N)

    ma = nk.models.RBM(
        alpha=1,
        param_dtype=float,
        hidden_bias_init=normal(),
        visible_bias_init=normal(),
    )

    return nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )


@common.skipif_distributed
def test_hdf5log(vstate, tmp_path):
    # skip test if hdf5py not installed
    h5py = pytest.importorskip("h5py")

    path = str(tmp_path) + "/dir1/dir2"

    log = nk.logging.HDF5Log(path + "/output")

    for i in range(30):
        log(i, {"Energy": jnp.array(1.0), "complex": jnp.array(1.0 + 1j)}, vstate)

    log.flush()
    del log

    files = glob.glob(path + "/*")
    assert len(files) >= 1

    f = h5py.File(files[0], "r")
    energy = np.array(f["data/Energy/value"])
    complex = np.array(f["data/complex/value"])
    params = np.array(f["variational_state/parameters/Dense/kernel"])
    iters = np.array(f["variational_state/iter"])
    assert energy.shape[0] == 30
    assert complex.shape[0] == 30
    assert params.shape[0] == 30
    assert iters.shape[0] == 30
    assert f["data/Energy/value"].chunks[0] > 1


@common.skipif_distributed
def test_hdf5log_preserves_structured_layout(tmp_path):
    h5py = pytest.importorskip("h5py")

    Pair = namedtuple("Pair", ["x", "y"])
    path = tmp_path / "output.h5"

    log = nk.logging.HDF5Log(str(path), save_params=False)
    log(
        7,
        {
            "scalar": jnp.array(1.0),
            "alist": [jnp.array(2.0), jnp.array(3.0)],
            "atuple": (jnp.array(4.0), jnp.array(5.0)),
            "pair": Pair(jnp.array(6.0), jnp.array(7.0)),
            "stats": nk.stats.Stats(
                mean=8.0,
                error_of_mean=0.1,
                variance=0.2,
                tau_corr=0.3,
                R_hat=1.0,
            ),
        },
        None,
    )
    log.close()

    with h5py.File(path, "r") as f:
        assert np.array_equal(np.array(f["data/scalar/iter"]), np.array([7]))
        assert np.array_equal(np.array(f["data/scalar/value"]), np.array([1.0]))

        assert np.array_equal(np.array(f["data/alist/0/iter"]), np.array([7]))
        assert np.array_equal(np.array(f["data/alist/0/value"]), np.array([2.0]))
        assert np.array_equal(np.array(f["data/alist/1/iter"]), np.array([7]))
        assert np.array_equal(np.array(f["data/alist/1/value"]), np.array([3.0]))

        assert np.array_equal(np.array(f["data/atuple/iter"]), np.array([7]))
        assert np.array_equal(np.array(f["data/atuple/0"]), np.array([4.0]))
        assert np.array_equal(np.array(f["data/atuple/1"]), np.array([5.0]))

        assert np.array_equal(np.array(f["data/pair/iter"]), np.array([7]))
        assert np.array_equal(np.array(f["data/pair/x"]), np.array([6.0]))
        assert np.array_equal(np.array(f["data/pair/y"]), np.array([7.0]))

        assert np.array_equal(np.array(f["data/stats/iter"]), np.array([7]))
        assert np.array_equal(np.array(f["data/stats/Mean"]), np.array([8.0]))
        assert np.array_equal(np.array(f["data/stats/Sigma"]), np.array([0.1]))


@common.skipif_distributed
def test_lazy_init(tmp_path):
    # skip test if hdf5py not installed
    pytest.importorskip("h5py")

    path = str(tmp_path) + "/dir1"

    nk.logging.HDF5Log(path)

    files = glob.glob(path + "/*")
    assert len(files) == 0


@common.onlyif_distributed
def test_write_only_on_master(vstate, tmp_path):
    # Check that the logger runs everywhere but serializes only on rank 0
    # skip test if tensorboardX not installed
    pytest.importorskip("h5py")

    rank = jax.process_index()
    path = str(tmp_path) + "/dir1/r{rank}"

    log = nk.logging.HDF5Log(path + "/output")

    for i in range(30):
        log(i, {"Energy": jnp.array(1.0), "complex": jnp.array(1.0 + 1j)}, vstate)

    log.flush()
    del log

    files = glob.glob(path + "/*")
    if rank == 0:
        assert len(files) >= 1
    else:
        assert len(files) == 0


@common.skipif_distributed
def test_flush_saves_final_parameters(vstate, tmp_path):
    h5py = pytest.importorskip("h5py")

    path = tmp_path / "output.h5"

    log = nk.logging.HDF5Log(str(path), save_params_every=10)
    for i in range(3):
        log(i, {"Energy": jnp.array(float(i))}, vstate)

    log.flush(vstate)
    log.close()

    with h5py.File(path, "r") as f:
        iters = np.array(f["variational_state/iter"])

    assert np.array_equal(iters, np.array([0, 3]))


@common.skipif_distributed
def test_append_mode_reopens_existing_file(vstate, tmp_path):
    h5py = pytest.importorskip("h5py")

    path = tmp_path / "output.h5"

    log = nk.logging.HDF5Log(str(path), mode="write")
    for i in range(2):
        log(i, {"Energy": jnp.array(float(i))}, vstate)
    log.close()

    log = nk.logging.HDF5Log(str(path), mode="append")
    for i in range(2, 4):
        log(i, {"Energy": jnp.array(float(i))}, vstate)
    log.close()

    with h5py.File(path, "r") as f:
        energy = np.array(f["data/Energy/value"])
        iters = np.array(f["data/Energy/iter"])

    assert np.array_equal(energy, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(iters, np.array([0, 1, 2, 3]))


@common.skipif_distributed
def test_flush_does_not_close_writer(vstate, tmp_path):
    pytest.importorskip("h5py")

    path = tmp_path / "output.h5"

    log = nk.logging.HDF5Log(str(path))
    log(0, {"Energy": jnp.array(1.0)}, vstate)
    log.flush(vstate)

    assert log._writer is not None
    assert log._writer.id.valid

    log.close()


@common.skipif_distributed
def test_mode_aliases(vstate, tmp_path):
    pytest.importorskip("h5py")

    path = tmp_path / "output.h5"

    log = nk.logging.HDF5Log(str(path), mode="w")
    log(0, {"Energy": jnp.array(1.0)}, vstate)
    log.close()

    log = nk.logging.HDF5Log(str(path), mode="a")
    log(1, {"Energy": jnp.array(2.0)}, vstate)
    log.close()


def test_experimental_hdf5log_is_deprecated_alias():
    with pytest.warns(
        DeprecationWarning,
        match=r"netket\.experimental\.logging\.HDF5Log is now stable",
    ):
        HDF5Log = nkx.logging.HDF5Log

    assert HDF5Log is nk.logging.HDF5Log
