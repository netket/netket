import pytest

import tarfile
import glob

import netket as nk
from jax.nn.initializers import normal

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


def test_tar(vstate, tmp_path):
    path = str(tmp_path) + "/dir1/dir2"

    # check that overwriting works
    for k in range(1, 3):
        log = nk.logging.StateLog(path, "w", tar=True, save_every=k)

        for i in range(10):
            log(i, None, vstate)

        log.close()
        tfile = tarfile.TarFile(path + ".tar", "r")
        files = tfile.getnames()

        assert len(files) == 10 / k
        assert log._file_step == len(files)

        for file in files:
            assert file.endswith(".mpack")

    # check that x fails
    with pytest.raises(ValueError):
        log = nk.logging.StateLog(path, "x", tar=True)

    tfile = tarfile.TarFile(path + ".tar", "r")
    files = tfile.getnames()

    # test appending
    log = nk.logging.StateLog(path, "a", tar=True)
    log._init_output()
    assert log._file_step == 5
    for i in range(10):
        log(i, None, vstate)

    assert log._file_step == 10 + 5

    del log
    tfile = tarfile.TarFile(path + ".tar", "r")
    files = tfile.getnames()

    assert len(files) == 10 + 5

    for file in files:
        assert file.endswith(".mpack")


def test_dir(vstate, tmp_path):
    path = str(tmp_path) + "/dir1/dir2"

    # check that overwriting works
    for k in range(1, 3):
        log = nk.logging.StateLog(path, "w", tar=False, save_every=k)

        for i in range(10):
            log(i, None, vstate)

        files = glob.glob(path + "/*.mpack")
        assert len(files) == 10 / k
        assert log._file_step == len(files)

        for file in files:
            assert file.endswith(".mpack")

    # check that x fails
    with pytest.raises(ValueError):
        log = nk.logging.StateLog(path, "x", tar=False)

    # test appending
    log = nk.logging.StateLog(path, "a", tar=False)
    log._init_output()
    assert log._file_step == 5
    for i in range(10):
        log(i, None, vstate)

    assert log._file_step == 10 + 5

    files = glob.glob(path + "/*.mpack")

    assert len(files) == 10 + 5

    for file in files:
        assert file.endswith(".mpack")


def test_lazy_init(tmp_path):
    path = str(tmp_path) + "/dir1/dir2"

    # check that overwriting works
    log = nk.logging.StateLog(path, "w", tar=False, save_every=1)

    files = glob.glob(path + "/*")
    assert len(files) == 0
