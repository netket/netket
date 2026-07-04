import pytest

import glob
from collections import namedtuple

import jax
from jax.nn.initializers import normal
from jax import numpy as jnp

import netket as nk
from netket.utils.tree_walk import walk_tree_with_path
from netket._src.logging.tensorboard import (
    _collect_tensorboard_leaf,
    _expand_tensorboard_node,
    _write_tensorboard_leaf,
)

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
def test_tblog(vstate, tmp_path):
    # skip test if tensorboardX not installed
    pytest.importorskip("tensorboardX")

    path = str(tmp_path) + "/dir1/dir2"

    log = nk.logging.TensorBoardLog(path)

    for i in range(10):
        log(i, {"Energy": jnp.array(1.0), "complex": jnp.array(1.0 + 1j)}, vstate)

    log.flush()
    del log

    files = glob.glob(path + "/*")
    assert len(files) >= 1


@common.skipif_distributed
def test_tblog_tree_log_paths():
    Pair = namedtuple("Pair", ["x", "y"])

    data = []
    walk_tree_with_path(
        {
            "scalar": 1.0,
            "complex": 2.0 + 3.0j,
            "pair": Pair(4.0, 5.0),
            "stats": nk.stats.Stats(
                mean=6.0,
                error_of_mean=0.1,
                variance=0.2,
                tau_corr=0.3,
                R_hat=1.0,
            ),
        },
        "",
        visit_leaf=_collect_tensorboard_leaf,
        expand_node=_expand_tensorboard_node,
        data=data,
    )

    assert data == [
        ("/scalar", 1.0),
        ("/complex/re", 2.0),
        ("/complex/im", 3.0),
        ("/pair/x", 4.0),
        ("/pair/y", 5.0),
        ("/stats/Mean", 6.0),
        ("/stats/Variance", 0.2),
        ("/stats/Sigma", 0.1),
        ("/stats/R_hat", 1.0),
        ("/stats/TauCorr", 0.3),
    ]


class _FakeWriter:
    """Minimal writer capturing add_scalar calls for testing."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


@common.skipif_distributed
def test_tblog_writes_jax_scalars():
    # Regression test for gh #2244: jax/numpy 0-dim array scalars (e.g.
    # `acceptance`, which arrives as a jaxlib ArrayImpl) were silently dropped
    # because they are not `numbers.Number` instances.
    import numpy as np

    writer = _FakeWriter()
    item = {
        "acceptance": jnp.array(0.7),  # jax real scalar
        "energy_complex": jnp.array(1.0 + 2.0j),  # jax complex scalar
        "np_scalar": np.array(3.0),  # numpy 0-dim scalar
        "vector": jnp.array([1.0, 2.0, 3.0]),  # non-scalar -> skipped
    }

    walk_tree_with_path(
        item,
        "",
        visit_leaf=_write_tensorboard_leaf,
        expand_node=_expand_tensorboard_node,
        writer=writer,
        step=0,
    )

    tags = {tag: value for tag, value, _ in writer.scalars}
    assert tags["acceptance"] == 0.7
    assert tags["np_scalar"] == 3.0
    assert tags["energy_complex/re"] == 1.0
    assert tags["energy_complex/im"] == 2.0
    # multi-element arrays are not scalars and must not be written
    assert "vector" not in tags


@common.skipif_distributed
def test_lazy_init(tmp_path):
    # skip test if tensorboardX not installed
    pytest.importorskip("tensorboardX")

    path = str(tmp_path) + "/dir1"

    nk.logging.TensorBoardLog(path)

    files = glob.glob(path + "/*")
    assert len(files) == 0


@common.onlyif_distributed
def test_write_only_on_master(vstate, tmp_path):
    # Check that the logger runs everywhere but serializes only on rank 0
    # skip test if tensorboardX not installed
    pytest.importorskip("tensorboardX")

    rank = jax.process_index()

    path = str(tmp_path) + "/dir1/r{rank}"

    log = nk.logging.TensorBoardLog(path)

    for i in range(10):
        log(i, {"Energy": jnp.array(1.0), "complex": jnp.array(1.0 + 1j)}, vstate)

    log.flush()
    del log

    files = glob.glob(path + "/*")
    if rank == 0:
        assert len(files) >= 1
    else:
        assert len(files) == 0
