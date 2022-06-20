import pytest

import glob

import numpy as np
import netket as nk
import netket.experimental as nkx
from jax.nn.initializers import normal
from jax import numpy as jnp

import orjson

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


def test_serialize(vstate, tmp_path):
    log = nk.logging.RuntimeLog()
    e = vstate.expect(nk.operator.spin.sigmax(vstate.hilbert, 0))

    for i in np.arange(0, 1, 0.1):
        log(
            i,
            {
                "energy": e,
                "vals": {
                    "energy": e,
                    "random": 1.0,
                    "matrix": jnp.array(np.random.rand(3)),
                },
            },
            None,
        )

    log.serialize(tmp_path / "file_1")
    filename = tmp_path / "file_1.json"
    with open(filename, "rb") as f:
        data1 = orjson.loads(f.read())

    log.serialize(str(tmp_path) + "/file_2.log")
    filename = tmp_path / "file_2.log"
    with open(filename, "rb") as f:
        data2 = orjson.loads(f.read())

    filename = tmp_path / "file_3"
    with open(filename, "wb") as f:
        log.serialize(f)

    with open(filename, "rb") as f:
        data3 = orjson.loads(f.read())

    assert data1 == data2
    assert data2 == data3

    assert "energy" in data1
    assert "vals" in data1
    assert "energy" in data1["vals"]
    assert "random" in data1["vals"]
    assert "value" in data1["vals"]["random"]
    assert "iters" in data1["vals"]["random"]

    assert len(data1["vals"]["random"]["value"]) == 10
    assert len(data1["vals"]["random"]["iters"]) == 10

    assert np.array(data1["vals"]["matrix"]["value"]).shape == (10, 3)

    assert repr(log).startswith("RuntimeLog")
