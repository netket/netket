import pytest

import numpy as np
import orjson

import jax
from jax.nn.initializers import normal
from jax import numpy as jnp

import netket as nk


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
def test_serialize(vstate, tmp_path):
    log = nk.logging.RuntimeLog()
    e = vstate.expect(nk.operator.spin.sigmax(vstate.hilbert, 0))

    steps = np.arange(0, 1, 0.1)
    for i in steps:
        log(
            i,
            {
                "energy": e,
                "vals": {
                    "energy": e,
                    "scalar": 1.0,
                    "scalar_ndarray": jnp.array(1.0),
                    "vector": jnp.array([1.0, 1.0]),
                    "np_vector": np.array([1.0, 1.0]),
                    "matrix": jnp.array([[1.0, 2.0], [2.0, 1.0]]),
                    "complex_scalar": 1.0j,
                    "complex_matrix": jnp.array([[1.0j], [1.0j]]),
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

    assert "scalar" in data1["vals"]
    assert "value" in data1["vals"]["scalar"]
    assert "iters" in data1["vals"]["scalar"]
    assert len(data1["vals"]["scalar"]["value"]) == 10
    assert len(data1["vals"]["scalar"]["iters"]) == 10

    assert "scalar_ndarray" in data1["vals"]
    assert "value" in data1["vals"]["scalar_ndarray"]
    assert "iters" in data1["vals"]["scalar_ndarray"]
    assert len(data1["vals"]["scalar_ndarray"]["value"]) == 10
    assert len(data1["vals"]["scalar_ndarray"]["iters"]) == 10

    assert "vector" in data1["vals"]
    assert np.array(data1["vals"]["vector"]["value"]).shape == (10, 2)

    assert "np_vector" in data1["vals"]
    assert np.array(data1["vals"]["np_vector"]["value"]).shape == (10, 2)

    assert "matrix" in data1["vals"]
    assert np.array(data1["vals"]["matrix"]["value"]).shape == (10, 2, 2)

    assert "complex_scalar" in data1["vals"]
    assert "real" in data1["vals"]["complex_scalar"]["value"]
    assert "imag" in data1["vals"]["complex_scalar"]["value"]

    assert "complex_matrix" in data1["vals"]
    assert "real" in data1["vals"]["complex_matrix"]["value"]
    assert "imag" in data1["vals"]["complex_matrix"]["value"]
    shape = (10, 2, 1)
    assert np.array(data1["vals"]["complex_matrix"]["value"]["real"]).shape == shape
    assert np.array(data1["vals"]["complex_matrix"]["value"]["imag"]).shape == shape

    assert repr(log).startswith("RuntimeLog")

    # check log without state specified.
    log(23, {"energy": e})
    assert len(log.data["energy"]["iters"] == 11)


@common.onlyif_distributed
def test_write_only_on_master(tmp_path):
    # Check that the logger runs everywhere but serializes only on rank 0

    log = nk.logging.RuntimeLog()

    n_steps = 10
    for i in range(n_steps):
        log(
            i,
            {
                "value": float(i),
            },
        )

    assert len(log.data["value"]) == n_steps
    np.testing.assert_allclose(log.data["value"].iters, np.arange(n_steps))
    np.testing.assert_allclose(log.data["value"].value, np.arange(n_steps))

    if nk.config.netket_experimental_sharding:
        rank = jax.process_index()
    else:
        rank = nk.utils.mpi.rank

    tmp_path = tmp_path / f"out_{rank}.log"

    log.serialize(tmp_path)

    if rank == 0:
        assert tmp_path.exists()
    else:
        assert not tmp_path.exists()
