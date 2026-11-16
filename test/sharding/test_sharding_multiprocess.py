"""Multi-process tests: 2 ``jax.distributed`` processes with 2 CPU devices each.

They cover what a single process with several devices cannot: collectives across
processes (a rank that skips one deadlocks the others), arrays that are not fully
addressable, and I/O that must happen on rank 0 only.

All scenarios run in a single launch (the module fixture ``results``), each one
writing to its own directory; the tests then check what every rank returned and
wrote. A deadlock in any scenario surfaces as a timeout of the fixture.

Nothing here may import netket at module level: the spawned processes re-import
this module, and ``jax.distributed`` must be initialised before netket is imported.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ..multiprocess import run_distributed

N_PROCESSES = 2
DEVICES_PER_PROCESS = 2
N_STEPS = 4


def _setup(L=8):
    import jax
    import netket as nk

    g = nk.graph.Chain(L)
    hi = nk.hilbert.Spin(s=1 / 2, N=L)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=4 * jax.device_count())
    vs = nk.vqs.MCState(
        sa, nk.models.RBM(alpha=1, param_dtype=complex), n_samples=256, seed=0
    )
    ha = nk.operator.IsingJax(hilbert=hi, graph=g, h=1.0)
    return vs, ha


def _expect_and_grad():
    import jax

    vs, ha = _setup()
    assert not vs.samples.is_fully_addressable
    E, grad = vs.expect_and_grad(ha)
    grad = np.concatenate([np.ravel(x) for x in jax.tree.leaves(grad)])
    return {"energy": complex(E.mean), "grad": grad}


def _drivers(out: Path):
    """Runs VMC, VMC with SR and TDVP with the loggers attached, writing to ``out``,
    a directory of this rank."""
    import netket as nk
    import netket.experimental as nkx

    vs, ha = _setup()
    loggers = [
        nk.logging.JsonLog(str(out / "json"), save_params_every=1, write_every=1),
        nk.logging.HDF5Log(str(out / "log.h5")),
        nk.logging.StateLog(str(out / "state"), save_every=1),
    ]
    runtime = nk.logging.RuntimeLog()
    gs = nk.driver.VMC(ha, nk.optimizer.Sgd(0.01), variational_state=vs)
    gs.run(N_STEPS, out=[*loggers, runtime])
    # Reusing a logger across runs appends to it.
    gs.run(N_STEPS, out=loggers[1])
    runtime.serialize(str(out / "runtime"))

    sr = nk.driver.VMC_SR(
        ha, nk.optimizer.Sgd(0.01), variational_state=vs, diag_shift=0.1
    )
    sr.run(N_STEPS)

    te = nkx.TDVP(
        ha,
        variational_state=vs,
        ode_solver=nkx.dynamics.Euler(dt=0.001),
        qgt=nk.optimizer.qgt.QGTOnTheFly(holomorphic=True, diag_shift=1e-4),
    )
    te.run(T=0.002, show_progress=False)
    return {"energy": complex(vs.expect(ha).mean)}


def _serialization():
    import jax
    import flax.serialization

    vs, ha = _setup()
    vs.sample()
    data = flax.serialization.to_bytes(vs)
    vs2 = flax.serialization.from_bytes(vs, data)
    pars = [np.asarray(x) for x in jax.tree.leaves(vs.parameters)]
    pars2 = [np.asarray(x) for x in jax.tree.leaves(vs2.parameters)]
    for a, b in zip(pars, pars2):
        np.testing.assert_array_equal(a, b)
    return {"energy": complex(vs2.expect(ha).mean)}


def _all_scenarios(rank: int, tmpdir: str) -> dict:
    import jax

    out = Path(tmpdir) / f"r{rank}"
    out.mkdir()
    return {
        "topology": (jax.process_count(), jax.device_count(), jax.local_device_count()),
        "expect_and_grad": _expect_and_grad(),
        "drivers": _drivers(out),
        "serialization": _serialization(),
    }


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("multiprocess")
    res = run_distributed(
        _all_scenarios,
        str(tmpdir),
        nproc=N_PROCESSES,
        devices_per_process=DEVICES_PER_PROCESS,
    )
    return tmpdir, res


def test_topology(results):
    _, res = results
    n_devices = N_PROCESSES * DEVICES_PER_PROCESS
    assert [r["topology"] for r in res] == [
        (N_PROCESSES, n_devices, DEVICES_PER_PROCESS)
    ] * N_PROCESSES


@pytest.mark.parametrize("scenario", ["expect_and_grad", "drivers", "serialization"])
def test_ranks_agree(results, scenario):
    # Every rank sees the same, globally reduced, quantities.
    _, res = results
    r0, r1 = (r[scenario] for r in res)
    for k in r0:
        np.testing.assert_allclose(r0[k], r1[k])


def test_loggers_write_only_on_rank_0(results):
    tmpdir, _ = results
    assert list((tmpdir / "r1").iterdir()) == []

    out = tmpdir / "r0"
    with open(out / "json.log") as f:
        assert len(json.load(f)["Energy"]["iters"]) == N_STEPS
    assert (out / "json.mpack").exists()
    assert len(list((out / "state").iterdir())) >= N_STEPS
    with open(out / "runtime.json") as f:
        assert len(json.load(f)["Energy"]["iters"]) == N_STEPS

    h5py = pytest.importorskip("h5py")
    with h5py.File(out / "log.h5", "r") as f:
        assert f["data/Energy/Mean"].shape[0] == 2 * N_STEPS
