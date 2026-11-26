"""Tests on 2 ``jax.distributed`` processes with 2 CPU devices each.

They cover what a single process with several devices cannot: collectives across
processes (a rank that skips one deadlocks the others), arrays that are not fully
addressable, and I/O that must happen on rank 0 only. Each test runs functions on
the processes of the ``distributed`` fixture (see :mod:`test.multiprocess`) and
checks that all ranks return the same, globally reduced, values.
"""

import json
from pathlib import Path

import flax.serialization
import jax
import numpy as np
import pytest

import netket as nk
import netket.experimental as nkx

N_STEPS = 4


def _setup(L=8):
    hi = nk.hilbert.Spin(s=1 / 2, N=L)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=4 * jax.device_count())
    vs = nk.vqs.MCState(
        sa, nk.models.RBM(alpha=1, param_dtype=complex), n_samples=256, seed=0
    )
    ha = nk.operator.IsingJax(hilbert=hi, graph=nk.graph.Chain(L), h=1.0)
    return vs, ha


def _flatten(tree):
    return np.concatenate([np.ravel(x) for x in jax.tree.leaves(tree)])


def _assert_ranks_agree(results):
    r0, *others = results
    for r in others:
        jax.tree.map(np.testing.assert_allclose, r0, r)


def _expect_and_grad():
    vs, ha = _setup()
    assert not vs.samples.is_fully_addressable
    E, grad = vs.expect_and_grad(ha)
    return complex(E.mean), _flatten(grad)


def test_expect_and_grad(distributed):
    _assert_ranks_agree(distributed.run(_expect_and_grad))


def _vmc_sr_and_tdvp():
    vs, ha = _setup()
    sr = nk.driver.VMC_SR(
        ha, nk.optimizer.Sgd(0.01), variational_state=vs, diag_shift=0.1
    )
    sr.run(N_STEPS, show_progress=False)
    te = nkx.TDVP(
        ha,
        variational_state=vs,
        ode_solver=nkx.dynamics.Euler(dt=0.001),
        qgt=nk.optimizer.qgt.QGTOnTheFly(holomorphic=True, diag_shift=1e-4),
    )
    te.run(T=0.002, show_progress=False)
    return complex(vs.expect(ha).mean), _flatten(vs.parameters)


def test_vmc_sr_and_tdvp(distributed):
    _assert_ranks_agree(distributed.run(_vmc_sr_and_tdvp))


def _serialization_roundtrip():
    vs, ha = _setup()
    vs.sample()
    vs2 = flax.serialization.from_bytes(vs, flax.serialization.to_bytes(vs))
    np.testing.assert_array_equal(_flatten(vs.parameters), _flatten(vs2.parameters))
    return complex(vs2.expect(ha).mean)


def test_serialization(distributed):
    _assert_ranks_agree(distributed.run(_serialization_roundtrip))


def _run_vmc_with_loggers(tmpdir):
    """Writes the logs of every rank to its own directory, ``tmpdir/r{rank}``."""
    out = Path(tmpdir) / f"r{jax.process_index()}"
    out.mkdir()
    vs, ha = _setup()
    loggers = [
        nk.logging.JsonLog(str(out / "json"), save_params_every=1, write_every=1),
        nk.logging.HDF5Log(str(out / "log.h5")),
        nk.logging.StateLog(str(out / "state"), save_every=1),
    ]
    runtime = nk.logging.RuntimeLog()
    gs = nk.driver.VMC(ha, nk.optimizer.Sgd(0.01), variational_state=vs)
    gs.run(N_STEPS, out=[*loggers, runtime], show_progress=False)
    # Reusing a logger across runs appends to it.
    gs.run(N_STEPS, out=loggers[1], show_progress=False)
    runtime.serialize(str(out / "runtime"))


def test_loggers_write_only_on_rank_0(distributed, tmp_path):
    distributed.run(_run_vmc_with_loggers, str(tmp_path))
    assert list((tmp_path / "r1").iterdir()) == []

    out = tmp_path / "r0"
    with open(out / "json.log") as f:
        assert len(json.load(f)["Energy"]["iters"]) == N_STEPS
    assert (out / "json.mpack").exists()
    assert len(list((out / "state").iterdir())) >= N_STEPS
    with open(out / "runtime.json") as f:
        assert len(json.load(f)["Energy"]["iters"]) == N_STEPS

    h5py = pytest.importorskip("h5py")
    with h5py.File(out / "log.h5", "r") as f:
        assert f["data/Energy/Mean"].shape[0] == 2 * N_STEPS
