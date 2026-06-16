import numpy as np
import pytest

import netket as nk

from .. import common


def _vmc():
    hi = nk.hilbert.Spin(0.5, 4)
    ha = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=1.0)
    vs = nk.vqs.FullSumState(hi, nk.models.RBM(alpha=1, param_dtype=float), seed=12)
    return nk.driver.VMC(ha, nk.optimizer.Sgd(0.05), variational_state=vs)


class _LogTuple(nk.callbacks.AbstractCallback):
    """Logs nested tuples: in an HDF5Log, groups without their own ``iter``."""

    def before_parameter_update(self, step, log_data, driver):
        log_data["pair"] = ((float(step), 2.0 * step), -float(step))


def test_runtime_log_drops_the_entries_from_the_restored_step():
    log = nk.logging.RuntimeLog()
    for step in range(6):
        log(step, {"x": float(step), "nested": {"y": 1.0}, "pair": (1.0, 2.0)})

    log.on_run_start(4, None)

    assert list(log["x"].iters) == [0, 1, 2, 3]
    assert list(log["nested"]["y"].iters) == [0, 1, 2, 3]
    assert [list(h.iters) for h in log["pair"]] == [[0, 1, 2, 3]] * 2
    assert log._old_step == 3


@common.skipif_distributed
@pytest.mark.parametrize("fmt", ["json", "hdf5"])
def test_a_resumed_run_does_not_log_its_steps_twice(tmp_path, fmt):
    prefix = str(tmp_path / "log")

    def run(mode, checkpoints):
        if fmt == "json":
            out = nk.logging.JsonLog(prefix, mode=mode, save_params=False)
            callback = None
        else:
            out = nk.logging.HDF5Log(prefix, mode=mode, metadata={"seed": 12})
            callback = _LogTuple()
        checkpointer = common.MemoryCheckpointer(every=2, checkpoints=checkpoints)
        _vmc().run(
            6,
            out=out,
            callback=callback,
            checkpointer=checkpointer,
            show_progress=False,
        )

    def read():
        if fmt == "json":
            energy = nk.logging.RuntimeLog.deserialize(prefix + ".log")["Energy"]
            return np.asarray(energy.iters), np.asarray(energy.Mean), {}
        h5py = pytest.importorskip("h5py")
        with h5py.File(prefix + ".h5", "r") as f:
            rows = {}
            f["data/pair"].visititems(
                lambda name, d: (
                    rows.setdefault(name, len(d))
                    if isinstance(d, h5py.Dataset)
                    else None
                )
            )
            assert f["metadata"].attrs["seed"] == 12
            energy = f["data/Energy"]
            return energy["iter"][:], energy["Mean"][:], rows

    checkpoints = {}
    run("write", checkpoints)
    _, first, _ = read()
    # Killed after step 5, before the checkpoint of step 6: resumed from step 4.
    del checkpoints[6]
    run("append", checkpoints)
    iters, energy, rows = read()

    assert list(iters) == list(range(6))
    np.testing.assert_array_equal(energy, first)
    assert all(n == 6 for n in rows.values()), rows


@common.skipif_distributed
@pytest.mark.parametrize("fmt", ["json", "hdf5"])
def test_a_fresh_run_drops_the_rows_of_a_stale_log(tmp_path, fmt):
    # A stale log from an earlier run in the same output, a checkpointer with
    # nothing to restore (so no on_checkpoint_restore): on_run_start drops them.
    prefix = str(tmp_path / "log")

    def run(n_iter, mode):
        if fmt == "json":
            out = nk.logging.JsonLog(prefix, mode=mode, save_params=False)
        else:
            out = nk.logging.HDF5Log(prefix, mode=mode)
        checkpointer = common.MemoryCheckpointer()
        _vmc().run(n_iter, out=out, checkpointer=checkpointer, show_progress=False)

    def iters():
        if fmt == "json":
            return list(
                nk.logging.RuntimeLog.deserialize(prefix + ".log")["Energy"].iters
            )
        h5py = pytest.importorskip("h5py")
        with h5py.File(prefix + ".h5", "r") as f:
            return list(f["data/Energy/iter"][:])

    run(5, "write")
    run(3, "append")
    assert iters() == [0, 1, 2]
