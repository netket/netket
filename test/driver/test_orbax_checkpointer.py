import signal

import jax
import numpy as np
import pytest

import netket as nk
from netket.callbacks import AbstractCallback
from netket.utils import struct

from .orbax_checkpointer import Checkpointer


def _vmc():
    hi = nk.hilbert.Spin(0.5, 4)
    ha = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=1.0)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=4)
    vs = nk.vqs.MCState(
        sampler, nk.models.RBM(alpha=1), n_samples=64, seed=12, sampler_seed=13
    )
    return nk.driver.VMC(
        ha,
        nk.optimizer.Adam(0.01),
        variational_state=vs,
        callbacks=(nk.callbacks.EarlyStopping(patience=100),),
    )


def _assert_same(got, want):
    for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(x, y)


def _saved_steps(path):
    return sorted(int(p.name) for p in path.iterdir() if p.name.isdigit())


class _SignalAt(AbstractCallback):
    step: int = struct.field(pytree_node=False, serialize=False)

    def __init__(self, step):
        self.step = step

    def on_step_end(self, step, log_data, driver):
        if step == self.step:
            signal.raise_signal(signal.SIGTERM)


def test_saves_every_n_steps_and_keeps_the_last_k(tmp_path):
    driver = _vmc()
    ckpt = Checkpointer(tmp_path, save_every=2, keep_last_k=2)
    driver.run(5, checkpointer=ckpt, show_progress=False)

    assert _saved_steps(tmp_path) == [4, 5]
    checkpoint = Checkpointer(tmp_path).read()
    assert checkpoint.step == 5
    _assert_same(
        checkpoint.state["variational_state"]["variables"], driver.state.variables
    )
    assert checkpoint.state["callbacks"]["EarlyStopping"]["_best_iter"] == (
        driver.callbacks[0]._best_iter
    )


def test_a_preempted_run_resumes_as_if_never_interrupted(tmp_path):
    reference = _vmc()
    reference.run(6, show_progress=False)

    handler = signal.getsignal(signal.SIGTERM)
    preempted = _vmc()
    preempted.run(
        6,
        callback=_SignalAt(2),
        checkpointer=Checkpointer(tmp_path, save_every=100),
        show_progress=False,
    )
    # Stopped and saved after the step of the signal.
    assert preempted.step_count == 3 and _saved_steps(tmp_path) == [3]
    assert signal.getsignal(signal.SIGTERM) is handler

    resumed = _vmc()
    resumed.run(6, checkpointer=Checkpointer(tmp_path), show_progress=False)
    assert resumed.step_count == 6
    _assert_same(resumed.state.variables, reference.state.variables)
    _assert_same(resumed.state.sampler_state.σ, reference.state.sampler_state.σ)
    assert resumed.callbacks[0]._best_val == reference.callbacks[0]._best_val


def test_checkpoints_of_another_run_are_refused(tmp_path):
    _vmc().run(2, checkpointer=Checkpointer(tmp_path), show_progress=False)
    with pytest.raises(ValueError, match="belongs to a run from step 0 to 2"):
        _vmc().run(3, checkpointer=Checkpointer(tmp_path), show_progress=False)
