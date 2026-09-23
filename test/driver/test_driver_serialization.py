# Copyright 2026 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import numpy as np
import pytest
from flax import serialization

import netket as nk

from .. import common

pytestmark = common.skipif_distributed

SR_MOMENTUM = dict(momentum=0.9, use_ntk=True, on_the_fly=True)
SR_DENSE_MOMENTUM = dict(momentum=0.9, use_ntk=True, on_the_fly=False)


def _make_driver(kind, param_dtype):
    hi = nk.hilbert.Spin(0.5, 6)
    H = nk.operator.IsingJax(hi, nk.graph.Chain(6), h=1.0)

    def make_state(seed):
        return nk.vqs.MCState(
            nk.sampler.MetropolisLocal(hi, n_chains=16),
            nk.models.RBM(alpha=1, param_dtype=param_dtype),
            n_samples=256,
            seed=seed,
            sampler_seed=seed + 1,
        )

    vs = make_state(0)
    opt = nk.optimizer.Sgd(0.01)
    if kind == "VMC":
        return nk.driver.VMC(
            H, opt, variational_state=vs, preconditioner=nk.optimizer.SR()
        )
    elif kind == "VMC_SR":
        return nk.driver.VMC_SR(H, opt, variational_state=vs, diag_shift=0.1)
    elif kind == "VMC_SR+momentum":
        return nk.driver.VMC_SR(
            H, opt, variational_state=vs, diag_shift=0.1, **SR_MOMENTUM
        )
    elif kind == "VMC_SR+dense_momentum":
        return nk.driver.VMC_SR(
            H, opt, variational_state=vs, diag_shift=0.1, **SR_DENSE_MOMENTUM
        )
    elif kind == "SteadyState":
        lind = nk.operator.LocalLiouvillian(H.to_local_operator(), [])
        mixed = nk.vqs.MCMixedState(
            nk.sampler.MetropolisLocal(nk.hilbert.DoubledHilbert(hi), n_chains=16),
            nk.models.NDM(param_dtype=param_dtype),
            sampler_diag=nk.sampler.MetropolisLocal(hi, n_chains=16),
            n_samples=256,
            seed=0,
        )
        return nk.SteadyState(lind, opt, variational_state=mixed)
    elif kind == "Infidelity_SR+momentum":
        return nk.driver.Infidelity_SR(
            target_state=make_state(10),
            optimizer=opt,
            variational_state=vs,
            diag_shift=0.1,
            **SR_MOMENTUM,
        )
    raise ValueError(kind)


@pytest.mark.parametrize("param_dtype", [float, complex])
@pytest.mark.parametrize(
    "kind",
    [
        "VMC",
        "VMC_SR",
        "VMC_SR+momentum",
        "VMC_SR+dense_momentum",
        "Infidelity_SR+momentum",
        "SteadyState",
    ],
)
def test_restore_into_fresh_driver(kind, param_dtype):
    driver = _make_driver(kind, param_dtype)
    driver.run(2, out=None, show_progress=False)
    fresh = _make_driver(kind, param_dtype)

    # A new driver saves the same fields, with the same shapes and dtypes, as one
    # that has already run. Checkpoint libraries that restore into a template
    # object (e.g. orbax) rely on this.
    def layout(d):
        state = serialization.to_state_dict(d)
        return jax.tree.map(lambda x: (np.shape(x), np.result_type(x)), state)

    assert layout(fresh) == layout(driver)

    restored = serialization.from_bytes(fresh, serialization.to_bytes(driver))
    assert restored.step_count == driver.step_count
    restored.run(1, out=None, show_progress=False)


def test_restore_file_from_older_version():
    # Older versions also saved values that are recomputed at every step.
    driver = _make_driver("VMC_SR", float)
    driver.run(2, out=None, show_progress=False)
    state = serialization.to_state_dict(driver)
    state["_loss_stats"] = serialization.to_state_dict(driver._loss_stats)
    state["info"] = None

    restored = serialization.from_state_dict(_make_driver("VMC_SR", float), state)
    assert restored.step_count == driver.step_count


def test_change_n_samples_before_run():
    driver = _make_driver("VMC_SR", float)
    driver.state.n_samples = 512
    driver.state.sampler = driver.state.sampler.replace(n_chains=32)
    driver.run(2, out=None, show_progress=False)
    assert driver._loss_stats_online.n_chains == 32
