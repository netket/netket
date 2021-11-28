# Copyright 2021 The NetKet Authors - All Rights Reserved.
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

import pytest
import numpy as np

import netket as nk
import netket.experimental as nkx


SEED = 214748364


def _setup_system(L, *, dtype=np.float64):
    g = nk.graph.Chain(length=L)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ma = nk.models.RBM(alpha=1, dtype=dtype)
    sa = nk.sampler.ExactSampler(hilbert=hi, n_chains=16)

    vs = nk.vqs.MCState(sa, ma, n_samples=1000, seed=SEED)

    ha = nk.operator.Ising(hi, graph=g, h=1.0)

    # Add custom observable
    X = [[0, 1], [1, 0]]
    sx = nk.operator.LocalOperator(hi, [X] * L, [[i] for i in range(g.n_nodes)])
    obs = {"sx": sx}

    return ha, vs, obs


integrator_params = [
    pytest.param(nkx.dynamics.Heun(dt=0.01), id="Heun(dt=0.01)"),
    pytest.param(
        nkx.dynamics.RK23(dt=0.01, adaptive=True, rtol=1e-2),
        id="RK23(dt=0.01, adaptive=True)",
    ),
]


@pytest.mark.parametrize("integrator", integrator_params)
def test_stop_times(integrator):
    ha, vstate, _ = _setup_system(L=4)
    driver = nkx.TimeDependentVMC(
        ha,
        vstate,
        integrator,
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True),
        propagation_type="imag",
    )
    ts = []
    for i, t in enumerate(driver.iter(T=1.0)):
        assert t == driver.t
        assert t == driver.step_value
        assert i == driver.step_count
        ts.append(t)
    if driver.integrator.use_adaptive:
        assert np.all(np.less_equal(ts, 1.0))
        assert np.all(np.greater_equal(ts, 0.0))
    else:
        np.testing.assert_allclose(ts, np.linspace(0.0, 1.0, 101))

    ha, vstate, _ = _setup_system(L=4)
    driver = nkx.TimeDependentVMC(
        ha,
        vstate,
        integrator,
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True),
        propagation_type="imag",
    )
    tstops = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ts = []
    for i, t in enumerate(driver.iter(T=1.0, tstops=tstops)):
        assert t == driver.t
        assert t == driver.step_value
        ts.append(t)
    np.testing.assert_allclose(ts, tstops)

    with pytest.raises(ValueError, match="All tstops must be in range"):
        list(driver.iter(T=1.0, tstops=tstops))
    with pytest.raises(ValueError, match="All tstops must be in range"):
        list(driver.iter(T=1.0, tstops=[42.0]))


def test_run():
    pass
