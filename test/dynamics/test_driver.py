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
    pytest.param(nkx.dynamics.Euler(dt=0.01), id="Euler(dt=0.01)"),
    pytest.param(nkx.dynamics.Heun(dt=0.01), id="Heun(dt=0.01)"),
    pytest.param(
        nkx.dynamics.RK23(dt=0.01, adaptive=True, rtol=1e-2),
        id="RK23(dt=0.01, adaptive=True)",
    ),
    pytest.param(
        nkx.dynamics.RK45(dt=0.01, adaptive=True, rtol=1e-3),
        id="RK45(dt=0.01, adaptive=True)",
    ),
]


@pytest.mark.parametrize("integrator", integrator_params)
def test_stop_times(integrator):
    def make_driver():
        ha, vstate, _ = _setup_system(L=4)
        return nkx.TimeDependentVMC(
            ha,
            vstate,
            integrator,
            qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True),
            propagation_type="imag",
        )

    driver = make_driver()
    ts = []
    for i, t in enumerate(driver.iter(T=0.1)):
        assert t == driver.t
        assert t == driver.step_value
        assert i == driver.step_count
        ts.append(t)
    if driver.integrator.use_adaptive:
        assert np.all(np.greater_equal(ts, 0.0))
        assert np.all(np.less_equal(ts, 0.1))
    else:
        np.testing.assert_allclose(ts, np.linspace(0.0, 0.1, 11))

    driver = make_driver()
    tstops = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    ts = []
    for i, t in enumerate(driver.iter(T=0.1, tstops=tstops)):
        assert t == driver.t
        assert t == driver.step_value
        ts.append(t)
    np.testing.assert_allclose(ts, tstops)

    with pytest.raises(ValueError, match="All tstops must be in range"):
        list(driver.iter(T=0.1, tstops=tstops))
    with pytest.raises(ValueError, match="All tstops must be in range"):
        list(driver.iter(T=0.1, tstops=[42.0]))

    driver = make_driver()
    tstops = [0.012, 0.014, 0.016, 0.018, 0.020]
    ts = []
    for i, t in enumerate(driver.iter(T=0.03, tstops=tstops)):
        assert t == driver.t
        assert t == driver.step_value
        ts.append(t)
    np.testing.assert_allclose(ts, tstops)
