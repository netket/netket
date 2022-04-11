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

from functools import partial

import jax
import jax.numpy as jnp
import pytest
import numpy as np

import netket as nk
import netket.experimental as nkx


SEED = 214748364


def _setup_system(L, *, model=None, dtype=np.complex128):
    g = nk.graph.Chain(length=L)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    if model is None:
        model = nk.models.RBM(alpha=1, dtype=dtype)

    sa = nk.sampler.ExactSampler(hilbert=hi)

    vs = nk.vqs.MCState(sa, model, n_samples=1000, seed=SEED)

    ha = nk.operator.Ising(hi, graph=g, h=1.0)

    # Add custom observable
    X = [[0, 1], [1, 0]]
    sx = nk.operator.LocalOperator(hi, [X] * L, [[i] for i in range(g.n_nodes)])
    obs = {"sx": sx}

    return ha, vs, obs


def _stop_after_one_step(step, *_):
    """
    Callback to stop the driver after the first (successful) step.
    """
    return step == 0


fixed_step_integrators = [
    pytest.param(nkx.dynamics.Euler(dt=0.01), id="Euler(dt=0.01)"),
    pytest.param(nkx.dynamics.Heun(dt=0.01), id="Heun(dt=0.01)"),
]
adaptive_step_integrators = [
    pytest.param(
        nkx.dynamics.RK23(dt=0.01, adaptive=True, rtol=1e-2, atol=1e-2),
        id="RK23(dt=0.01, adaptive)",
    ),
    pytest.param(
        nkx.dynamics.RK45(dt=0.01, adaptive=True, rtol=1e-3, atol=1e-3),
        id="RK45(dt=0.01, adaptive)",
    ),
]
all_integrators = fixed_step_integrators + adaptive_step_integrators

nqs_models = [
    pytest.param(nk.models.RBM(alpha=1, dtype=np.complex128), id="RBM(complex128)"),
    pytest.param(
        nk.models.RBMModPhase(alpha=1, dtype=np.float64), id="RBMModPhase(float64)"
    ),
]


@pytest.mark.parametrize("model", nqs_models)
@pytest.mark.parametrize("integrator", fixed_step_integrators)
@pytest.mark.parametrize("propagation_type", ["real", "imag"])
def test_one_fixed_step(model, integrator, propagation_type):
    ha, vstate, _ = _setup_system(L=2, model=model)
    te = nkx.TDVP(
        ha,
        vstate,
        integrator,
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True),
        propagation_type=propagation_type,
    )
    te.run(T=0.01, callback=_stop_after_one_step)
    assert te.t == 0.01


def l4_norm(x):
    """
    Custom L4 error norm.
    """
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(lambda x: jnp.sum(jnp.abs(x) ** 4), x),
    ) ** (1.0 / 4.0)


@pytest.mark.parametrize("error_norm", ["euclidean", "qgt", "maximum", l4_norm])
@pytest.mark.parametrize("integrator", adaptive_step_integrators)
@pytest.mark.parametrize("propagation_type", ["real", "imag"])
def test_one_adaptive_step(integrator, error_norm, propagation_type):
    ha, vstate, _ = _setup_system(L=2)
    te = nkx.TDVP(
        ha,
        vstate,
        integrator,
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True),
        propagation_type=propagation_type,
        error_norm=error_norm,
    )
    te.run(T=0.01, callback=_stop_after_one_step)
    assert te.t > 0.0


@pytest.mark.parametrize("integrator", all_integrators)
def test_one_step_lindbladian(integrator):
    def _setup_lindbladian_system():
        L = 3
        hi = nk.hilbert.Spin(s=0.5) ** L
        ha = nk.operator.LocalOperator(hi)
        j_ops = []
        for i in range(L):
            ha += (0.3 / 2.0) * nk.operator.spin.sigmax(hi, i)
            ha += (
                (2.0 / 4.0)
                * nk.operator.spin.sigmaz(hi, i)
                * nk.operator.spin.sigmaz(hi, (i + 1) % L)
            )
            j_ops.append(nk.operator.spin.sigmam(hi, i))
        #  Create the liouvillian
        lind = nk.operator.LocalLiouvillian(ha, j_ops)

        # Create NDM and vstate
        ma = nk.models.NDM()
        sa = nk.sampler.MetropolisLocal(hilbert=nk.hilbert.DoubledHilbert(hi))
        sa_obs = nk.sampler.MetropolisLocal(hilbert=hi)
        vstate = nk.vqs.MCMixedState(
            sa, ma, sampler_diag=sa_obs, n_samples=1000, seed=SEED
        )

        return lind, vstate

    lind, vstate = _setup_lindbladian_system()
    te = nkx.TDVP(
        lind,
        vstate,
        integrator,
        propagation_type="real",
        linear_solver=partial(nk.optimizer.solver.svd, rcond=1e-3),
    )
    te.run(T=0.01, callback=_stop_after_one_step)
    assert te.t > 0.0


def test_dt_bounds():
    ha, vstate, _ = _setup_system(L=2, dtype=np.complex128)
    te = nkx.TDVP(
        ha,
        vstate,
        nkx.dynamics.RK23(dt=0.1, adaptive=True, dt_limits=(1e-2, None)),
        propagation_type="real",
    )
    with pytest.warns(UserWarning, match="RK solver: dt reached lower bound"):
        te.run(T=0.1, callback=_stop_after_one_step)


@pytest.mark.parametrize("integrator", all_integrators)
def test_stop_times(integrator):
    def make_driver():
        ha, vstate, _ = _setup_system(L=2)
        return nkx.TDVP(
            ha,
            vstate,
            integrator,
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


def test_repr_and_info():
    ha, vstate, _ = _setup_system(L=2)
    driver = nkx.TDVP(
        ha,
        vstate,
        nkx.dynamics.RK23(dt=0.01),
    )
    print(str(driver))
    assert "TDVP" in str(driver)

    info = driver.info()
    print(info)
    assert "TDVP" in info
    assert "generator" in info
    assert "integrator" in info
    assert "RK23" in info


def test_run_twice():
    # 1100
    ha, vstate, _ = _setup_system(L=2)
    driver = nkx.TDVP(
        ha,
        vstate,
        nkx.dynamics.RK23(dt=0.01),
    )
    driver.run(0.03)
    driver.run(0.03)
    np.testing.assert_allclose(driver.t, 0.06)


def test_change_integrator():
    ha, vstate, _ = _setup_system(L=2)
    driver = nkx.TDVP(
        ha,
        vstate,
        nkx.dynamics.RK23(dt=0.01, adaptive=False),
    )
    driver.run(0.03)
    np.testing.assert_allclose(driver.t, 0.03)

    integrator = nkx.dynamics.Euler(dt=0.05)
    driver.integrator = integrator
    np.testing.assert_allclose(driver.t, 0.03)
    np.testing.assert_allclose(driver.dt, 0.05)

    driver.run(0.1)
    np.testing.assert_allclose(driver.t, 0.13)


def test_change_norm():
    ha, vstate, _ = _setup_system(L=2)
    driver = nkx.TDVP(
        ha,
        vstate,
        nkx.dynamics.RK23(dt=0.01, adaptive=False),
    )
    driver.run(0.03)

    def norm(x):
        return 0.0

    driver.error_norm = norm
    driver.run(0.02)

    driver.error_norm = "qgt"
    driver.run(0.02)

    driver.error_norm = "maximum"
    driver.run(0.02)

    with pytest.raises(ValueError):
        driver.error_norm = "assd"
