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
import copy
import scipy

from flax import serialization

import netket as nk
import netket.experimental as nkx

from .. import common

pytestmark = common.skipif_distributed

SEED = 214748364


def _setup_system(L, *, model=None, dtype=np.complex128):
    g = nk.graph.Chain(length=L)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    if model is None:
        model = nk.models.RBM(alpha=1, param_dtype=dtype)

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


fixed_step_solvers = [
    pytest.param(nkx.dynamics.Euler(dt=0.01), id="Euler(dt=0.01)"),
    pytest.param(nkx.dynamics.Heun(dt=0.01), id="Heun(dt=0.01)"),
]
adaptive_step_solvers = [
    pytest.param(
        nkx.dynamics.RK23(dt=0.01, adaptive=True, rtol=1e-2, atol=1e-2),
        id="RK23(dt=0.01, adaptive)",
    ),
    pytest.param(
        nkx.dynamics.RK45(dt=0.01, adaptive=True, rtol=1e-3, atol=1e-3),
        id="RK45(dt=0.01, adaptive)",
    ),
]
all_solvers = fixed_step_solvers + adaptive_step_solvers

nqs_models = [
    pytest.param(
        nk.models.RBM(alpha=1, param_dtype=np.complex128), id="RBM(complex128)"
    ),
]


@pytest.mark.parametrize("model", nqs_models)
@pytest.mark.parametrize("solver", fixed_step_solvers)
@pytest.mark.parametrize("propagation_type", ["real", "imag"])
def test_one_fixed_step(model, solver, propagation_type):
    ha, vstate, _ = _setup_system(L=2, model=model)
    holomorphic = jnp.issubdtype(vstate.model.param_dtype, jnp.complexfloating)
    te = nkx.TDVP(
        ha,
        vstate,
        solver,
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=holomorphic),
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
        jax.tree_util.tree_map(lambda x: jnp.sum(jnp.abs(x) ** 4), x),
    ) ** (1.0 / 4.0)


@common.skipif_distributed
@pytest.mark.parametrize("error_norm", ["euclidean", "qgt", "maximum", l4_norm])
@pytest.mark.parametrize("solver", adaptive_step_solvers)
@pytest.mark.parametrize("propagation_type", ["real", "imag"])
def test_one_adaptive_step(solver, error_norm, propagation_type):
    ha, vstate, _ = _setup_system(L=2)
    te = nkx.TDVP(
        ha,
        vstate,
        solver,
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True),
        propagation_type=propagation_type,
        error_norm=error_norm,
    )
    te.run(T=0.01, callback=_stop_after_one_step)
    assert te.t > 0.0


@pytest.mark.slow
@pytest.mark.parametrize("error_norm", ["euclidean", "qgt", "maximum", l4_norm])
@pytest.mark.parametrize("solver", adaptive_step_solvers)
def test_one_adaptive_schmitt(solver, error_norm):
    ha, vstate, _ = _setup_system(L=2)
    te = nkx.driver.TDVPSchmitt(
        ha,
        vstate,
        solver,
        error_norm=error_norm,
    )
    te.run(T=0.01, callback=_stop_after_one_step)
    assert te.t > 0.0


@pytest.mark.parametrize("solver", all_solvers)
def test_one_step_lindbladian(solver):
    def _setup_lindbladian_system():
        L = 3
        hi = nk.hilbert.Spin(s=0.5) ** L
        ha = nk.operator.LocalOperatorNumba(hi)
        j_ops = []
        for i in range(L):
            ha += (0.3 / 2.0) * nk.operator.spin.sigmax(hi, i)
            ha += (
                (2.0 / 4.0)
                * nk.operator.spin.sigmaz(hi, i)
                @ nk.operator.spin.sigmaz(hi, (i + 1) % L)
            )
            j_ops.append(nk.operator.spin.sigmam(hi, i))
        # Create the liouvillian
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
        solver,
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
    with pytest.warns(UserWarning, match="ODE integrator: dt reached lower bound"):
        te.run(T=0.1, callback=_stop_after_one_step)


@pytest.mark.parametrize("solver", all_solvers)
def test_stop_times(solver):
    def make_driver():
        ha, vstate, _ = _setup_system(L=2)
        return nkx.TDVP(
            ha,
            vstate,
            solver,
            propagation_type="imag",
        )

    def run(driver, T, tstops=None):
        ts = []

        def callback(t, log_data, driver):
            assert t == driver.t
            assert t == driver.step_value
            ts.append((t, driver.step_count))
            return True

        driver.run(T, tstops=tstops, callback=callback, show_progress=False)
        return ts

    driver = make_driver()
    ts = run(driver, 0.1)
    assert [i for _, i in ts] == list(range(len(ts)))
    ts = [t for t, _ in ts]
    if driver.integrator.use_adaptive:
        assert np.all(np.greater_equal(ts, 0.0))
        assert np.all(np.less_equal(ts, 0.1))
    else:
        np.testing.assert_allclose(ts, np.linspace(0.0, 0.1, 11))

    driver = make_driver()
    tstops = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    ts = run(driver, 0.1, tstops)
    np.testing.assert_allclose([t for t, _ in ts], tstops)

    with pytest.raises(ValueError, match="All tstops must be in range"):
        driver.run(0.1, tstops=tstops)
    with pytest.raises(ValueError, match="All tstops must be in range"):
        driver.run(0.1, tstops=[42.0])

    driver = make_driver()
    tstops = [0.012, 0.014, 0.016, 0.018, 0.020]
    ts = run(driver, 0.03, tstops)
    np.testing.assert_allclose([t for t, _ in ts], tstops)


def test_repr_and_info():
    ha, vstate, _ = _setup_system(L=2)
    driver = nkx.TDVP(
        ha,
        vstate,
        nkx.dynamics.RK23(dt=0.01),
        qgt=nk.optimizer.qgt.QGTOnTheFly(holomorphic=True),
    )
    assert "TDVP" in str(driver)


def test_run_twice():
    # 1100
    ha, vstate, _ = _setup_system(L=2)
    driver = nkx.TDVP(
        ha,
        vstate,
        nkx.dynamics.RK23(dt=0.01),
        qgt=nk.optimizer.qgt.QGTOnTheFly(holomorphic=True),
    )
    driver.run(0.03)
    driver.run(0.03)
    np.testing.assert_allclose(driver.t, 0.06)


@pytest.mark.parametrize("t0", [-0.05, 0.0, 0.05])
def test_pbar_starts_zero_ends_hundred(t0):
    """Regression test: progress bar should display 0%→100% for any starting time t0.

    With the old code, `total` was set to `t0 + T` (absolute end time) and `pbar.n`
    tracked absolute integrator time, so the bar would start mid-way or with a
    negative total when t0 != 0.

    The regression is visible in the rendered output:
    - t0 < 0: negative percentages appear in the middle of the run.
    - t0 > 0: the bar jumps immediately from 0% to >20%, skipping the first steps.
    """
    import io
    import re
    from unittest.mock import patch
    from tqdm.auto import tqdm as real_tqdm

    dt = 0.01
    T = 0.1
    # One step advances the bar by dt/T * 100 = 10%; allow 2x tolerance.
    max_first_step_pct = 2 * dt / T * 100

    ha, vstate, _ = _setup_system(L=2)
    buf = io.StringIO()

    class BufferedTqdm(real_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = buf
            kwargs["dynamic_ncols"] = False
            kwargs["ncols"] = 80
            # Redraw on every update so that each step shows up in the output.
            kwargs["mininterval"] = 0
            kwargs["miniters"] = 0
            super().__init__(*args, **kwargs)

    driver = nkx.TDVP(
        ha,
        vstate,
        nkx.dynamics.Euler(dt=dt),
        propagation_type="imag",
        t0=t0,
    )

    with patch("netket._src.callbacks.progressbar.tqdm", BufferedTqdm):
        driver.run(T=T, show_progress=True)

    lines = [
        l.strip() for l in buf.getvalue().replace("\r", "\n").split("\n") if l.strip()
    ]
    assert lines, "No progress bar output was captured"

    percentages = [int(m.group(1)) for l in lines if (m := re.match(r"(-?\d+)%", l))]
    assert percentages, "Could not parse any percentages from progress bar output"

    # Deduplicate while preserving order to get the sequence of distinct states.
    unique_pcts = list(dict.fromkeys(percentages))

    assert unique_pcts[0] == 0, f"Bar should start at 0%, got {unique_pcts[0]}%"
    assert unique_pcts[-1] == 100, f"Bar should end at 100%, got {unique_pcts[-1]}%"
    assert all(p >= 0 for p in unique_pcts), (
        f"Negative percentages in progress bar: {unique_pcts}"
    )
    assert unique_pcts[1] <= max_first_step_pct, (
        f"Bar jumped from 0% to {unique_pcts[1]}% on first update "
        f"(expected ≤ {max_first_step_pct:.0f}% for one step)"
    )


def test_change_solver():
    ha, vstate, _ = _setup_system(L=2)
    driver = nkx.TDVP(
        ha,
        vstate,
        nkx.dynamics.RK23(dt=0.01, adaptive=False),
    )
    driver.run(0.03)
    np.testing.assert_allclose(driver.t, 0.03)

    solver = nkx.dynamics.Euler(dt=0.05)
    driver.ode_solver = solver
    assert driver.ode_solver is solver
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


class _RecordSteps(nk.callbacks.AbstractCallback):
    steps: list = nk.utils.struct.field(pytree_node=False, serialize=False)

    def __init__(self):
        self.steps = []

    def on_step_end(self, step, log_data, driver):
        self.steps.append(step)


@pytest.mark.parametrize("tstops", [None, [0.0, 0.02, 0.05]])
def test_run_logs_at_stopping_times(tstops, capsys):
    ha, vstate, obs = _setup_system(L=2)
    driver = nkx.TDVP(ha, vstate, nkx.dynamics.Euler(dt=0.01), propagation_type="imag")

    log = nk.logging.RuntimeLog()
    times = []
    recorder = _RecordSteps()

    def callback(t, log_data, driver):
        assert "sx" in log_data
        times.append(t)
        return True

    driver.run(0.05, out=log, obs=obs, tstops=tstops, callback=[callback, recorder])
    # Reaching the final time is not reported as stopping early.
    assert "Stopping early" not in capsys.readouterr().out

    # Loggers and callback functions are called at the stopping times, with the
    # time, including the start and the end of the run.
    expected = np.linspace(0.0, 0.05, 6) if tstops is None else tstops
    np.testing.assert_allclose(times, expected)
    np.testing.assert_allclose(log.data["sx"].iters, expected)
    np.testing.assert_allclose(log.data["t"].values, expected)
    # The generator is evaluated at the start of each step.
    np.testing.assert_allclose(log.data["Generator"].iters, expected[:-1])

    np.testing.assert_allclose(driver.t, 0.05)
    assert driver.step_count == 5
    # Other callbacks are called at every step, with the step count, and once
    # more at the end of the run.
    assert recorder.steps == list(range(6))


def test_observables_at_current_parameters():
    # With a multi-stage solver, the parameters are moved to the intermediate
    # stages during a step: the observables must still use those at time t.
    hi = nk.hilbert.Spin(s=0.5, N=2)
    ha = nk.operator.Ising(hi, graph=nk.graph.Chain(2), h=1.0)
    sx = sum(nk.operator.spin.sigmax(hi, i) for i in range(2))
    vstate = nk.vqs.FullSumState(hi, nk.models.RBM(alpha=1, param_dtype=complex))

    driver = nkx.TDVP(ha, vstate, nkx.dynamics.Heun(dt=0.01))
    log = nk.logging.RuntimeLog()
    driver.run(0.03, out=log, obs={"sx": sx})

    jax.tree.map(
        np.testing.assert_array_equal, driver.state.parameters, driver.integrator.y
    )
    np.testing.assert_allclose(
        log.data["sx"]["Mean"][-1], driver.state.expect(sx).mean, rtol=1e-12
    )


def _make_tdvp(kind, solver):
    ha, vstate, _ = _setup_system(L=2)
    if kind == "TDVP":
        return nkx.TDVP(
            ha, vstate, solver, qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
        )
    return nkx.driver.TDVPSchmitt(ha, vstate, solver, holomorphic=True)


@pytest.mark.parametrize("kind", ["TDVP", "TDVPSchmitt"])
@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(nkx.dynamics.Heun(dt=0.01), id="Heun"),
        pytest.param(
            nkx.dynamics.RK23(dt=0.01, adaptive=True, rtol=1e-3, atol=1e-3),
            id="RK23(adaptive)",
        ),
    ],
)
def test_restore_into_fresh_driver(kind, solver):
    driver = _make_tdvp(kind, solver)
    driver.run(0.02, show_progress=False)
    fresh = _make_tdvp(kind, solver)

    # A new driver saves the same fields, with the same shapes and dtypes, as one
    # that has already run.
    def layout(d):
        state = serialization.to_state_dict(d)
        return jax.tree.map(lambda x: (np.shape(x), np.result_type(x)), state)

    assert layout(fresh) == layout(driver)

    restored = serialization.from_bytes(fresh, serialization.to_bytes(driver))
    assert restored.step_count == driver.step_count
    np.testing.assert_array_equal(restored.t, driver.t)
    np.testing.assert_array_equal(restored.dt, driver.dt)

    # The restored driver continues exactly as the original one.
    driver.run(0.02, show_progress=False)
    restored.run(0.02, show_progress=False)
    assert restored.step_count == driver.step_count
    np.testing.assert_array_equal(restored.t, driver.t)
    jax.tree.map(
        np.testing.assert_array_equal,
        restored.state.parameters,
        driver.state.parameters,
    )


class _RestoreOnRunStart(nk.callbacks.AbstractCallback):
    """Restores a saved driver on run start, like a checkpointer resuming a run."""

    data: bytes = nk.utils.struct.field(pytree_node=False, serialize=False)

    def __init__(self, data):
        self.data = data

    def on_run_start(self, step, driver):
        restored = serialization.from_bytes(driver, self.data)
        driver._step_count = restored._step_count
        driver._integrator = restored._integrator
        driver.state.parameters = restored.state.parameters


def test_run_end_ignores_time_restored_on_run_start():
    solver = nkx.dynamics.Euler(dt=0.01)
    saved = _make_tdvp("TDVP", solver)
    saved.run(0.03, show_progress=False)
    data = serialization.to_bytes(saved)

    driver = _make_tdvp("TDVP", solver)
    log = nk.logging.RuntimeLog()
    tstops = np.linspace(0.0, 0.05, 6)
    driver.run(0.05, out=log, tstops=tstops, callback=_RestoreOnRunStart(data))

    # The run still ends at t=0.05, and only logs the stopping times after the
    # restored one.
    np.testing.assert_allclose(driver.t, 0.05)
    assert driver.step_count == 5
    np.testing.assert_allclose(log.data["t"].iters, tstops[3:])


def exact_time_evolution(H, psi0, T, dt, obs):
    # expm prefers csc format
    H_matrix = H.to_sparse().tocsc()
    initial_state = psi0
    times = np.linspace(0, T, int(T / dt) + 1)
    expectations = {name: [] for name in obs}

    # Precompute the dense matrices for the observables
    obs_matrices = {name: op.to_dense() for name, op in obs.items()}

    for t in times:
        U = scipy.sparse.linalg.expm(-1j * H_matrix * t)
        psi_t = U @ initial_state
        for name, op_matrix in obs_matrices.items():
            exp_t = np.vdot(psi_t, op_matrix @ psi_t).real
            expectations[name].append(exp_t)

    return expectations


# This test verifies a case where SNR = Rho = 0 which used to give NaNs in TDVP Schmitt but not standard TDVP.
# See bug report https://github.com/orgs/netket/discussions/1959 and PR to fix it
# https://github.com/netket/netket/pull/1960
@pytest.mark.slow
def test_tdvp_drivers():
    """Test time evolution comparing TDVP methods against exact evolution for a mean-field"""
    L = 2
    total_time = 0.2
    dt = 0.001

    hi = nk.hilbert.Spin(0.5, L)
    h = 1.0
    J = 1.0
    h_eff = h + J

    H1 = nk.operator.LocalOperator(hi, dtype=np.complex128)
    for i in range(L):
        H1 -= h_eff * nk.operator.spin.sigmaz(hi, i)

    modelExact = nk.models.LogStateVector(hi)
    sa = nk.sampler.MetropolisLocal(hilbert=hi)

    vs_schmitt = nk.vqs.MCState(
        model=modelExact,
        sampler=sa,
        n_samples=2**10,
        seed=214748364,
    )
    vs_tdvp = copy.copy(vs_schmitt)
    vs_exact = copy.copy(vs_schmitt).to_array()

    obs = {
        "sum_sx": sum(nk.operator.spin.sigmax(hi, i) for i in range(L)),
        "sum_sy": sum(nk.operator.spin.sigmay(hi, i) for i in range(L)),
    }

    integrator = nkx.dynamics.RK4(dt=dt)

    # Exact time evolution
    expectations = exact_time_evolution(H1, vs_exact, total_time, dt, obs)
    sx_exact = expectations["sum_sx"]
    sy_exact = expectations["sum_sy"]

    # TDVPSchmitt time evolution
    te_schmitt = nkx.driver.TDVPSchmitt(H1, vs_schmitt, integrator, holomorphic=True)
    log_schmitt = nk.logging.RuntimeLog()
    te_schmitt.run(T=total_time, out=log_schmitt, obs=obs)

    # TDVP time evolution
    te_tdvp = nkx.driver.TDVP(H1, vs_tdvp, integrator)
    log_tdvp = nk.logging.RuntimeLog()
    te_tdvp.run(T=total_time, out=log_tdvp, obs=obs)

    sx_schmitt = np.array(log_schmitt.data["sum_sx"]).real
    sy_schmitt = np.array(log_schmitt.data["sum_sy"]).real

    np.testing.assert_allclose(sx_schmitt, sx_exact)
    np.testing.assert_allclose(sy_schmitt, sy_exact)

    sx_tdvp = np.array(log_tdvp.data["sum_sx"]).real
    sy_tdvp = np.array(log_tdvp.data["sum_sy"]).real

    np.testing.assert_allclose(sx_tdvp, sx_exact)
    np.testing.assert_allclose(sy_tdvp, sy_exact)


def test_float32_dtype():
    # Issue https://github.com/netket/netket/issues/1916
    # Type stability in KahnSummator and norm

    import netket as nk
    from netket import experimental

    import jax.numpy as jnp

    N = 3
    hi = nk.hilbert.Spin(0.5, N)
    ha = sum(nk.operator.spin.sigmax(hi, i) for i in range(N))

    sa = nk.sampler.MetropolisLocal(hi)
    model = nk.models.RBM(alpha=1, param_dtype=jnp.float32)
    vstate = nk.vqs.MCState(sa, model, n_samples=1008, n_discard_per_chain=16)
    solver = experimental.dynamics.RK12(dt=1e-2, adaptive=True, rtol=1e-5)
    tdvp = experimental.TDVP(ha, vstate, solver, error_norm="qgt")

    tdvp_log = nk.logging.RuntimeLog()
    tdvp.run(T=0.1, out=tdvp_log)
