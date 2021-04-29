import numpy as np
from pytest import approx, raises
from scipy.integrate import solve_ivp
from scipy.linalg import norm

import netket.legacy as nk
from netket.exact import PyExactTimePropagation

import pytest

pytestmark = pytest.mark.legacy

ATOL = 1e-9
RTOL = 1e-9
TIME = 20.0


def _setup_model():
    L = 8
    hi = nk.hilbert.Spin(0.5) ** L
    ham = nk.operator.Heisenberg(hi, nk.graph.Hypercube(L, 1))

    psi0 = np.random.rand(hi.n_states) + 1j * np.random.rand(hi.n_states)
    psi0 /= norm(psi0)

    return ham, psi0


def overlap(phi, psi):
    return np.abs(np.vdot(phi, psi)) ** 2 / (np.vdot(phi, phi) * np.vdot(psi, psi)).real


def test_python_real_time_evolution():
    ham, psi0 = _setup_model()
    ham_matrix = ham.to_linear_operator()

    driver = PyExactTimePropagation(
        ham,
        dt=TIME,
        t0=0.0,
        initial_state=psi0,
        propagation_type="real",
        solver_kwargs={"atol": ATOL, "rtol": RTOL},
    )

    assert np.allclose(driver.state, psi0)

    driver.advance(1)
    assert driver.t == TIME

    def rhs_real(t, x):
        return -1.0j * ham_matrix.dot(x)

    res = solve_ivp(rhs_real, (0.0, TIME), psi0, atol=ATOL, rtol=RTOL)
    psi_scipy = res.y[:, -1]

    assert driver.t == approx(res.t[-1])
    assert overlap(driver.state, psi_scipy) == approx(1.0)


def test_python_imag_time_evolution():
    ham, psi0 = _setup_model()
    ham_matrix = ham.to_linear_operator()

    driver = PyExactTimePropagation(
        ham,
        dt=TIME,
        t0=0.0,
        initial_state=psi0,
        propagation_type="imaginary",
        solver_kwargs={"atol": ATOL, "rtol": RTOL},
    )

    assert np.allclose(driver.state, psi0)

    driver.advance(1)
    assert driver.t == TIME

    def rhs_imag(t, x):
        mean = np.vdot(x, ham_matrix.dot(x))
        return -ham_matrix.dot(x) + mean * x

    res = solve_ivp(rhs_imag, (0.0, TIME), psi0, atol=ATOL, rtol=RTOL)
    psi_scipy = res.y[:, -1]

    assert driver.t == approx(res.t[-1])
    assert overlap(driver.state, psi_scipy) == approx(1.0)


def test_time_stepping():
    ham, psi0 = _setup_model()

    driver = PyExactTimePropagation(
        ham,
        t0=0.0,
        dt=0.05,
        initial_state=psi0,
        propagation_type="real",
        solver_kwargs={"atol": 1e-2, "rtol": 1e-2},
    )
    ts = [driver.t for _ in driver.iter(21)]

    ts_ref = np.linspace(0, 1, 21, endpoint=True)
    assert np.allclose(ts, ts_ref)
