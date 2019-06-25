import numpy as np
from pytest import approx
from scipy.integrate import solve_ivp
from scipy.linalg import norm

import netket as nk
from netket.dynamics import timestepper
from netket.exact import ExactTimePropagation


ATOL = 1e-9
RTOL = 1e-9
TIME = 20.0


def _setup_model():
    g = nk.graph.Hypercube(8, 1)
    hi = nk.hilbert.Spin(g, 0.5)
    ham = nk.operator.Heisenberg(hi)

    ts = timestepper(hi.n_states, abs_tol=ATOL, rel_tol=RTOL)
    psi0 = np.random.rand(hi.n_states) + 1j * np.random.rand(hi.n_states)
    psi0 /= norm(psi0)

    return hi, ham, ts, psi0


def overlap(phi, psi):
    return np.abs(np.vdot(phi, psi)) ** 2 / (np.vdot(phi, phi) * np.vdot(psi, psi)).real


def test_real_time_evolution():
    hi, ham, ts, psi0 = _setup_model()
    ham_matrix = ham.to_sparse()

    driver = ExactTimePropagation(
        ham, ts, t0=0.0, initial_state=psi0, propagation_type="real"
    )

    assert overlap(driver.state, psi0) == approx(1.0)

    driver.advance(TIME)
    assert driver.t == TIME

    def rhs_real(t, x):
        return -1.0j * ham_matrix.dot(x)

    res = solve_ivp(rhs_real, (0.0, TIME), psi0, atol=ATOL, rtol=RTOL)
    psi_scipy = res.y[:, -1]

    assert driver.t == approx(res.t[-1])
    assert overlap(driver.state, psi_scipy) == approx(1.0)


def test_imag_time_evolution():
    hi, ham, ts, psi0 = _setup_model()
    ham_matrix = ham.to_sparse()

    driver = ExactTimePropagation(
        ham, ts, t0=0.0, initial_state=psi0, propagation_type="imaginary"
    )

    assert overlap(driver.state, psi0) == approx(1.0)

    driver.advance(TIME)
    assert driver.t == TIME

    def rhs_imag(t, x):
        mean = np.vdot(x, ham_matrix.dot(x))
        return -ham_matrix.dot(x) + mean * x

    res = solve_ivp(rhs_imag, (0.0, TIME), psi0, atol=ATOL, rtol=RTOL)
    psi_scipy = res.y[:, -1]

    assert driver.t == approx(res.t[-1])
    assert overlap(driver.state, psi_scipy) == approx(1.0)
