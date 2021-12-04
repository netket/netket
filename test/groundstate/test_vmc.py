from functools import partial
from io import StringIO

import pytest
from pytest import approx, raises

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk

from contextlib import redirect_stderr
import tempfile
import re

from .. import common

pytestmark = common.skipif_mpi

SEED = 214748364


def _setup_vmc(dtype=np.float32, sr=True):
    L = 4
    g = nk.graph.Hypercube(length=L, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ma = nk.models.RBM(alpha=1, dtype=dtype)
    sa = nk.sampler.ExactSampler(hilbert=hi)

    vs = nk.vqs.MCState(sa, ma, n_samples=1000, seed=SEED)

    ha = nk.operator.Ising(hi, graph=g, h=1.0)
    op = nk.optimizer.Sgd(learning_rate=0.05)

    # Add custom observable
    X = [[0, 1], [1, 0]]
    sx = nk.operator.LocalOperator(hi, [X] * L, [[i] for i in range(8)])

    if sr:
        sr_config = nk.optimizer.SR()
    else:
        sr_config = None
    driver = nk.VMC(ha, op, variational_state=vs, preconditioner=sr_config)

    return ha, sx, vs, sa, driver


#####
@partial(jax.jit, static_argnums=0)
@partial(jax.vmap, in_axes=(None, None, 0, 0, 0), out_axes=(0))
def local_value_kernel(logpsi, pars, σ, σp, mel):
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


def local_values(logpsi, variables, Ô, σ):
    """
    Returns loc_vals for an operator O
    """
    σp, mels = Ô.get_conn_padded(np.asarray(σ).reshape((-1, σ.shape[-1])))
    loc_vals = local_value_kernel(
        logpsi, variables, σ.reshape((-1, σ.shape[-1])), σp, mels
    )
    return loc_vals.reshape(σ.shape[:-1])


def estimate_gradient(logpsi, variables, Ô, σ):
    """
    Estimate the gradient of hermmitian operator O using legacy code
    """
    O_loc = local_values(logpsi, variables, Ô, σ)
    O_loc -= O_loc.mean()
    O_loc /= np.prod(O_loc.shape)

    σr = σ.reshape((-1, σ.shape[-1]))
    O_loc_r = O_loc.reshape(-1)

    _, vjp = nk.jax.vjp(logpsi, variables, σr, conjugate=True)
    Opsi = vjp(O_loc_r.conjugate())[0]

    return Opsi


####


def test_estimate():
    ha, *_, driver = _setup_vmc()
    driver.estimate(ha)
    driver.advance(1)
    driver.estimate(ha)


def test_raise_n_iter():
    ha, sx, ma, sampler, driver = _setup_vmc()
    with raises(
        ValueError,
    ):
        driver.run("prova", 12)


def test_reset():
    ha, *_, driver = _setup_vmc()
    assert driver.step_count == 0
    driver.advance(1)
    assert driver.step_count == 1
    driver.reset()
    assert driver.step_count == 0


def test_vmc_construction_vstate():
    ha, sx, ma, sa, driver = _setup_vmc()

    op = nk.optimizer.Sgd(learning_rate=0.05)

    driver = nk.Vmc(ha, op, sa, nk.models.RBM(), n_samples=1000, seed=SEED)

    driver.run(1)

    assert driver.step_count == 1

    with raises(TypeError):
        ha2 = nk.operator.LocalOperator(ha.hilbert * ha.hilbert)
        driver = nk.VMC(ha2, op, variational_state=driver.state)


def test_vmc_functions():
    ha, sx, ma, sampler, driver = _setup_vmc()

    driver.advance(500)

    assert driver.energy.mean == approx(ma.expect(ha).mean, abs=1e-5)

    state = ma.to_array()

    n_samples = 16000
    ma.n_samples = n_samples
    ma.n_discard_per_chain = 100

    # Check zero gradient
    _, grads = ma.expect_and_grad(ha)

    def check_shape(a, b):
        assert a.shape == b.shape

    jax.tree_multimap(check_shape, grads, ma.parameters)
    grads, _ = nk.jax.tree_ravel(grads)

    assert np.mean(np.abs(grads) ** 2) == approx(0.0, abs=1e-8)
    # end

    for op, name in (ha, "ha"), (sx, "sx"):
        print("Testing expectation of op={}".format(name))

        exact_ex = (state.T.conj() @ op.to_linear_operator() @ state).real

        op_stats = ma.expect(op)

        mean = op_stats.mean
        var = op_stats.variance
        print(mean, var)

        # 5-sigma test for expectation values
        tol = np.sqrt(var / float(ma.n_samples)) * 5
        assert mean.real == approx(exact_ex, abs=tol)


def test_vmc_progress_bar():
    ha, sx, ma, sampler, driver = _setup_vmc()
    tempdir = tempfile.mkdtemp()
    prefix = tempdir + "/vmc_progressbar_test"

    f = StringIO()
    with redirect_stderr(f):
        driver.run(5, prefix)
    pbar = f.getvalue().split("\r")[-1]
    assert re.match(r"100%\|#*\| (\d+)/\1", pbar)
    assert re.search(r"Energy=[-+]?[0-9]*\.?[0-9]*", pbar)
    assert re.search(r"σ²=[-+]?[0-9]*\.?[0-9]*", pbar)

    f = StringIO()
    with redirect_stderr(f):
        driver.run(5, prefix, show_progress=None)
    pbar = f.getvalue()
    assert not len(pbar)


def _energy(par, vstate, H):
    vstate.parameters = par
    psi = vstate.to_array()
    return np.real(psi.conj() @ H @ psi)


def central_diff_grad(func, x, eps, *args):
    grad = np.zeros(len(x), dtype=x.dtype)
    epsd = np.zeros(len(x), dtype=x.dtype)
    epsd[0] = eps
    for i in range(len(x)):
        assert not np.any(np.isnan(x + epsd))
        grad_r = 0.5 * (func(x + epsd, *args) - func(x - epsd, *args))
        if nk.jax.is_complex(x):
            grad_i = 0.5 * (func(x + 1j * epsd, *args) - func(x - 1j * epsd, *args))
            grad[i] = 0.5 * grad_r + 0.5j * grad_i
        else:
            grad_i = 0.0
            grad[i] = 0.5 * grad_r

        assert not np.isnan(grad[i])
        grad[i] /= eps
        epsd = np.roll(epsd, 1)
    return grad


def same_derivatives(der_log, num_der_log, abs_eps=1.0e-6, rel_eps=1.0e-6):
    assert der_log.shape == num_der_log.shape

    np.testing.assert_allclose(
        der_log.real, num_der_log.real, rtol=rel_eps, atol=abs_eps
    )

    # The imaginary part is a bit more tricky, there might be an arbitrary phase shift
    assert np.max(
        np.abs(np.exp(np.imag(der_log - num_der_log) * 1.0j) - 1.0)
    ) == approx(0.0, rel=rel_eps, abs=abs_eps)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_vmc_gradient(dtype):
    ha, sx, ma, sampler, driver = _setup_vmc(dtype=dtype, sr=False)
    driver.run(3, out=None)

    # old formula
    _, grads = ma.expect_and_grad(ha)
    grad_old = estimate_gradient(
        lambda w, x: ma._apply_fun({"params": w}, x), ma.parameters, ha, ma.samples
    )

    # check that the old fformula and the new give same result
    jax.tree_multimap(lambda x, y: same_derivatives(x, y), grads, grad_old)
    # end old

    pars_0 = ma.parameters
    pars, unravel = nk.jax.tree_ravel(pars_0)

    def energy_fun(par, vstate, H):
        return _energy(unravel(par), vstate, H)

    grad_exact = central_diff_grad(energy_fun, pars, 1.0e-5, ma, ha.to_sparse())

    driver.state.n_samples = 1e5
    driver.state.n_discard_per_chain = 1e3
    driver.state.parameters = pars_0
    _, _grad_approx = ma.expect_and_grad(ha)  # driver._forward_and_backward()
    grad_approx, _ = nk.jax.tree_ravel(_grad_approx)

    err = 6 / np.sqrt(driver.state.n_samples)  # improve error bound
    same_derivatives(grad_approx, grad_exact, abs_eps=err, rel_eps=1.0e-3)


def test_vmc_sr_legacy_api():
    ha, sx, ma, sampler, driver = _setup_vmc(sr=True)
    op = driver.optimizer
    vs = driver.state
    sr_config = driver.preconditioner

    with pytest.warns(FutureWarning):
        driver = nk.VMC(
            ha,
            op,
            variational_state=vs,
            sr=sr_config,
        )

    with pytest.warns(FutureWarning):
        driver = nk.VMC(
            ha,
            op,
            variational_state=vs,
            preconditioner=sr_config,
            sr_restart=True,
        )

    with pytest.raises(ValueError):
        driver = nk.VMC(
            ha, op, variational_state=vs, sr=sr_config, preconditioner=sr_config
        )
