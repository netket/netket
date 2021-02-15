from functools import partial
from io import StringIO

import pytest
from pytest import approx, raises

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket import nn as nknn
import flax

from contextlib import redirect_stderr
import tempfile
import re

SEED = 214748364

machines = {}

standard_init = flax.linen.initializers.normal()
RBM = partial(nk.models.RBM, bias_init=standard_init, visible_bias_init=standard_init)
RBMModPhase = partial(nk.models.RBMModPhase, bias_init=standard_init)


machines["model:(R->R)"] = RBM(alpha=1, dtype=float)
machines["model:(R->C)"] = RBMModPhase(alpha=1, dtype=float)
machines["operator:(C->C)"] = RBM(alpha=1, dtype=complex)

operators = {}

L = 4
g = nk.graph.Hypercube(length=L, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=L)

operators["operator:(Hermitian Real)"] = nk.operator.Ising(hi, graph=g, h=1.0)

H = nk.operator.Ising(hi, graph=g, h=1.0)
for i in range(H.hilbert.size):
    H += nk.operator.spin.sigmay(H.hilbert, i)

operators["operator:(Hermitian Complex)"] = H
# operators["Non Hermitian"] =


@pytest.fixture(params=[pytest.param(ma, id=name) for name, ma in machines.items()])
def vstate(request):
    ma = request.param

    sa = nk.sampler.ExactSampler(hilbert=hi, n_chains=16)

    vs = nk.variational.MCState(sa, ma, n_samples=1000, seed=SEED)

    return vs


@pytest.mark.parametrize(
    "operator", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_expect(vstate, operator):
    #  Use lots of samples
    vstate.n_samples = 2 * 1e5
    vstate.n_discard = 1e3

    # sample the expectation value and gradient with tons of samples
    O_stat1 = vstate.expect(operator)
    O_stat, O_grad = vstate.expect_and_grad(operator)

    # Check that expect and expect_and_grad give same expect. value
    O1_mean = np.asarray(O_stat1.mean)
    O_mean = np.asarray(O_stat1.mean)

    assert O1_mean.real == approx(O_mean.real, abs=1e-8)
    if not operator.is_hermitian:
        assert O1_mean.imag == approx(O_mean.imag, abs=1e-8)

    assert np.asarray(O_stat1.variance) == approx(np.asarray(O_stat.variance), abs=1e-5)

    # Prepare the exact estimations
    pars_0 = vstate.parameters
    pars, unravel = nk.jax.tree_ravel(pars_0)
    op_sparse = operator.to_sparse()

    def expval_fun(par, vstate, H):
        return _expval(unravel(par), vstate, H, real=operator.is_hermitian)

    # Compute the expval and gradient with exact formula
    O_exact = expval_fun(pars, vstate, op_sparse)
    grad_exact = central_diff_grad(expval_fun, pars, 1.0e-5, vstate, op_sparse)

    # compare the two
    err = 6 / np.sqrt(vstate.n_samples)
    O_grad, _ = nk.jax.tree_ravel(O_grad)
    same_derivatives(O_grad, grad_exact, abs_eps=err, rel_eps=1.0e-3)

    # check the exppectation values
    assert O_stat.mean == approx(O_exact, abs=err)


###
def _expval(par, vstate, H, real=False):
    vstate.parameters = par
    psi = vstate.to_array()
    expval = psi.conj() @ H @ psi
    if real:
        expval = np.real(expval)

    return expval


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

    assert np.max(np.real(der_log - num_der_log)) == approx(
        0.0, rel=rel_eps, abs=abs_eps
    )
    # The imaginary part is a bit more tricky, there might be an arbitrary phase shift
    assert np.max(np.exp(np.imag(der_log - num_der_log) * 1.0j) - 1.0) == approx(
        0.0, rel=rel_eps, abs=abs_eps
    )
