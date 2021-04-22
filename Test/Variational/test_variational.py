from functools import partial
from io import StringIO

import pytest
from pytest import approx, raises

import numpy as np
from numpy import testing
import jax
import jax.numpy as jnp
import netket as nk
from netket import nn as nknn
import flax

from contextlib import redirect_stderr
import tempfile
import re

nk.config.update("NETKET_EXPERIMENTAL", True)

SEED = 2148364

machines = {}

standard_init = flax.linen.initializers.normal()
RBM = partial(
    nk.models.RBM, hidden_bias_init=standard_init, visible_bias_init=standard_init
)
RBMModPhase = partial(nk.models.RBMModPhase, hidden_bias_init=standard_init)

nk.models.RBM(
    alpha=1,
    dtype=complex,
    kernel_init=nk.nn.initializers.normal(stddev=0.1),
    hidden_bias_init=nk.nn.initializers.normal(stddev=0.1),
)
machines["model:(R->R)"] = RBM(
    alpha=1,
    dtype=float,
    kernel_init=nk.nn.initializers.normal(stddev=0.1),
    hidden_bias_init=nk.nn.initializers.normal(stddev=0.1),
)
machines["model:(R->C)"] = RBMModPhase(
    alpha=1,
    dtype=float,
    kernel_init=nk.nn.initializers.normal(stddev=0.1),
    hidden_bias_init=nk.nn.initializers.normal(stddev=0.1),
)
machines["model:(C->C)"] = RBM(
    alpha=1,
    dtype=complex,
    kernel_init=nk.nn.initializers.normal(stddev=0.1),
    hidden_bias_init=nk.nn.initializers.normal(stddev=0.1),
)

operators = {}

L = 4
g = nk.graph.Hypercube(length=L, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=L)

operators["operator:(Hermitian Real)"] = nk.operator.Ising(hi, graph=g, h=1.0)

H = nk.operator.Ising(hi, graph=g, h=1.0)
for i in range(H.hilbert.size):
    H += nk.operator.spin.sigmay(H.hilbert, i)

operators["operator:(Hermitian Complex)"] = H

H = H.copy()
for i in range(H.hilbert.size):
    H += nk.operator.spin.sigmap(H.hilbert, i)

operators["operator:(Non Hermitian)"] = H


@pytest.fixture(params=[pytest.param(ma, id=name) for name, ma in machines.items()])
def vstate(request):
    ma = request.param

    sa = nk.sampler.ExactSampler(hilbert=hi, n_chains=16)

    vs = nk.variational.MCState(sa, ma, n_samples=1000, seed=SEED)

    return vs


def test_n_samples_api(vstate):
    with raises(
        ValueError,
    ):
        vstate.n_samples = -1

    with raises(
        ValueError,
    ):
        vstate.chain_length = -2

    with raises(
        ValueError,
    ):
        vstate.n_discard = -1

    vstate.n_samples = 2
    assert vstate.samples.shape[0:2] == (1, vstate.sampler.n_chains)

    vstate.chain_length = 2
    assert vstate.n_samples == 2 * vstate.sampler.n_chains
    assert vstate.samples.shape[0:2] == (2, vstate.sampler.n_chains)

    vstate.n_samples = 1000
    vstate.n_discard = None
    assert vstate.n_discard == 0

    vstate.sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)
    vstate.n_discard = None
    assert vstate.n_discard == vstate.n_samples // 10


def test_serialization(vstate):
    from flax import serialization

    bdata = serialization.to_bytes(vstate)

    old_params = vstate.parameters
    old_samples = vstate.samples
    old_nsamples = vstate.n_samples
    old_ndiscard = vstate.n_discard

    vstate = nk.variational.MCState(
        vstate.sampler, vstate.model, n_samples=10, seed=SEED + 100
    )

    vstate = serialization.from_bytes(vstate, bdata)

    jax.tree_multimap(np.testing.assert_allclose, vstate.parameters, old_params)
    np.testing.assert_allclose(vstate.samples, old_samples)
    assert vstate.n_samples == old_nsamples
    assert vstate.n_discard == old_ndiscard


def test_init_parameters(vstate):
    vstate.init_parameters(seed=SEED)
    pars = vstate.parameters
    vstate.init_parameters(nk.nn.initializers.normal(stddev=0.01), seed=SEED)
    pars2 = vstate.parameters

    def _f(x, y):
        np.testing.assert_allclose(x, y)

    jax.tree_multimap(_f, pars, pars2)


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(
            op,
            id=name,
        )
        for name, op in operators.items()
    ],
)
def test_expect_numpysampler_works(vstate, operator):
    sampl = nk.sampler.MetropolisLocalNumpy(vstate.hilbert)
    vstate.sampler = sampl
    out = vstate.expect(operator)
    assert isinstance(out, nk.stats.Stats)


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(
            op,
            id=name,
            marks=pytest.mark.xfail(
                reason="MUSTFIX: Non hermitian gradient is known to be wrong"
            )
            if not op.is_hermitian
            else [],
        )
        for name, op in operators.items()
    ],
)
def test_expect(vstate, operator):
    #  Use lots of samples
    vstate.n_samples = 5 * 1e5
    vstate.n_discard = 1e3

    # sample the expectation value and gradient with tons of samples
    O_stat1 = vstate.expect(operator)
    O_stat, O_grad = vstate.expect_and_grad(operator)

    O1_mean = np.asarray(O_stat1.mean)
    O_mean = np.asarray(O_stat.mean)

    # check that vstate.expect gives the right result
    O_expval_exact = _expval(
        vstate.parameters, vstate, operator, real=operator.is_hermitian
    )
    np.testing.assert_allclose(O_expval_exact.real, O1_mean.real, atol=1e-3, rtol=1e-3)
    if not operator.is_hermitian:
        np.testing.assert_allclose(
            O_expval_exact.imag, O1_mean.imag, atol=1e-3, rtol=1e-3
        )

    # Check that expect and expect_and_grad give same expect. value
    assert O1_mean.real == approx(O_mean.real, abs=1e-5)
    if not operator.is_hermitian:
        assert O1_mean.imag == approx(O_mean.imag, abs=1e-5)

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

    if not operator.is_hermitian:
        grad_exact = jax.tree_map(lambda x: x * 2, grad_exact)

    # compare the two
    err = 5 / np.sqrt(vstate.n_samples)

    # check the expectation values
    assert O_stat.mean == approx(O_exact, abs=err)

    O_grad, _ = nk.jax.tree_ravel(O_grad)
    same_derivatives(O_grad, grad_exact, abs_eps=err, rel_eps=err)


###
def _expval(par, vstate, H, real=False):
    vstate.parameters = par
    psi = vstate.to_array()
    expval = psi.conj() @ (H @ psi)
    if real:
        expval = np.real(expval)

    return expval


def central_diff_grad(func, x, eps, *args, dtype=None):
    if dtype is None:
        dtype = x.dtype

    grad = np.zeros(
        len(x), dtype=nk.jax.maybe_promote_to_complex(x.dtype, func(x, *args).dtype)
    )
    epsd = np.zeros(len(x), dtype=dtype)
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
    np.testing.assert_allclose(
        np.mod(der_log.imag, np.pi * 2),
        np.mod(num_der_log.imag, np.pi * 2),
        rtol=rel_eps,
        atol=abs_eps,
    )
