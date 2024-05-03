# Copyright 2021 The NetKet Authors - All rights reserved.
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
import copy

import pytest
from pytest import approx, raises

import jax
import numpy as np

import netket as nk
from jax.nn.initializers import normal

from .finite_diff import expval as _expval, central_diff_grad, same_derivatives
from .. import common

nk.config.update("NETKET_EXPERIMENTAL", True)

SEED = 2148364

machines = {}

standard_init = normal()
RBM = partial(
    nk.models.RBM, hidden_bias_init=standard_init, visible_bias_init=standard_init
)
RBMModPhase = partial(nk.models.RBMModPhase, hidden_bias_init=standard_init)

nk.models.RBM(
    alpha=1,
    param_dtype=complex,
    kernel_init=normal(stddev=0.1),
    hidden_bias_init=normal(stddev=0.1),
)
machines["model:(R->R)"] = RBM(
    alpha=1,
    param_dtype=float,
    kernel_init=normal(stddev=0.1),
    hidden_bias_init=normal(stddev=0.1),
)
machines["model:(R->C)"] = RBMModPhase(
    alpha=1,
    param_dtype=float,
    kernel_init=normal(stddev=0.1),
    hidden_bias_init=normal(stddev=0.1),
)
machines["model:(C->C)"] = RBM(
    alpha=1,
    param_dtype=complex,
    kernel_init=normal(stddev=0.1),
    hidden_bias_init=normal(stddev=0.1),
)
machines["model:(C->C,nonholo)"] = nk.models.RBM(
    alpha=1,
    param_dtype=complex,
    activation=lambda x: nk.nn.log_cosh(0.3 * x * x + 0.6 * x.conj()),
)

operators = {}

L = 4
g = nk.graph.Hypercube(length=L, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=L)

H = nk.operator.Ising(hi, graph=g, h=1.0)
operators["operator:(Hermitian Real)"] = H
operators["operator:(IsingJax)"] = H.to_jax_operator()

H2 = H @ H
operators["operator:(Hermitian Real Squared)"] = H2

H = nk.operator.Ising(hi, graph=g, h=1.0, dtype=complex)
for i in range(H.hilbert.size):
    H += nk.operator.spin.sigmay(H.hilbert, i)

operators["operator:(Hermitian Complex)"] = H
operators["operator:(Hermitian Complex Squared)"] = H.H @ H

H = H.copy()
for i in range(H.hilbert.size):
    H += nk.operator.spin.sigmap(H.hilbert, i)

operators["operator:(Non Hermitian)"] = H


@pytest.fixture(params=[pytest.param(ma, id=name) for name, ma in machines.items()])
def vstate(request):
    ma = request.param

    sa = nk.sampler.ExactSampler(hilbert=hi)

    vs = nk.vqs.MCState(sa, ma, n_samples=1000, seed=SEED)

    # wrap the apply function with a check that the samples have exactly 0 or 1 batch axis
    def apply_fun_check_shape(apply_fun, params, x, *args, **kwargs):
        assert x.ndim in (1, 2)
        return apply_fun(params, x, *args, **kwargs)

    apply_fun = vs._apply_fun
    vs._apply_fun = partial(apply_fun_check_shape, apply_fun)

    return vs


def check_consistent(vstate, mpi_size):
    assert vstate.n_samples == vstate.n_samples_per_rank * mpi_size
    assert vstate.n_samples == vstate.chain_length * vstate.sampler.n_chains


def test_n_samples_api(vstate, _device_count):
    with raises(TypeError, match="should be a subtype"):
        vstate.sampler = 1

    with raises(
        ValueError,
    ):
        vstate.n_samples = -1

    with raises(
        ValueError,
    ):
        vstate.n_samples_per_rank = -1

    with raises(
        ValueError,
    ):
        vstate.chain_length = -2

    with raises(
        ValueError,
    ):
        vstate.n_discard_per_chain = -1

    # Tests for `ExactSampler` with `n_chains == 1`
    vstate.n_samples = 3
    check_consistent(vstate, _device_count)
    assert vstate.samples.shape[0:2] == (
        vstate.sampler.n_batches,
        int(np.ceil(3 / _device_count)),
    )

    vstate.n_samples_per_rank = 4
    check_consistent(vstate, _device_count)
    assert vstate.samples.shape[0:2] == (vstate.sampler.n_batches, 4)

    vstate.chain_length = 2
    check_consistent(vstate, _device_count)
    assert vstate.samples.shape[0:2] == (vstate.sampler.n_batches, 2)

    vstate.n_samples = 1000
    vstate.n_discard_per_chain = None
    assert vstate.n_discard_per_chain == 0

    # Tests for `MetropolisLocal` with `n_chains > 1`
    vstate.sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)
    # `n_samples` is rounded up
    assert vstate.n_samples == 1008
    assert vstate.chain_length == 63
    check_consistent(vstate, _device_count)

    vstate.n_discard_per_chain = None
    assert vstate.n_discard_per_chain == vstate.n_samples // 10

    vstate.n_samples = 3
    check_consistent(vstate, _device_count)
    # `n_samples` is rounded up
    assert vstate.samples.shape[0:2] == (vstate.sampler.n_batches, 1)

    vstate.n_samples_per_rank = 16 // _device_count + 1
    check_consistent(vstate, _device_count)
    # `n_samples` is rounded up
    assert vstate.samples.shape[0:2] == (vstate.sampler.n_batches, 2)

    vstate.chain_length = 2
    check_consistent(vstate, _device_count)
    assert vstate.samples.shape[0:2] == (vstate.sampler.n_batches, 2)


@common.skipif_mpi
def test_chunk_size_api(vstate, _mpi_size):
    assert vstate.chunk_size is None

    with raises(
        ValueError,
    ):
        vstate.chunk_size = -1

    vstate.n_samples = 1008

    # does not divide n_samples
    with raises(
        ValueError,
    ):
        vstate.chunk_size = 100

    assert vstate.chunk_size is None

    vstate.chunk_size = 126
    assert vstate.chunk_size == 126

    vstate.n_samples = 1008 * 2
    assert vstate.chunk_size == 126

    with raises(
        ValueError,
    ):
        vstate.chunk_size = 500

    _ = vstate.sample()
    _ = vstate.sample(n_samples=vstate.n_samples)
    with raises(
        ValueError,
    ):
        vstate.sample(n_samples=1008 + 16)

    with raises(
        ValueError,
    ):
        vstate.sample(n_samples=1008, chain_length=100)


@common.skipif_mpi
def test_constructor():
    sampler = nk.sampler.MetropolisLocal(hilbert=hi)
    model = nk.models.RBM()
    vs_good = nk.vqs.MCState(sampler, model)

    vs = nk.vqs.MCState(
        sampler, model, n_samples_per_rank=sampler.n_chains_per_rank * 2
    )
    assert vs.n_samples_per_rank == sampler.n_chains_per_rank * 2

    with pytest.raises(ValueError, match="Only one argument between"):
        vs = nk.vqs.MCState(sampler, model, n_samples=100, n_samples_per_rank=100)

    with pytest.raises(ValueError, match="Must either pass the model or apply_fun"):
        vs = nk.vqs.MCState(sampler)

    # test init with parameters and variables
    vs = nk.vqs.MCState(sampler, apply_fun=model.apply, init_fun=model.init)

    vs = nk.vqs.MCState(sampler, apply_fun=model.apply, variables=vs_good.variables)
    with pytest.raises(RuntimeError, match="you did not supply a valid init_function"):
        vs.init()

    with pytest.raises(ValueError, match="you must pass a valid init_fun."):
        vs = nk.vqs.MCState(sampler, apply_fun=model.apply)


@common.skipif_mpi
def test_serialization(vstate):
    from flax import serialization

    vstate.chunk_size = 12345

    bdata = serialization.to_bytes(vstate)

    old_params = vstate.parameters
    old_samples = vstate.samples
    old_nsamples = vstate.n_samples
    old_ndiscard = vstate.n_discard_per_chain
    old_chunksize = vstate.chunk_size

    # test same samples, serialization before sampling
    vstate = nk.vqs.MCState(vstate.sampler, vstate.model, n_samples=10, seed=SEED + 100)

    vstate = serialization.from_bytes(vstate, bdata)

    jax.tree_util.tree_map(np.testing.assert_allclose, vstate.parameters, old_params)
    np.testing.assert_allclose(vstate.samples, old_samples)
    assert vstate.n_samples == old_nsamples
    assert vstate.n_discard_per_chain == old_ndiscard
    assert vstate.chunk_size == old_chunksize

    # test samples before serailziation
    old_samples = vstate.samples
    bdata = serialization.to_bytes(vstate)
    vstate = serialization.from_bytes(vstate, bdata)

    np.testing.assert_allclose(vstate.samples, old_samples)


@common.skipif_mpi
def test_init_parameters(vstate):
    vstate.init_parameters(seed=SEED)
    pars = vstate.parameters
    vstate.init_parameters(normal(stddev=0.01), seed=SEED)
    pars2 = vstate.parameters

    def _f(x, y):
        np.testing.assert_allclose(x, y)

    jax.tree_util.tree_map(_f, pars, pars2)


@common.skipif_mpi
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


@common.skipif_mpi
def test_qutip_conversion(vstate):
    # skip test if qutip not installed
    pytest.importorskip("qutip")

    ket = vstate.to_array()
    q_obj = vstate.to_qobj()

    assert q_obj.type == "ket"
    assert len(q_obj.dims) == 2
    assert q_obj.dims[0] == list(vstate.hilbert.shape)
    assert q_obj.dims[1] == [1 for i in range(vstate.hilbert.size)]

    assert q_obj.shape == (vstate.hilbert.n_states, 1)
    np.testing.assert_allclose(q_obj.data.to_array(), ket.reshape(q_obj.shape))


@common.skipif_mpi
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
    # Use lots of samples
    vstate.n_samples = 256 * 1024
    vstate.n_discard_per_chain = 1024

    # sample the expectation value and gradient with tons of samples
    O_stat1 = vstate.expect(operator)
    O_stat, O_grad = vstate.expect_and_grad(operator)

    O1_mean = np.asarray(O_stat1.mean)
    O_mean = np.asarray(O_stat.mean)
    err = 5 * O_stat1.error_of_mean

    # check that vstate.expect gives the right result
    O_expval_exact = _expval(
        vstate.parameters, vstate, operator, real=operator.is_hermitian
    )

    np.testing.assert_allclose(O_expval_exact.real, O1_mean.real, atol=err, rtol=err)
    if not operator.is_hermitian:
        np.testing.assert_allclose(
            O_expval_exact.imag, O1_mean.imag, atol=err, rtol=err
        )

    # Check that expect and expect_and_grad give same expect. value
    assert O1_mean.real == approx(O_mean.real, abs=1e-5)
    if not operator.is_hermitian:
        assert O1_mean.imag == approx(O_mean.imag, abs=1e-5)

    np.testing.assert_allclose(O_stat1.variance, O_stat.variance, atol=1e-5)

    # Prepare the exact estimations
    pars_0 = vstate.parameters
    pars, unravel = nk.jax.tree_ravel(pars_0)
    op_sparse = operator.to_sparse()

    def expval_fun(par, vstate, H):
        return _expval(unravel(par), vstate, H, real=operator.is_hermitian)

    # Compute the expval and gradient with exact formula
    O_exact = expval_fun(pars, vstate, op_sparse)
    grad_exact = central_diff_grad(expval_fun, pars, 1.0e-5, vstate, op_sparse)

    # check the expectation values
    err = 5 * O_stat.error_of_mean
    assert O_stat.mean == approx(O_exact, abs=err)

    O_grad, _ = nk.jax.tree_ravel(O_grad)
    same_derivatives(O_grad, grad_exact, abs_eps=err, rel_eps=err)


@common.skipif_mpi
@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(
            op,
            id=name,
            marks=pytest.mark.xfail(
                reason="MUSTFIX: Non-hermitian and Squared forces known to be wrong"
            )
            if isinstance(op, nk.operator.Squared) or (not op.is_hermitian)
            else [],
        )
        for name, op in operators.items()
    ],
)
def test_forces(vstate, operator):
    _, O_grad1 = vstate.expect_and_grad(operator)
    O_grad2 = vstate.grad(operator)
    jax.tree_util.tree_map(np.testing.assert_array_equal, O_grad1, O_grad2)

    _, f1 = vstate.expect_and_forces(operator)
    if nk.jax.tree_leaf_iscomplex(vstate.parameters):
        g1 = jax.tree_util.tree_map(lambda x: 2.0 * x, f1)
    else:
        g1 = jax.tree_util.tree_map(lambda x: 2.0 * np.real(x), f1)
    jax.tree_util.tree_map(np.testing.assert_array_equal, g1, O_grad1)


@common.skipif_mpi
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
def test_local_estimators(vstate, operator):
    def assert_stats_equal(st1, st2):
        assert st1.mean == pytest.approx(st2.mean)
        assert st1.variance == pytest.approx(st2.variance)
        assert st1.error_of_mean == pytest.approx(st2.error_of_mean)

    def inner_test():
        oloc = vstate.local_estimators(operator)
        assert oloc.shape == (vstate.sampler.n_chains, vstate.chain_length)

        stats1 = nk.stats.statistics(oloc)
        stats2 = vstate.expect(operator)
        assert_stats_equal(stats1, stats2)

    # no chunking
    inner_test()
    # chunking
    vstate.chunk_size = 2
    inner_test()


# Have a different test because the above is marked as xfail.
# This only checks that the code runs.
@common.xfailif_mpi
def test_expect_grad_nonhermitian_works(vstate):
    op = nk.operator.spin.sigmap(vstate.hilbert, 0)
    O_stat, O_grad = vstate.expect_and_grad(op)


@common.skipif_mpi
@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(
            op,
            id=name,
        )
        for name, op in operators.items()
        if op.is_hermitian
    ],
)
@pytest.mark.parametrize("n_chunks", [1, 2])
def test_expect_chunking(vstate, operator, n_chunks):
    vstate.n_samples = 200
    chunk_size = vstate.n_samples_per_rank // n_chunks

    eval_nochunk = vstate.expect(operator)
    vstate.chunk_size = chunk_size
    eval_chunk = vstate.expect(operator)

    jax.tree_util.tree_map(
        partial(np.testing.assert_allclose, atol=1e-13), eval_nochunk, eval_chunk
    )

    vstate.chunk_size = None
    grad_nochunk = vstate.grad(operator)
    vstate.chunk_size = chunk_size
    grad_chunk = vstate.grad(operator)

    jax.tree_util.tree_map(
        partial(np.testing.assert_allclose, atol=1e-13), grad_nochunk, grad_chunk
    )


def test_reproducible_copy():
    # This checks that if i duplicate a variational state and perform the same operations
    # I get exactly the same samples

    hi = nk.hilbert.Spin(0.5, 10)
    ma = nk.models.RBM()
    sa = nk.sampler.MetropolisLocal(hilbert=hi)
    vs = nk.vqs.MCState(sa, ma, n_samples=64)

    # If i copy, I have same sampler_state
    vs2 = copy.copy(vs)
    s1 = vs.samples
    s2 = vs2.samples
    np.testing.assert_allclose(s1, s2)

    # If i change the sampler, I get a new sampler_state and
    # the seed should be computed
    sa = nk.sampler.MetropolisLocal(hilbert=hi, sweep_size=4)

    vs.sampler = sa
    vs2.sampler = sa
    s1_2 = vs.samples
    s2_2 = vs.samples

    # same seed for the new sampler state.
    np.testing.assert_allclose(s1_2, s2_2)
    # But different samples
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(s1, s1_2)
