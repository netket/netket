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

import pytest
from pytest import approx, raises

import numpy as np
import jax
import netket as nk
from jax.nn.initializers import normal

from .finite_diff import expval as _expval, central_diff_grad, same_derivatives
from .. import common

nk.config.update("NETKET_EXPERIMENTAL", True)

SEED = 2148364

machines = {}

standard_init = normal()
NDM = partial(nk.models.NDM, bias_init=standard_init, visible_bias_init=standard_init)

machines["model:(R->C)"] = NDM(
    alpha=1,
    beta=1,
    param_dtype=float,
    kernel_init=normal(stddev=0.1),
    bias_init=normal(stddev=0.1),
)
machines["model:(C->C)"] = nk.models.RBM(
    alpha=1,
    param_dtype=complex,
    kernel_init=normal(stddev=0.1),
    visible_bias_init=normal(stddev=0.1),
    hidden_bias_init=normal(stddev=0.1),
)

operators = {}
superoperators = {}

L = 2
g = nk.graph.Hypercube(length=L, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=L)

ha = nk.operator.Ising(hi, graph=g, h=1.0, dtype=complex)
jump_ops = [nk.operator.spin.sigmam(hi, i) for i in range(L)]

liouv = nk.operator.LocalLiouvillian(ha.to_local_operator(), jump_ops)
LdagL = liouv.H @ liouv

# operators["operator:(Lind)"] = liouv
operators["operator:(Lind^2)"] = LdagL
operators["operator:H"] = ha
operators["operator:sigmam"] = jump_ops[0]

superoperators["superop:(Lind^2)"] = LdagL


@pytest.fixture(params=[pytest.param(ma, id=name) for name, ma in machines.items()])
def vstate(request):
    ma = request.param

    sa = nk.sampler.ExactSampler(hilbert=nk.hilbert.DoubledHilbert(hi))

    vs = nk.vqs.MCMixedState(sa, ma, n_samples=1000, seed=SEED)

    return vs


def test_n_samples_api(vstate, _device_count):
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
    vstate.sampler = nk.sampler.MetropolisLocal(
        hilbert=nk.hilbert.DoubledHilbert(hi), n_chains=16
    )
    # `n_samples` is rounded up
    assert vstate.n_samples == 1008
    assert vstate.chain_length == 63
    check_consistent(vstate, _device_count)

    vstate.n_discard_per_chain = None
    assert vstate.n_discard_per_chain == vstate.n_samples // 10

    vstate.n_samples = 3
    check_consistent(vstate, _device_count)
    # `n_samples` is rounded up
    print(vstate.samples.shape)
    assert vstate.samples.shape[0:2] == (vstate.sampler.n_batches, 1)

    vstate.n_samples_per_rank = 16 // _device_count + 1
    check_consistent(vstate, _device_count)
    # `n_samples` is rounded up
    assert vstate.samples.shape[0:2] == (vstate.sampler.n_batches, 2)

    vstate.chain_length = 2
    check_consistent(vstate, _device_count)
    assert vstate.samples.shape[0:2] == (vstate.sampler.n_batches, 2)


def test_n_samples_diag_api(vstate, _device_count):
    with raises(
        ValueError,
    ):
        vstate.n_samples_diag = -1

    with raises(
        ValueError,
    ):
        vstate.chain_length_diag = -2

    with raises(
        ValueError,
    ):
        vstate.n_discard_per_chain_diag = -1

    # Tests for `ExactSampler` with `n_chains == 1`
    vstate.n_samples_diag = 3
    check_consistent_diag(vstate)
    assert (
        vstate.diagonal.samples.shape[0:2]
        == (vstate.sampler_diag.n_batches, int(np.ceil(3 / _device_count)))
        == (vstate.diagonal.sampler.n_batches, int(np.ceil(3 / _device_count)))
    )

    vstate.chain_length_diag = 2
    check_consistent_diag(vstate)
    assert (
        vstate.diagonal.samples.shape[0:2]
        == (vstate.sampler_diag.n_batches, 2)
        == (vstate.diagonal.sampler.n_batches, 2)
    )

    vstate.n_samples_diag = 1000
    vstate.n_discard_per_chain_diag = None
    assert vstate.n_discard_per_chain_diag == vstate.diagonal.n_discard_per_chain == 0

    # Tests for `MetropolisLocal` with `n_chains > 1`
    vstate.sampler_diag = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)
    # `n_samples_diag` is rounded up
    assert vstate.n_samples_diag == vstate.diagonal.n_samples == 1008
    assert vstate.chain_length_diag == vstate.diagonal.chain_length == 63
    check_consistent_diag(vstate)

    vstate.n_discard_per_chain_diag = None
    assert (
        vstate.n_discard_per_chain_diag
        == vstate.diagonal.n_discard_per_chain
        == vstate.n_samples_diag // 10
    )

    vstate.n_samples_diag = 3
    check_consistent_diag(vstate)
    # `n_samples_diag` is rounded up
    assert (
        vstate.diagonal.samples.shape[0:2]
        == (vstate.sampler_diag.n_batches, 1)
        == (vstate.diagonal.sampler.n_batches, 1)
    )

    vstate.chain_length_diag = 2
    check_consistent_diag(vstate)
    assert (
        vstate.n_samples_diag
        == vstate.chain_length_diag * vstate.sampler_diag.n_chains
        == vstate.diagonal.chain_length * vstate.diagonal.sampler.n_chains
    )
    assert (
        vstate.diagonal.samples.shape[0:2]
        == (vstate.sampler_diag.n_batches, 2)
        == (vstate.diagonal.sampler.n_batches, 2)
    )


@common.skipif_mpi
def test_serialization(vstate):
    from flax import serialization

    vstate.chunk_size = 12345

    bdata = serialization.to_bytes(vstate)

    vstate_new = nk.vqs.MCMixedState(
        vstate.sampler, vstate.model, n_samples=10, seed=SEED + 313
    )

    vstate_new = serialization.from_bytes(vstate_new, bdata)

    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate.parameters, vstate_new.parameters
    )
    np.testing.assert_allclose(vstate.samples, vstate_new.samples)
    np.testing.assert_allclose(vstate.diagonal.samples, vstate_new.diagonal.samples)
    assert vstate.n_samples == vstate_new.n_samples
    assert vstate.n_discard_per_chain == vstate_new.n_discard_per_chain
    assert vstate.n_samples_diag == vstate_new.n_samples_diag
    assert vstate.n_discard_per_chain_diag == vstate_new.n_discard_per_chain_diag
    assert vstate.chunk_size == vstate_new.chunk_size

    old_samples = vstate.samples
    bdata = serialization.to_bytes(vstate)
    vstate_new = serialization.from_bytes(vstate_new, bdata)

    np.testing.assert_allclose(old_samples, vstate_new.samples)


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
    vstate.n_samples = 208
    chunk_size = vstate.n_samples_per_rank // n_chunks
    chunk_size_diag = vstate.diagonal.n_samples_per_rank // n_chunks

    eval_nochunk = vstate.expect(operator)
    vstate.chunk_size = chunk_size
    vstate.diagonal.chunk_size = chunk_size_diag
    eval_chunk = vstate.expect(operator)

    jax.tree_util.tree_map(
        partial(np.testing.assert_allclose, atol=1e-13), eval_nochunk, eval_chunk
    )


@common.skipif_mpi
@pytest.mark.parametrize("n_chunks", [1, 2])
def test_expect_grad_chunking(vstate, n_chunks):
    operator = LdagL

    vstate.n_samples = 208
    chunk_size = vstate.n_samples_per_rank // n_chunks
    chunk_size_diag = vstate.diagonal.n_samples_per_rank // n_chunks

    vstate.chunk_size = None
    vstate.diagonal.chunk_size = None
    grad_nochunk = vstate.grad(operator)
    vstate.chunk_size = chunk_size
    vstate.diagonal.chunk_size = chunk_size_diag
    grad_chunk = vstate.grad(operator)

    jax.tree_util.tree_map(
        partial(np.testing.assert_allclose, atol=1e-13), grad_nochunk, grad_chunk
    )


@common.skipif_mpi
def test_qutip_conversion(vstate):
    # skip test if qutip not installed
    pytest.importorskip("qutip")

    rho = vstate.to_matrix()
    q_obj = vstate.to_qobj()

    assert q_obj.type == "oper"
    assert len(q_obj.dims) == 2
    assert q_obj.dims[0] == list(vstate.hilbert_physical.shape)
    assert q_obj.dims[1] == list(vstate.hilbert_physical.shape)

    assert q_obj.shape == (
        vstate.hilbert_physical.n_states,
        vstate.hilbert_physical.n_states,
    )
    np.testing.assert_allclose(q_obj.data.to_array(), rho)


###


def check_consistent(vstate, mpi_size):
    assert vstate.n_samples == vstate.n_samples_per_rank * mpi_size
    assert vstate.n_samples == vstate.chain_length * vstate.sampler.n_chains


# There is no `MCMixedState.n_samples_diag_per_rank`, so we don't need MPI here
def check_consistent_diag(vstate):
    assert (
        vstate.n_samples_diag
        == vstate.chain_length_diag * vstate.sampler_diag.n_chains
        == vstate.diagonal.chain_length * vstate.diagonal.sampler.n_chains
    )


@common.skipif_mpi
@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(
            op,
            id=name,
        )
        for name, op in superoperators.items()
    ],
)
def test_expect_exact(vstate, operator):
    # Use lots of samples
    vstate.n_samples = 2 * 1e5
    vstate.n_discard_per_chain = 1e3

    # sample the expectation value and gradient with tons of samples
    O_stat = vstate.expect(operator)

    O_mean = np.asarray(O_stat.mean)
    err = 5 * O_stat.error_of_mean

    # check that vstate.expect gives the right result
    O_expval_exact = _expval(
        vstate.parameters, vstate, operator.to_sparse(), real=operator.is_hermitian
    )

    np.testing.assert_allclose(O_expval_exact.real, O_mean.real, atol=err, rtol=err)
    if not operator.is_hermitian:
        np.testing.assert_allclose(O_expval_exact.imag, O_mean.imag, atol=err, rtol=err)


# This test is 'broken' on CI when running under pytest-xdist. It does pass with
# pytest -n0 . I cannot reproduce locally, and I suspect it's a bug in Jax itself
# so I still include it in the local runs. Should be tested in a while to see if
# Jax fixed this bug.
@common.skipif_ci
@common.skipif_mpi
@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(
            op,
            id=name,
        )
        for name, op in superoperators.items()
    ],
)
def test_grad_finitedifferences(vstate, operator):
    op_sparse = operator.to_sparse()

    # Use lots of samples
    vstate.n_samples = 5 * 1e5
    vstate.n_discard_per_chain = 1e3

    # sample the expectation value and gradient with tons of samples
    O_stat1 = vstate.expect(operator)
    O_stat, O_grad = vstate.expect_and_grad(operator)

    O1_mean = np.asarray(O_stat1.mean)
    O_mean = np.asarray(O_stat.mean)
    err = 5 * O_stat1.error_of_mean

    # Check that expect and expect_and_grad give same expect. value
    assert O1_mean.real == approx(O_mean.real, abs=1e-5)
    if not operator.is_hermitian:
        assert O1_mean.imag == approx(O_mean.imag, abs=1e-5)

    np.testing.assert_allclose(O_stat1.variance, O_stat.variance, atol=1e-5)

    # Prepare the exact estimations
    pars_0 = vstate.parameters
    pars, unravel = nk.jax.tree_ravel(pars_0)

    def expval_fun(par, vstate, H):
        return _expval(unravel(par), vstate, H, real=operator.is_hermitian)

    # Compute the gradient with exact formula
    grad_exact = central_diff_grad(expval_fun, pars, 1.0e-5, vstate, op_sparse)

    if not operator.is_hermitian:
        grad_exact = jax.tree_util.tree_map(lambda x: x * 2, grad_exact)

    O_grad, _ = nk.jax.tree_ravel(O_grad)
    same_derivatives(O_grad, grad_exact, abs_eps=err, rel_eps=err)
