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
from pytest import raises

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from jax.nn.initializers import normal

from netket.optimizer.linear_operator import LinearOperator

from .. import common

SEED = 2148364

machines = {}

standard_init = normal()
RBM = partial(
    nk.models.RBM, hidden_bias_init=standard_init, visible_bias_init=standard_init
)
RBMModPhase = partial(nk.models.RBMModPhase, hidden_bias_init=standard_init)

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
machines["model:(C->C,alpha=5)"] = RBM(
    alpha=5,
    param_dtype=complex,
    kernel_init=normal(stddev=0.3),
    hidden_bias_init=normal(stddev=0.3),
)

qgt_types = {
    "QGTJacobianDense": nk.optimizer.qgt.QGTJacobianDense(),
    "QGTJacobianPyTree": nk.optimizer.qgt.QGTJacobianPyTree(),
}

operators = {}

L = 4
g = nk.graph.Hypercube(length=L, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=L)

operators["operator:(Hermitian Real)"] = nk.operator.Ising(hi, graph=g, h=1.0)

H = nk.operator.Ising(hi, graph=g, h=1.0, dtype=complex)
for i in range(H.hilbert.size):
    H += nk.operator.spin.sigmay(H.hilbert, i)

operators["operator:(Hermitian Complex)"] = H


QGT_objects = {}

QGT_objects["OnTheFly"] = nk.optimizer.qgt.QGTOnTheFly

QGT_objects["JacobianPyTree"] = nk.optimizer.qgt.QGTJacobianPyTree


@common.skipif_mpi
@pytest.fixture(params=[pytest.param(ma, id=name) for name, ma in machines.items()])
def vstate(request):
    ma = request.param

    return nk.vqs.FullSumState(hi, ma)


@common.skipif_mpi
def test_init_parameters(vstate):
    vstate.init_parameters(seed=SEED)
    pars = vstate.parameters
    vstate.init_parameters(normal(stddev=0.01), seed=SEED)
    pars2 = vstate.parameters

    def _f(x, y):
        np.testing.assert_allclose(x, y)

    jax.tree_map(_f, pars, pars2)


@common.skipif_mpi
def test_basic_methods(vstate):
    key1, key2 = jax.random.split(nk.jax.PRNGKey())
    s = vstate.hilbert.random_state(key1)
    assert np.shape(vstate.log_value(s)) == ()

    s = vstate.hilbert.random_state(key2, size=2)
    assert np.shape(vstate.log_value(s)) == (2,)


@common.skipif_mpi
@pytest.mark.parametrize(
    "qgtT", [pytest.param(qgtT, id=name) for name, qgtT in qgt_types.items()]
)
def test_qgt_construction(vstate, qgtT):
    qgt = vstate.quantum_geometric_tensor(qgtT)
    assert isinstance(qgt, LinearOperator)


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
    np.testing.assert_allclose(q_obj.data.todense(), ket.reshape(q_obj.shape))


@pytest.mark.parametrize(
    "machine", [pytest.param(ma, id=name) for name, ma in machines.items()]
)
def test_derivatives_agree(machine):
    err = 1e-3
    g = nk.graph.Chain(length=8, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=1)
    vs = nk.vqs.FullSumState(hi, machine)

    e_expect = vs.expect(ha)
    assert isinstance(e_expect, nk.stats.Stats)
    np.testing.assert_almost_equal(e_expect.error_of_mean, 0)

    e, grads_exact = vs.expect_and_grad(ha)
    assert isinstance(e, nk.stats.Stats)
    np.testing.assert_almost_equal(e.error_of_mean, 0)

    # check that energies match
    np.testing.assert_almost_equal(e_expect.mean, e.mean)
    np.testing.assert_almost_equal(e_expect.variance, e.variance)

    # Prepare the exact estimations
    pars_0 = vs.parameters
    pars, unravel = nk.jax.tree_ravel(pars_0)
    op_sparse = ha.to_sparse()

    def expval_fun(par, vstate, H):
        return _expval(unravel(par), vstate, H)

    grad_finite = central_diff_grad(expval_fun, pars, 1.0e-5, vs, op_sparse)

    O_grad, _ = nk.jax.tree_ravel(grads_exact)
    same_derivatives(O_grad, grad_finite, abs_eps=err, rel_eps=err)


###
def _expval(par, vs, H):
    vs.parameters = par
    psi = vs.to_array()
    expval = psi.conj() @ (H @ psi)

    return np.real(expval)


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
        if jnp.iscomplexobj(x):
            grad_i = 0.5 * (func(x + 1j * epsd, *args) - func(x - 1j * epsd, *args))
            grad[i] = 0.5 * grad_r + 0.5j * grad_i
        else:
            # grad_i = 0.0
            grad[i] = grad_r

        assert not np.isnan(grad[i])
        grad[i] /= eps
        epsd = np.roll(epsd, 1)
    return grad


@common.skipif_mpi
@pytest.mark.parametrize(
    "L,n_iterations,h",
    [(4, 100, 1), (6, 100, 2)],
)
@pytest.mark.parametrize(
    "machine", [pytest.param(ma, id=name) for name, ma in machines.items()]
)
@pytest.mark.parametrize(
    "qgtT", [pytest.param(qgtT, id=name) for name, qgtT in qgt_types.items()]
)
def test_TFIM_energy_strictly_decreases(
    L,
    n_iterations,
    h,
    machine,
    qgtT,
    abs_eps=1.0e-3,
    rel_eps=1.0e-4,
):
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=h)
    vs = nk.vqs.FullSumState(hi, machine)

    op = nk.optimizer.Sgd(learning_rate=0.003)

    gs = nk.driver.VMC(
        ha,
        op,
        variational_state=vs,
        preconditioner=nk.optimizer.SR(qgt=qgtT),
    )

    log = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iterations, out=log, show_progress=False)

    energies = log.data["Energy"]["Mean"]

    for i in range(len(energies) - 1):
        assert energies[i + 1] < energies[i]


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


@common.skipif_mpi
def test_chunk_size_api(vstate, _mpi_size):
    assert vstate.chunk_size is None

    with raises(
        ValueError,
    ):
        vstate.chunk_size = -1

    # does not divide hi.n_states
    with raises(
        ValueError,
    ):
        vstate.chunk_size = 3

    assert vstate.chunk_size is None

    vstate.chunk_size = vstate.hilbert.n_states // 4
    assert vstate.chunk_size == vstate.hilbert.n_states // 4


@pytest.mark.parametrize(
    "qgt", [pytest.param(qgt, id=name) for name, qgt in QGT_objects.items()]
)
@pytest.mark.parametrize("n_chunks", [1, 2])
def test_qgt_chunking(vstate, qgt, n_chunks):
    chunk_size = vstate.hilbert.n_states // n_chunks

    vec = vstate.parameters
    S_nonchunk = vstate.quantum_geometric_tensor(qgt)
    eval_nochunk = S_nonchunk @ vec
    vstate.chunk_size = chunk_size
    S_chunk = vstate.quantum_geometric_tensor(qgt)
    eval_chunk = S_chunk @ vec

    jax.tree_map(
        partial(np.testing.assert_allclose, atol=1e-13), eval_nochunk, eval_chunk
    )
