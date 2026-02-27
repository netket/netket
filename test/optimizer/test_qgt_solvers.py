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

import pytest

from functools import partial

import numpy as np

import jax
import jax.flatten_util
import jax.numpy as jnp
from jax.nn.initializers import normal

import netket as nk
from netket.optimizer import qgt
from netket.optimizer.solver import cholesky, pinv_smooth, nan_fallback

from .. import common  # noqa: F401

QGT_types = {}
QGT_types["QGTOnTheFly"] = nk.optimizer.qgt.QGTOnTheFly
QGT_types["QGTJacobianDense"] = nk.optimizer.qgt.QGTJacobianDense
QGT_types["QGTJacobianPyTree"] = nk.optimizer.qgt.QGTJacobianPyTree

QGT_objects = {}

QGT_objects["JacobianPyTree"] = partial(qgt.QGTJacobianPyTree, diag_shift=0.00)

solvers = {}
solvers["svd"] = nk.optimizer.solver.svd
solvers["cholesky"] = nk.optimizer.solver.cholesky
solvers["LU"] = nk.optimizer.solver.LU
solvers["solve"] = nk.optimizer.solver.solve
solvers["pinv"] = nk.optimizer.solver.pinv
solvers["pinv_smooth"] = nk.optimizer.solver.pinv_smooth

dtypes = {"float": float, "complex": complex}


@pytest.fixture(params=[pytest.param(dtype, id=name) for name, dtype in dtypes.items()])
def vstate(request):
    N = 5
    hi = nk.hilbert.Spin(1 / 2, N)

    dtype = request.param

    ma = nk.models.RBM(
        alpha=1,
        param_dtype=dtype,
        hidden_bias_init=normal(),
        visible_bias_init=normal(),
    )

    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )
    vstate.init_parameters(normal(stddev=0.001), seed=jax.random.PRNGKey(3))

    vstate.sample()

    return vstate


@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
@pytest.mark.parametrize(
    "solver",
    [pytest.param(solver, id=name) for name, solver in solvers.items()],
)
def test_qgt_solve(qgt, vstate, solver):
    is_holo = nk.jax.is_complex_dtype(vstate.model.param_dtype)
    S = qgt(vstate, holomorphic=is_holo)

    x, _ = S.solve(solver, vstate.parameters)


# Issue #789 https://github.com/netket/netket/issues/789
# cannot multiply real qgt by complex vector
@common.skipif_mpi
@pytest.mark.parametrize(
    "SType", [pytest.param(T, id=name) for name, T in QGT_types.items()]
)
def test_qgt_throws(SType):
    hi = nk.hilbert.Spin(s=1 / 2, N=5)
    ma = nk.models.RBMModPhase(alpha=1, param_dtype=float)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16, reset_chains=False)
    vs = nk.vqs.MCState(sa, ma, n_samples=100, n_discard_per_chain=100)

    S = vs.quantum_geometric_tensor(SType)
    g_cmplx = jax.tree_util.tree_map(lambda x: x + x * 0.1j, vs.parameters)

    with pytest.raises(
        nk.errors.RealQGTComplexDomainError, match="Cannot multiply the"
    ):
        S @ g_cmplx


@common.skipif_mpi
@pytest.mark.parametrize(
    "SType", [pytest.param(T, id=name) for name, T in QGT_types.items()]
)
def test_qgt_nondiff_sigma(SType):
    # Test that we dont attempt to differentiate through the samples
    # by testing a model that would fail in that case because its
    # nondifferentiable.

    hi = nk.hilbert.Spin(s=1 / 2, N=5)
    ma = nk.models.LogStateVector(hi)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=2, reset_chains=False)
    vs = nk.vqs.MCState(sa, ma, n_samples=2, n_discard_per_chain=0)

    S = SType(vs, holomorphic=True)
    S @ vs.parameters


@common.skipif_mpi
def test_qgt_otf_scale_err():
    N = 5
    hi = nk.hilbert.Spin(1 / 2, N)
    ma = nk.models.RBM()
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )
    with pytest.raises(NotImplementedError):
        nk.optimizer.qgt.QGTOnTheFly(vstate, diag_scale=0.01)


@pytest.mark.parametrize(
    "SType", [pytest.param(T, id=name) for name, T in QGT_types.items()]
)
def test_qgt_explicit_chunk_size(SType):
    hi = nk.hilbert.Spin(s=1 / 2, N=5)
    ma = nk.models.RBMModPhase(alpha=1, param_dtype=float)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16, reset_chains=False)
    vs = nk.vqs.MCState(sa, ma, n_samples=16 * 8, n_discard_per_chain=100)

    SType(vs, chunk_size=16 * 4)


@pytest.mark.parametrize(
    "solver",
    [pytest.param(solver, id=name) for name, solver in solvers.items()],
)
def test_solver_kwargs_partial_api(solver):
    # create the partial interface
    solver_partial = solver(x0=None)
    assert isinstance(solver_partial, partial)
    assert len(solver_partial.args) == 0
    assert solver_partial.keywords == {"x0": None}


@pytest.mark.parametrize(
    "solver",
    [pytest.param(solver, id=name) for name, solver in solvers.items()],
)
def test_solver_dense_api(solver):
    A = jax.random.normal(jax.random.key(1), (10, 10))
    A = A @ A.conj().T
    b = jax.random.normal(jax.random.key(2), (10,))

    x, _ = solver(A, b)
    np.testing.assert_allclose(A @ x, b)


# Not all solvers had 'rcond' deprecated, so we test only for pinv and pinv_smooth.
@pytest.mark.parametrize("solver_func", [solvers["pinv"], solvers["pinv_smooth"]])
def test_solver_rcond_deprecation(solver_func):
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (10, 10))
    A = A @ A.T  # Make PSD
    b = jax.random.normal(key, (10,))

    # Test with new argument (rtol)
    x_new, _ = solver_func(A, b, rtol=1e-6)

    with pytest.warns(FutureWarning, match="'rcond' argument is deprecated"):
        x_old, _ = solver_func(A, b, rcond=1e-6)

    # Check that results are identical
    np.testing.assert_allclose(x_new, x_old, rtol=1e-10)

    # Check that the solution is correct
    np.testing.assert_allclose(A @ x_new, b, rtol=1e-6)


# ---------------------------------------------------------------------------
# nan_fallback tests
# ---------------------------------------------------------------------------


def _make_psd(n, key):
    A_raw = jax.random.normal(key, (n, n))
    return A_raw @ A_raw.T + n * jnp.eye(n)


def _make_illconditioned(n, key):
    evals = jnp.concatenate([jnp.ones(3), jnp.full(n - 3, 1e-20)])
    vecs = jnp.linalg.qr(jax.random.normal(key, (n, n)))[0]
    return vecs @ jnp.diag(evals) @ vecs.T


def test_nan_fallback_no_fallback_under_jit():
    """Primary solver succeeds: solver_fallback is False and result is correct."""
    n = 8
    A = _make_psd(n, jax.random.PRNGKey(0))
    b = jax.random.normal(jax.random.PRNGKey(1), (n,))

    solver = nan_fallback(cholesky, pinv_smooth)
    x, info = jax.jit(solver)(A, b)

    assert not info["solver_fallback"]
    np.testing.assert_allclose(A @ x, b, rtol=1e-5)


def test_nan_fallback_triggered_by_output_nan_under_jit():
    """Ill-conditioned A causes cholesky to produce NaN/Inf; pinv_smooth recovers."""
    n = 8
    A = _make_illconditioned(n, jax.random.PRNGKey(0))
    b = jax.random.normal(jax.random.PRNGKey(1), (n,))

    solver = nan_fallback(cholesky, pinv_smooth)
    x, info = jax.jit(solver)(A, b)

    assert info["solver_fallback"]
    assert not jnp.any(jnp.isnan(x))

    # Result should match pinv_smooth directly
    x_ref, _ = jax.jit(pinv_smooth)(A, b)
    np.testing.assert_allclose(x, x_ref, rtol=1e-5)


def test_nan_fallback_pytree_b_under_jit():
    """Works correctly when b is a pytree (dict of arrays)."""
    n = 8
    A = _make_psd(n, jax.random.PRNGKey(0))
    b_flat = jax.random.normal(jax.random.PRNGKey(1), (n,))
    b_tree = {"a": b_flat[:4], "c": b_flat[4:]}

    solver = nan_fallback(cholesky, pinv_smooth)
    x_tree, info = jax.jit(solver)(A, b_tree)

    assert not info["solver_fallback"]
    assert set(x_tree.keys()) == {"a", "c"}
    x_rec = jnp.concatenate([x_tree["a"], x_tree["c"]])
    np.testing.assert_allclose(A @ x_rec, b_flat, rtol=1e-5)


def test_nan_fallback_fallback_not_called_when_primary_succeeds():
    """Fallback body does not execute at runtime when the primary is healthy."""
    n = 8
    A = _make_psd(n, jax.random.PRNGKey(0))
    b = jax.random.normal(jax.random.PRNGKey(1), (n,))

    call_log = []

    def counting_fallback(A, b, *, x0=None):
        jax.debug.callback(lambda: call_log.append(1))
        return pinv_smooth(A, b, x0=x0)

    solver = nan_fallback(cholesky, counting_fallback)
    jax.jit(solver)(A, b)

    assert len(call_log) == 0, "fallback should not have run"


def test_nan_fallback_equality_and_hash():
    """Two nan_fallback calls with the same solvers are equal and hash-equal."""
    s1 = nan_fallback(cholesky, pinv_smooth)
    s2 = nan_fallback(cholesky, pinv_smooth)

    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert s1 is not s2  # distinct objects
