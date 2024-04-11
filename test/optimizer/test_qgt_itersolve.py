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

import jax
import jax.numpy as jnp
import numpy as np
from numpy import testing
import jax.flatten_util
from jax.nn.initializers import normal

import netket as nk
import netket.jax as nkjax
from netket.optimizer import qgt

from netket.optimizer.qgt.qgt_jacobian_pytree import QGTJacobianPyTreeT
from netket.optimizer.qgt.qgt_jacobian_dense import QGTJacobianDenseT

from .. import common

QGT_objects = {}

QGT_objects["OnTheFly"] = partial(qgt.QGTOnTheFly, diag_shift=0.01)
QGT_objects["OnTheFly"] = partial(qgt.QGTOnTheFly, diag_shift=0.01, holomorphic=True)

QGT_objects["JacobianPyTree"] = partial(qgt.QGTJacobianPyTree, diag_shift=0.01)
QGT_objects["JacobianPyTree(mode=holomorphic)"] = partial(
    qgt.QGTJacobianPyTree, holomorphic=True, diag_shift=0.01
)
QGT_objects["JacobianPyTree(diag_scale=0.01)"] = partial(
    qgt.QGTJacobianPyTree, diag_scale=0.01, diag_shift=0.0
)
QGT_objects["JacobianPyTree(diag_scale=0.01, diag_shift=0.01)"] = partial(
    qgt.QGTJacobianPyTree, diag_scale=0.01, diag_shift=0.01
)

QGT_objects["JacobianDense"] = partial(qgt.QGTJacobianDense, diag_shift=0.01)
QGT_objects["JacobianDense(mode=holomorphic)"] = partial(
    qgt.QGTJacobianDense, holomorphic=True, diag_shift=0.01
)
QGT_objects["JacobianDense(diag_scale=0.01)"] = partial(
    qgt.QGTJacobianDense, diag_scale=0.01, diag_shift=0.0
)
QGT_objects["JacobianDense(diag_scale=0.01, diag_shift=0.01)"] = partial(
    qgt.QGTJacobianDense, diag_scale=0.01, diag_shift=0.01
)

solvers = {}
solvers_tol = {}

solvers["gmres"] = partial(jax.scipy.sparse.linalg.gmres, tol=1e-6)
solvers_tol[solvers["gmres"], np.dtype("float64")] = 5e-4, 0
solvers_tol[solvers["gmres"], np.dtype("float32")] = 1e-2, 1e-4
solvers["cholesky"] = nk.optimizer.solver.cholesky
solvers_tol[solvers["cholesky"], np.dtype("float64")] = 1e-8, 0
solvers_tol[solvers["cholesky"], np.dtype("float32")] = 1e-2, 1e-4

matmul_tol = {}
matmul_tol[np.dtype("float64")] = 1e-7, 0
matmul_tol[np.dtype("float32")] = 5e-4, 0


dense_tol = {}
dense_tol[np.dtype("float64")] = 1e-5, 1e-15
dense_tol[np.dtype("float32")] = 5e-4, 1e-6


RBM = partial(
    nk.models.RBM,
    hidden_bias_init=normal(),
    visible_bias_init=normal(),
)
RBMModPhase = partial(
    nk.models.RBMModPhase,
    hidden_bias_init=normal(),
    kernel_init=normal(),
)

models = {
    "RBM[dtype=float64]": partial(RBM, param_dtype=np.dtype("float64")),
    "RBM[dtype=float32]": partial(RBM, param_dtype=np.dtype("float32")),
    "RBM[dtype=complex128]": partial(RBM, param_dtype=np.dtype("complex128")),
    "RBMModPhase[dtype=float64]": partial(RBMModPhase, param_dtype=np.dtype("float64")),
}


@pytest.fixture(
    params=[pytest.param(modelT, id=name) for name, modelT in models.items()]
)
def model(request):
    modelT = request.param

    return modelT(alpha=1)


@pytest.fixture
def vstate(request, model, chunk_size):
    N = 5
    hi = nk.hilbert.Spin(1 / 2, N)

    k = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(k)

    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        model,
        seed=k1,
        n_samples=1024,
    )

    # initialize the same parameters on every rank
    vstate.init_parameters(normal(stddev=0.001), seed=k2)

    vstate.sample()

    vstate.chunk_size = chunk_size

    return vstate


def is_complex_failing(vstate, qgt_partial):
    """
    returns true if this qgt should error on construction
    """
    if not jnp.issubdtype(vstate.model.param_dtype, jnp.complexfloating):
        if qgt_partial.keywords.get("holomorphic", False):
            return True
    return False


@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
@pytest.mark.parametrize(
    "solver",
    [pytest.param(solver, id=name) for name, solver in solvers.items()],
)
@pytest.mark.parametrize(
    "chunk_size", [pytest.param(x, id=f"chunk={x}") for x in [None, 16]]
)
def test_qgt_solve(qgt, vstate, solver, _mpi_size, _mpi_rank):
    if is_complex_failing(vstate, qgt):
        with pytest.raises(
            nk.errors.IllegalHolomorphicDeclarationForRealParametersError
        ):
            S = qgt(vstate)
        return
    else:
        S = qgt(vstate)

    x, _ = S.solve(solver, vstate.parameters)

    rtol, atol = solvers_tol[solver, nk.jax.dtype_real(vstate.model.param_dtype)]
    jax.tree_util.tree_map(
        partial(testing.assert_allclose, rtol=rtol, atol=atol),
        S @ x,
        vstate.parameters,
    )

    if _mpi_size > 1:
        # other check
        with common.netket_disable_mpi():
            import mpi4jax

            samples, _ = mpi4jax.allgather(
                vstate.samples, comm=nk.utils.mpi.MPI_jax_comm
            )
            assert samples.shape == (_mpi_size, *vstate.samples.shape)
            vstate._samples = samples.reshape((-1, *vstate.samples.shape[1:]))

            S = qgt(vstate)
            x_all, _ = S.solve(solver, vstate.parameters)

            jax.tree_util.tree_map(
                lambda a, b: np.testing.assert_allclose(a, b, rtol=rtol, atol=atol),
                x,
                x_all,
            )


@common.skipif_mpi
@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
@pytest.mark.parametrize("chunk_size", [None])
def test_qgt_solve_with_x0(qgt, vstate):
    if is_complex_failing(vstate, qgt):
        return

    solver = jax.scipy.sparse.linalg.gmres
    x0 = jax.tree_util.tree_map(jnp.zeros_like, vstate.parameters)

    S = qgt(vstate)
    x, _ = S.solve(solver, vstate.parameters, x0=x0)


@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
@pytest.mark.parametrize(
    "chunk_size", [pytest.param(x, id=f"chunk={x}") for x in [None, 16]]
)
def test_qgt_matmul(qgt, vstate, _mpi_size, _mpi_rank):
    if is_complex_failing(vstate, qgt):
        return

    rtol, atol = matmul_tol[nk.jax.dtype_real(vstate.model.param_dtype)]

    S = qgt(vstate)
    rng = nkjax.PRNGSeq(0)
    y = jax.tree_util.tree_map(
        lambda x: 0.001 * jax.random.normal(rng.next(), x.shape, dtype=x.dtype),
        vstate.parameters,
    )
    x = S @ y

    def check_same_dtype(x, y):
        assert x.dtype == y.dtype

    jax.tree_util.tree_map(check_same_dtype, x, y)

    # test multiplication by dense gives same result...
    y_dense, unravel = nk.jax.tree_ravel(y)
    x_dense = S @ y_dense
    x_dense_unravelled = unravel(x_dense)

    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=rtol, atol=atol),
        x,
        x_dense_unravelled,
    )

    if _mpi_size > 1:
        # other check
        with common.netket_disable_mpi():
            import mpi4jax

            samples, _ = mpi4jax.allgather(
                vstate.samples, comm=nk.utils.mpi.MPI_jax_comm
            )
            assert samples.shape == (_mpi_size, *vstate.samples.shape)
            vstate._samples = samples.reshape((-1, *vstate.samples.shape[1:]))

            S = qgt(vstate)
            x_all = S @ y

            jax.tree_util.tree_map(
                lambda a, b: np.testing.assert_allclose(a, b, rtol=rtol, atol=atol),
                x,
                x_all,
            )


@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
@pytest.mark.parametrize(
    "chunk_size", [pytest.param(x, id=f"chunk={x}") for x in [None, 16]]
)
def test_qgt_dense(qgt, vstate, _mpi_size, _mpi_rank):
    if is_complex_failing(vstate, qgt):
        return

    rtol, atol = dense_tol[nk.jax.dtype_real(vstate.model.param_dtype)]

    S = qgt(vstate)

    # test repr
    str(S)

    Sd = S.to_dense()

    assert Sd.ndim == 2
    if hasattr(S, "mode"):
        if S.mode == "complex" and np.issubdtype(
            vstate.model.param_dtype, np.complexfloating
        ):
            assert Sd.shape == (2 * vstate.n_parameters, 2 * vstate.n_parameters)
        else:
            assert Sd.shape == (vstate.n_parameters, vstate.n_parameters)
    else:
        assert Sd.shape == (vstate.n_parameters, vstate.n_parameters)

    if _mpi_size > 1:
        # other check
        with common.netket_disable_mpi():
            import mpi4jax

            samples, _ = mpi4jax.allgather(
                vstate.samples, comm=nk.utils.mpi.MPI_jax_comm
            )
            assert samples.shape == (_mpi_size, *vstate.samples.shape)
            vstate._samples = samples.reshape((-1, *vstate.samples.shape[1:]))

            S = qgt(vstate)
            Sd_all = S.to_dense()

            np.testing.assert_allclose(Sd_all, Sd, rtol=rtol, atol=atol)


@common.skipif_mpi
@pytest.mark.parametrize(
    "qgt", [pytest.param(sr, id=name) for name, sr in QGT_objects.items()]
)
@pytest.mark.parametrize(
    "chunk_size",
    [
        None,
    ],
)
def test_qgt_pytree_diag_shift(qgt, vstate):
    if is_complex_failing(vstate, qgt):
        return

    v = vstate.parameters
    S = qgt(vstate)
    expected = S @ v
    diag_shift = S.diag_shift
    if isinstance(S, (QGTJacobianPyTreeT, QGTJacobianDenseT)):
        # extract the necessary shape for the diag_shift
        if S.mode == "complex":
            t = jax.eval_shape(partial(jax.tree_util.tree_map, lambda x: x[0, 0], S.O))
        else:
            t = jax.eval_shape(partial(jax.tree_util.tree_map, lambda x: x[0], S.O))
    else:
        t = v
    diag_shift_tree = jax.tree_util.tree_map(
        lambda x: diag_shift * jnp.ones(x.shape, dtype=x.dtype), t
    )
    S = S.replace(diag_shift=diag_shift_tree)
    res = S @ v
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(a, b), res, expected)


@common.skipif_mpi
def test_qgt_holomorphic_real_pars_throws():
    hi = nk.hilbert.Spin(1 / 2, 5)
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        nk.models.RBM(param_dtype=float),
    )

    with pytest.raises(nk.errors.IllegalHolomorphicDeclarationForRealParametersError):
        vstate.quantum_geometric_tensor(qgt.QGTJacobianPyTree(holomorphic=True))
    with pytest.raises(nk.errors.IllegalHolomorphicDeclarationForRealParametersError):
        vstate.quantum_geometric_tensor(qgt.QGTJacobianDense(holomorphic=True))

    return vstate
