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

QGT_objects["JacobianPyTree"] = partial(qgt.QGTJacobianPyTree, diag_shift=0.01)
QGT_objects["JacobianPyTree(mode=holomorphic)"] = partial(
    qgt.QGTJacobianPyTree, holomorphic=True, diag_shift=0.01
)
QGT_objects["JacobianPyTree(rescale_shift=True)"] = partial(
    qgt.QGTJacobianPyTree, rescale_shift=True, diag_shift=0.01
)

QGT_objects["JacobianDense"] = partial(qgt.QGTJacobianDense, diag_shift=0.01)
QGT_objects["JacobianDense(mode=holomorphic)"] = partial(
    qgt.QGTJacobianDense, holomorphic=True, diag_shift=0.01
)
QGT_objects["JacobianDense(rescale_shift=True)"] = partial(
    qgt.QGTJacobianDense, rescale_shift=True, diag_shift=0.01
)

solvers = {}
solvers_tol = {}

solvers["gmres"] = partial(jax.scipy.sparse.linalg.gmres, tol=1e-6)
solvers_tol[solvers["gmres"]] = 4e-4
solvers["cholesky"] = nk.optimizer.solver.cholesky
solvers_tol[solvers["cholesky"]] = 1e-8


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
    "RBM[dtype=float]": partial(RBM, dtype=float),
    "RBM[dtype=complex]": partial(RBM, dtype=complex),
    "RBMModPhase[dtype=float]": partial(RBMModPhase, dtype=float),
}

dtypes = {"float": float, "complex": complex}


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

    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        model,
    )
    vstate.init_parameters(normal(stddev=0.001), seed=jax.random.PRNGKey(3))

    vstate.sample()

    vstate.chunk_size = chunk_size

    return vstate


@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
@pytest.mark.parametrize(
    "solver",
    [pytest.param(solver, id=name) for name, solver in solvers.items()],
)
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_qgt_solve(qgt, vstate, solver, _mpi_size, _mpi_rank):
    S = qgt(vstate)
    x, _ = S.solve(solver, vstate.parameters)

    jax.tree_multimap(
        partial(testing.assert_allclose, rtol=solvers_tol[solver]),
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

            jax.tree_multimap(
                lambda a, b: np.testing.assert_allclose(a, b, rtol=0.00045), x, x_all
            )


@pytest.mark.skipif_mpi
@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
@pytest.mark.parametrize("chunk_size", [None])
def test_qgt_solve_with_x0(qgt, vstate):
    solver = jax.scipy.sparse.linalg.gmres
    x0 = jax.tree_map(jnp.zeros_like, vstate.parameters)

    S = qgt(vstate)
    x, _ = S.solve(solver, vstate.parameters, x0=x0)


@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_qgt_matmul(qgt, vstate, _mpi_size, _mpi_rank):
    S = qgt(vstate)
    rng = nkjax.PRNGSeq(0)
    y = jax.tree_map(
        lambda x: 0.001 * jax.random.normal(rng.next(), x.shape, dtype=x.dtype),
        vstate.parameters,
    )
    x = S @ y

    def check_same_dtype(x, y):
        assert x.dtype == y.dtype

    jax.tree_multimap(check_same_dtype, x, y)

    # test multiplication by dense gives same result...
    y_dense, unravel = nk.jax.tree_ravel(y)
    x_dense = S @ y_dense
    x_dense_unravelled = unravel(x_dense)

    jax.tree_multimap(
        lambda a, b: np.testing.assert_allclose(a, b), x, x_dense_unravelled
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

            jax.tree_multimap(lambda a, b: np.testing.assert_allclose(a, b), x, x_all)


@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_qgt_dense(qgt, vstate, _mpi_size, _mpi_rank):
    S = qgt(vstate)

    # test repr
    str(S)

    Sd = S.to_dense()

    assert Sd.ndim == 2
    if hasattr(S, "mode"):
        if S.mode == "complex" and np.issubdtype(
            vstate.model.dtype, np.complexfloating
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

            np.testing.assert_allclose(Sd_all, Sd, rtol=1e-5, atol=1e-15)


@pytest.mark.skipif_mpi
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
    v = vstate.parameters
    S = qgt(vstate)
    expected = S @ v
    diag_shift = S.diag_shift
    if isinstance(S, (QGTJacobianPyTreeT, QGTJacobianDenseT)):
        # extract the necessary shape for the diag_shift
        t = jax.eval_shape(partial(jax.tree_map, lambda x: x[0], S.O))
    else:
        t = v
    diag_shift_tree = jax.tree_map(
        lambda x: diag_shift * jnp.ones(x.shape, dtype=x.dtype), t
    )
    S = S.replace(diag_shift=diag_shift_tree)
    res = S @ v
    jax.tree_multimap(lambda a, b: np.testing.assert_allclose(a, b), res, expected)
