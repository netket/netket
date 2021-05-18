import pytest

import itertools
from functools import partial


import flax
import jax

import numpy as np
from numpy import testing

import jax.numpy as jnp
import jax.flatten_util
from jax.scipy.sparse.linalg import cg

import netket as nk
from netket.optimizer import qgt
from netket.optimizer.qgt import qgt_onthefly_logic as _sr_onthefly_logic

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

solvers["gmres"] = jax.scipy.sparse.linalg.gmres
solvers_tol[solvers["gmres"]] = 1e-5
solvers["cholesky"] = nk.optimizer.solver.cholesky
solvers_tol[solvers["cholesky"]] = 1e-8


dtypes = {"float": float, "complex": complex}


@pytest.fixture(params=[pytest.param(dtype, id=name) for name, dtype in dtypes.items()])
def vstate(request):
    N = 5
    hi = nk.hilbert.Spin(1 / 2, N)
    g = nk.graph.Chain(N)

    dtype = request.param

    ma = nk.models.RBM(
        alpha=1,
        dtype=dtype,
        hidden_bias_init=nk.nn.initializers.normal(),
        visible_bias_init=nk.nn.initializers.normal(),
    )

    vstate = nk.variational.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )
    vstate.init_parameters(
        nk.nn.initializers.normal(stddev=0.001), seed=jax.random.PRNGKey(3)
    )

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

            jax.tree_multimap(lambda a, b: np.testing.assert_allclose(a, b), x, x_all)


@pytest.mark.parametrize(
    "qgt",
    [pytest.param(sr, id=name) for name, sr in QGT_objects.items()],
)
def test_qgt_matmul(qgt, vstate, _mpi_size, _mpi_rank):
    S = qgt(vstate)
    y = vstate.parameters
    x = S @ y

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
def test_qgt_dense(qgt, vstate, _mpi_size, _mpi_rank):
    S = qgt(vstate)

    Sd = S.to_dense()

    assert Sd.ndim == 2
    if hasattr(S, "mode"):
        if S.mode == "complex":
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

            np.testing.assert_allclose(Sd_all, Sd, rtol=1e-5, atol=1e-17)
