import pytest

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from jax.scipy.sparse.linalg import cg
from netket.optimizer.qgt import qgt_onthefly_logic as _sr_onthefly_logic
from functools import partial
import itertools
from numpy import testing

import jax

from netket.optimizer import qgt
import netket as nk

from .. import common

QGT_objects = {}

QGT_objects["OnTheFly"] = qgt.QGTOnTheFly

solvers = {}
solvers["gmres"] = jax.scipy.sparse.linalg.gmres

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
    x = S @ vstate.parameters

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
            x_all = S @ vstate.parameters

            jax.tree_multimap(lambda a, b: np.testing.assert_allclose(a, b), x, x_all)


# TODO: this test only tests r2r and holo, but should also do r2c.
# to add in a future rewrite
@pytest.mark.parametrize(
    "solver",
    [pytest.param(solver, id=name) for name, solver in solvers.items()],
)
def test_srjacobian_solve(vstate, solver, _mpi_size, _mpi_rank):
    if vstate.model.dtype is float:
        qgtT = partial(qgt.QGTJacobian, mode="R2R")
    else:
        qgtT = partial(qgt.QGTJacobian, mode="holomorphic")

    S = qgtT(vstate)
    x, _ = S.solve(solver, vstate.parameters)

    if _mpi_size > 1:
        # other check
        with common.netket_disable_mpi():
            import mpi4jax

            samples, _ = mpi4jax.allgather(vstate.samples, comm=nk.utils.MPI_jax_comm)
            assert samples.shape == (_mpi_size, *vstate.samples.shape)
            vstate._samples = samples.reshape((-1, *vstate.samples.shape[1:]))

            S = qgtT(vstate)
            x_all, _ = S.solve(solver, vstate.parameters)

            jax.tree_multimap(lambda a, b: np.testing.assert_allclose(a, b), x, x_all)


# TODO: this test only tests r2r and holo, but should also do r2c.
# to add in a future rewrite
def test_srjacobian_matmul(vstate, _mpi_size, _mpi_rank):
    if vstate.model.dtype is float:
        qgtT = partial(qgt.QGTJacobian, mode="R2R")
    else:
        qgtT = partial(qgt.QGTJacobian, mode="holomorphic")

    S = qgtT(vstate)
    x = S @ vstate.parameters

    if _mpi_size > 1:
        # other check
        with common.netket_disable_mpi():
            import mpi4jax

            samples, _ = mpi4jax.allgather(vstate.samples, comm=nk.utils.MPI_jax_comm)
            assert samples.shape == (_mpi_size, *vstate.samples.shape)
            vstate._samples = samples.reshape((-1, *vstate.samples.shape[1:]))

            S = qgtT(vstate)
            x_all = S @ vstate.parameters

            jax.tree_multimap(lambda a, b: np.testing.assert_allclose(a, b), x, x_all)


# TODO add to_dense tests
