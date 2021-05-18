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

QGT_objects["JacobianPyTree"] = partial(qgt.QGTJacobianPyTree, diag_shift=0.00)

solvers = {}
solvers["svd"] = nk.optimizer.solver.svd
solvers["cholesky"] = nk.optimizer.solver.cholesky
solvers["LU"] = nk.optimizer.solver.LU
solvers["solve"] = nk.optimizer.solver.solve

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
