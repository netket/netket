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
import jax.flatten_util
from jax.nn.initializers import normal

import netket as nk
from netket.optimizer import qgt

from .. import common  # noqa: F401

QGT_types = {}
QGT_types["QGTOnTheFly"] = nk.optimizer.qgt.QGTOnTheFly
# QGT_types["QGTJacobianDense"] = nk.optimizer.qgt.QGTJacobianDense
QGT_types["QGTJacobianPyTree"] = nk.optimizer.qgt.QGTJacobianPyTree

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

    dtype = request.param

    ma = nk.models.RBM(
        alpha=1,
        dtype=dtype,
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
def test_qgt_solve(qgt, vstate, solver, _mpi_size, _mpi_rank):
    S = qgt(vstate)

    x, _ = S.solve(solver, vstate.parameters)


# Issue #789 https://github.com/netket/netket/issues/789
# cannot multiply real qgt by complex vector
@common.skipif_mpi
@pytest.mark.parametrize(
    "SType", [pytest.param(T, id=name) for name, T in QGT_types.items()]
)
def test_qgt_throws(SType):
    hi = nk.hilbert.Spin(s=1 / 2, N=5)
    ma = nk.models.RBMModPhase(alpha=1, dtype=float)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16, reset_chains=False)
    vs = nk.vqs.MCState(sa, ma, n_samples=100, n_discard_per_chain=100)

    S = vs.quantum_geometric_tensor(SType)
    g_cmplx = jax.tree_map(lambda x: x + x * 0.1j, vs.parameters)

    with pytest.raises(TypeError, match="Cannot multiply the"):
        S @ g_cmplx
