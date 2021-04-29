import pytest

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from jax.scipy.sparse.linalg import cg
from netket.optimizer.sr import _sr_onthefly_logic
from functools import partial
import itertools
from numpy import testing

import jax

from netket.optimizer import sr
import netket as nk

from .. import common

SR_objects = {}

SR_objects["CG()"] = sr.SRLazyCG()
SR_objects["CG(centered=True)"] = sr.SRLazyCG(centered=True)
SR_objects["CG(maxiter=3)"] = sr.SRLazyCG(maxiter=3)

SR_objects["GMRES()"] = sr.SRLazyGMRES()
SR_objects["GMRES(restart=5)"] = sr.SRLazyGMRES(restart=5)
SR_objects["GMRES(solve_method=incremental)"] = sr.SRLazyGMRES(
    solve_method="incremental"
)

dtypes = {"float": float, "complex": complex}


@pytest.fixture(params=[pytest.param(dtype, id=name) for name, dtype in dtypes.items()])
def vstate(request):
    N = 8
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
    "sr",
    [pytest.param(sr, id=name) for name, sr in SR_objects.items()],
)
def test_sr_solve(sr, vstate, _mpi_size, _mpi_rank):
    S = vstate.quantum_geometric_tensor(sr)
    x, _ = S.solve(vstate.parameters)

    if _mpi_size > 1:
        # other check
        with common.netket_disable_mpi():
            import mpi4jax

            samples, _ = mpi4jax.allgather(vstate.samples, comm=nk.utils.MPI_jax_comm)
            assert samples.shape == (_mpi_size, *vstate.samples.shape)
            vstate._samples = samples.reshape((-1, *vstate.samples.shape[1:]))

            S = vstate.quantum_geometric_tensor(sr)
            x_all, _ = S.solve(vstate.parameters)

            jax.tree_multimap(lambda a, b: np.testing.assert_allclose(a, b), x, x_all)


@pytest.mark.parametrize(
    "sr",
    [pytest.param(sr, id=name) for name, sr in SR_objects.items()],
)
def test_sr_matmul(sr, vstate, _mpi_size, _mpi_rank):
    S = vstate.quantum_geometric_tensor(sr)
    x = S @ vstate.parameters

    if _mpi_size > 1:
        # other check
        with common.netket_disable_mpi():
            import mpi4jax

            samples, _ = mpi4jax.allgather(vstate.samples, comm=nk.utils.MPI_jax_comm)
            assert samples.shape == (_mpi_size, *vstate.samples.shape)
            vstate._samples = samples.reshape((-1, *vstate.samples.shape[1:]))

            S = vstate.quantum_geometric_tensor(sr)
            x_all = S @ vstate.parameters

            jax.tree_multimap(lambda a, b: np.testing.assert_allclose(a, b), x, x_all)
