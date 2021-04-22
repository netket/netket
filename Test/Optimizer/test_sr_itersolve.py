import pytest

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from jax.scipy.sparse.linalg import cg
from netket.optimizer.sr import _sr_onthefly_logic
from functools import partial
import itertools

from netket.optimizer import sr
import netket as nk

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

    return nk.variational.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )


@pytest.mark.parametrize(
    "sr",
    [pytest.param(sr, id=name) for name, sr in SR_objects.items()],
)
def test_sr_solve(sr, vstate):
    S = vstate.quantum_geometric_tensor(sr)
    x, _ = S.solve(vstate.parameters)
