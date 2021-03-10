from functools import partial
from io import StringIO

import pytest
from pytest import approx, raises

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket import nn as nknn
import flax

from contextlib import redirect_stderr
import tempfile
import re


@pytest.mark.parametrize("diag", [False, True])
def test_mps(diag):
    L = 6
    g = nk.graph.Hypercube(length=L, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ma = nk.models.MPSPeriodic(hilbert=hi, graph=g, bond_dim=2, diag=diag)
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)

    vs = nk.variational.MCState(sa, ma, n_samples=1000)

    ha = nk.operator.Ising(hi, graph=g, h=1.0)
    op = nk.optimizer.Sgd(learning_rate=0.05)

    driver = nk.Vmc(ha, op, variational_state=vs)

    driver.run(3)
