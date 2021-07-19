import pytest
from pytest import raises

import numpy as np
import netket as nk

from .. import common

pytestmark = common.skipif_mpi

SEED = 214748364


def _setup_system():
    L = 3

    hi = nk.hilbert.Spin(s=0.5) ** L

    ha = nk.operator.LocalOperator(hi)
    j_ops = []
    for i in range(L):
        ha += (0.3 / 2.0) * nk.operator.spin.sigmax(hi, i)
        ha += (
            (2.0 / 4.0)
            * nk.operator.spin.sigmaz(hi, i)
            * nk.operator.spin.sigmaz(hi, (i + 1) % L)
        )
        j_ops.append(nk.operator.spin.sigmam(hi, i))

    #  Create the liouvillian
    lind = nk.operator.LocalLiouvillian(ha, j_ops)
    return hi, lind


def _setup_ss(dtype=np.float32, sr=True):
    hi, lind = _setup_system()

    ma = nk.models.NDM()
    # sa = nk.sampler.ExactSampler(hilbert=nk.hilbert.DoubledHilber(hi), n_chains=16)

    sa = nk.sampler.MetropolisLocal(hilbert=nk.hilbert.DoubledHilbert(hi))
    sa_obs = nk.sampler.MetropolisLocal(hilbert=hi)

    vs = nk.vqs.MCMixedState(sa, ma, sampler_diag=sa_obs, n_samples=1000, seed=SEED)

    op = nk.optimizer.Sgd(learning_rate=0.05)
    if sr:
        sr_config = nk.optimizer.SR()
    else:
        sr_config = None

    driver = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr_config)

    return lind, vs, driver


def _setup_obs(L):
    hi = nk.hilbert.Spin(s=0.5) ** L

    obs_sx = nk.operator.LocalOperator(hi)
    for i in range(L):
        obs_sx += nk.operator.spin.sigmax(hi, i)

    obs = {"SigmaX": obs_sx}
    return obs


####


def test_estimate():
    lind, _, driver = _setup_ss()

    driver.estimate(lind.H @ lind)
    driver.advance(1)
    driver.estimate(lind.H @ lind)


def test_raise_n_iter():
    lind, _, driver = _setup_ss()
    with raises(
        ValueError,
    ):
        driver.run("prova", 12)


def test_steadystate_construction_vstate():
    lind, vs, driver = _setup_ss()

    sa = vs.sampler
    op = nk.optimizer.Sgd(learning_rate=0.05)

    driver = nk.SteadyState(lind, op, sa, nk.models.NDM(), n_samples=1000, seed=SEED)

    driver.run(1)

    assert driver.step_count == 1

    with raises(TypeError):
        ha2 = nk.operator.LocalOperator(lind.hilbert_physical)
        driver = nk.SteadyState(ha2, op, variational_state=driver.state)


def test_steadystate_steadystate_legacy_api():
    lind, vs, driver = _setup_ss()
    op = driver.optimizer
    vs = driver.state
    sr_config = driver.preconditioner

    with pytest.raises(ValueError):
        driver = nk.SteadyState(
            lind, op, variational_state=vs, sr=sr_config, preconditioner=sr_config
        )

    driver = nk.SteadyState(
        lind, op, variational_state=vs, sr=sr_config, sr_restart=True
    )
    assert driver.preconditioner == sr_config
