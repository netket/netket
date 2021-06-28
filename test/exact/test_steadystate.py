from pytest import approx
import netket as nk
import numpy as np

from .. import common

pytestmark = common.skipif_mpi

SEED = 3141592
L = 4

sx = [[0, 1], [1, 0]]
sy = [[0, -1j], [1j, 0]]
sz = [[1, 0], [0, -1]]

sigmam = [[0, 0], [1, 0]]


def _setup_system():
    hi = nk.hilbert.Spin(s=0.5) ** L

    ha = nk.operator.LocalOperator(hi)
    j_ops = []
    for i in range(L):
        ha += (0.3 / 2.0) * nk.operator.LocalOperator(hi, sx, [i])
        ha += (2.0 / 4.0) * nk.operator.LocalOperator(
            hi, np.kron(sz, sz), [i, (i + 1) % L]
        )
        j_ops.append(nk.operator.LocalOperator(hi, sigmam, [i]))

    # Create the liouvillian
    lind = nk.operator.LocalLiouvillian(ha, j_ops)
    return hi, lind


def _setup_ss(**kwargs):
    nk.random.seed(SEED)
    np.random.seed(SEED)

    hi, lind = _setup_system()

    ma = nk.machine.density_matrix.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.01, seed=SEED)

    sa = nk.sampler.MetropolisLocal(machine=ma)
    sa_obs = nk.sampler.MetropolisLocal(machine=ma.diagonal())

    op = nk.optimizer.Sgd(ma, learning_rate=0.1)

    if "sr" in kwargs:
        sr = nk.optimizer.SR(ma, **kwargs["sr"])
        kwargs["sr"] = sr

    ss = nk.SteadyState(
        lindblad=lind, sampler=sa, optimizer=op, sampler_obs=sa_obs, **kwargs
    )

    return ma, ss


def _setup_obs():
    hi = nk.hilbert.Spin(s=0.5) ** L

    obs_sx = nk.operator.LocalOperator(hi)
    for i in range(L):
        obs_sx += nk.operator.LocalOperator(hi, sx, [i])

    obs = {"SigmaX": obs_sx}
    return obs


def test_exact_ss_ed():
    _, lind = _setup_system()

    dm_ss = nk.exact.steady_state(lind, method="ed", sparse=True)
    Lop = lind.to_sparse()

    mat = np.abs(Lop @ dm_ss.reshape(-1))
    print(mat)
    assert np.all(mat == approx(0.0, rel=1e-4, abs=1e-4))
    assert dm_ss.trace() - 1 == approx(0.0, rel=1e-5, abs=1e-8)

    # dm_ss_d = nk.exact.steady_state(lind, method="ed", sparse=False)
    # Lop = lind.to_sparse()

    # mat = np.abs(dm_ss - dm_ss_d)
    # print(mat)
    # assert np.all(mat == approx(0.0, rel=1e-4, abs=1e-4))


def test_exact_ss_iterative():
    _, lind = _setup_system()

    dm_ss = nk.exact.steady_state(lind, sparse=True, method="iterative", tol=1e-5)
    Lop = lind.to_linear_operator()

    mat = np.abs(Lop @ dm_ss.reshape(-1))
    print(mat)
    assert np.all(mat == approx(0.0, rel=1e-5, abs=1e-5))
    assert dm_ss.trace() - 1 == approx(0.0, rel=1e-5, abs=1e-5)
