from pytest import approx
import netket as nk
from netket.variational import Vmc
from mpi4py import MPI


SEED = 3141592


def test_vmc_iterator():
    g = nk.graph.Hypercube(length=8, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(seed=SEED, sigma=0.01)

    ha = nk.operator.Ising(hi, h=1.0)
    sa = nk.sampler.MetropolisLocal(machine=ma)
    sa.seed(SEED)
    op = nk.optimizer.Sgd(learning_rate=0.1)

    vmc = Vmc(
        hamiltonian=ha,
        sampler=sa,
        optimizer=op,
        n_samples=500,
        diag_shift=0.01)

    count = 0
    last_obs = None
    for i, step in enumerate(vmc.iter(300)):
        count += 1
        assert step == i
        obs = vmc.get_observable_stats()
        for name in 'Energy', 'EnergyVariance':
            assert name in obs
            e = obs[name]
            assert 'Mean' in e and 'Sigma' in e and 'Taucorr' in e
        last_obs = obs

    assert count == 300
    assert last_obs['Energy']['Mean'] == approx(-10.25, abs=0.2)

def test_ed():
    first_n=3
    g = nk.graph.Hypercube(length=8, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    ha = nk.operator.Ising(h=1.0, hilbert=hi)
    
    # Test Lanczos ED with eigenvectors
    res = nk.exact.lanczos_ed(ha, first_n=first_n, compute_eigenvectors=True)
    assert len(res.eigenvalues) == first_n
    assert len(res.eigenvectors) == first_n
    gse = res.mean(ha, 0)
    fse = res.mean(ha, 1)
    assert gse == approx(res.eigenvalues[0], rel=1e-12, abs=1e-12)
    assert fse == approx(res.eigenvalues[1], rel=1e-12, abs=1e-12)

    # Test Lanczos ED without eigenvectors
    res = nk.exact.lanczos_ed(ha, first_n=first_n, compute_eigenvectors=False)
    assert len(res.eigenvalues) == first_n
    assert len(res.eigenvectors) == 0

    # Test Full ED with eigenvectors
    res = nk.exact.full_ed(ha, first_n=first_n, compute_eigenvectors=True)
    assert len(res.eigenvalues) == first_n
    assert len(res.eigenvectors) == first_n
    gse = res.mean(ha, 0)
    fse = res.mean(ha, 1)
    assert gse == approx(res.eigenvalues[0], rel=1e-12, abs=1e-12)
    assert fse == approx(res.eigenvalues[1], rel=1e-12, abs=1e-12)

    # Test Full ED without eigenvectors
    res = nk.exact.full_ed(ha, first_n=first_n, compute_eigenvectors=False)
    assert len(res.eigenvalues) == first_n
    assert len(res.eigenvectors) == 0
