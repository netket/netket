from pytest import approx
import netket as nk
from netket.vmc import Vmc
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
    last_step = None
    for i, step in enumerate(vmc.iter(300, store_params=False)):
        assert i == step.current_step
        count += 1
        obs = step.observables
        assert len(obs) == 2
        for name in 'Energy', 'EnergyVariance':
            assert name in obs
            e = obs[name]
            assert 'Mean' in e and 'Sigma' in e and 'Taucorr' in e
        last_step = step

    assert count == 300
    assert last_step.observables.Energy['Mean'] == approx(-10.25, abs=0.2)
