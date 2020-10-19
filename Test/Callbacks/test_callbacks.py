import netket as nk
import tempfile
import shutil

SEED = 3141592


def _run_vmc(**kwargs):
    nk.random.seed(SEED)
    g = nk.graph.Hypercube(length=8, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.01, seed=SEED)

    ha = nk.operator.Ising(hi, h=1.0)
    sa = nk.sampler.MetropolisLocal(machine=ma)

    op = nk.optimizer.Sgd(ma, learning_rate=0.1)

    vmc = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=500)
    es = nk.callbacks.EarlyStopping(**kwargs)
    vmc.run(300, callback=es)


def test_earlystopping_with_patience():
    _run_vmc(patience=10)  # Runs for ~50 iterations only


def test_earlystopping_with_baseline():
    _run_vmc(baseline=-10)  # Runs for ~2 iterations only
