import netket as nk
import tempfile
import shutil

SEED = 3141592


def _setup_vmc(**kwargs):
    nk.random.seed(SEED)
    g = nk.graph.Hypercube(length=8, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.01, seed=SEED)

    ha = nk.operator.Ising(hi, h=1.0)
    sa = nk.sampler.MetropolisLocal(machine=ma)

    op = nk.optimizer.Sgd(ma, learning_rate=0.1)

    vmc = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, **kwargs)

    return ma, vmc


def test_earlystopping_with_patience():
    es = nk.callbacks.EarlyStopping(patience=10)
    ma, vmc = _setup_vmc(n_samples=500)
    tempdir = tempfile.mkdtemp()
    print("Writing test output files to: {}".format(tempdir))
    prefix = tempdir + "/vmc_test"
    vmc.run(prefix, 300, callback=es)
    shutil.rmtree(tempdir)


def test_earlystopping_with_baseline():
    es = nk.callbacks.EarlyStopping(baseline=100)
    ma, vmc = _setup_vmc(n_samples=500)
    tempdir = tempfile.mkdtemp()
    print("Writing test output files to: {}".format(tempdir))
    prefix = tempdir + "/vmc_test"
    vmc.run(prefix, 300, callback=es)
    shutil.rmtree(tempdir)
