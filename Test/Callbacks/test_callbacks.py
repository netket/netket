import netket as nk

SEED = 3141592


def _run_vmc(initial_best_value=None, **kwargs):
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
    if initial_best_value is not None:
        es._best_val = initial_best_value
    vmc.run(20, callback=es)
    return vmc.step_count


def test_earlystopping_with_patience():
    patience = 10
    step_value = _run_vmc(initial_best_value=-1e6, patience=patience)
    assert step_value == patience


def test_earlystopping_with_baseline():
    step_value = _run_vmc(baseline=-10)
