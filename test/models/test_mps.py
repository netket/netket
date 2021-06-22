import pytest

import netket as nk


@pytest.mark.parametrize("diag", [False, True])
def test_mps(diag):
    L = 6
    g = nk.graph.Hypercube(length=L, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ma = nk.models.MPSPeriodic(hilbert=hi, graph=g, bond_dim=2, diag=diag)
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)

    vs = nk.vqs.MCState(sa, ma, n_samples=1000)

    ha = nk.operator.Ising(hi, graph=g, h=1.0)
    op = nk.optimizer.Sgd(learning_rate=0.05)

    driver = nk.Vmc(ha, op, variational_state=vs)

    driver.run(3)


def test_mps_nonchain():
    g = nk.graph.Hypercube(3, 2)
    hi = nk.hilbert.Spin(1 / 2, g.n_nodes)
    with pytest.warns(
        UserWarning,
        match="graph is not isomorphic to chain with periodic boundary conditions",
    ):
        ma = nk.models.MPSPeriodic(hi, g, bond_dim=2)
        sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)
        _ = nk.vqs.MCState(sa, ma, n_samples=1000)
