import pytest

import jax
import numpy as np
import netket as nk
import netket.experimental as nkx

from flax import serialization
from jax.sharding import PositionalSharding


def _setup(L, alpha=1):
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ma = nk.models.RBM(alpha=alpha, param_dtype=np.complex128)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16 * jax.device_count(), dtype=np.int8)
    vs = nk.vqs.MCState(sa, ma, n_samples=1024, n_discard_per_chain=8)
    ha = nk.operator.IsingJax(hilbert=vs.hilbert, graph=g, h=1.0)
    return vs, g, ha


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_setup():
    # make sure that the tests are running with >1 devices
    assert jax.device_count() > 1


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_sampling():
    vs, *_ = _setup(16)
    n_chains = 16 * jax.device_count()
    n_samples = 1024

    # check sampler state has correct sharding
    x = vs.sampler_state.σ
    assert x.shape == (n_chains, vs.hilbert.size)
    assert isinstance(x.sharding, PositionalSharding)
    assert x.sharding.shape == (jax.device_count(), 1)
    assert x.sharding.device_set == set(jax.devices())

    # check samples have correct sharding
    samples = vs.sample()
    assert samples.shape == (n_chains, n_samples // n_chains, vs.hilbert.size)
    assert isinstance(samples.sharding, PositionalSharding)
    assert samples.sharding.shape == (jax.device_count(), 1, 1)
    assert samples.sharding.device_set == set(jax.devices())

    # check sampler state still has correct sharding after having sampled
    x = vs.sampler_state.σ
    assert x.shape == (n_chains, vs.hilbert.size)
    assert isinstance(x.sharding, PositionalSharding)
    assert x.sharding.shape == (jax.device_count(), 1)
    assert x.sharding.device_set == set(jax.devices())


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_expect():
    vs, _, ha = _setup(16)
    E = vs.expect(ha)
    # check printing works
    str(E)
    for l in jax.tree_util.tree_leaves(E):
        assert l.is_fully_replicated
        assert isinstance(l.sharding, PositionalSharding)
        assert l.sharding.device_set == set(jax.devices())


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_grad():
    vs, g, ha = _setup(16)
    E, G = vs.expect_and_grad(ha)
    # check printing works
    str(E)
    for l in jax.tree_util.tree_leaves(E):
        assert l.is_fully_replicated
        assert isinstance(l.sharding, PositionalSharding)
        assert l.sharding.device_set == set(jax.devices())

    for l in jax.tree_util.tree_leaves(G):
        assert l.is_fully_replicated
        assert isinstance(l.sharding, PositionalSharding)
        assert l.sharding.device_set == set(jax.devices())


@pytest.mark.parametrize(
    "Op",
    [pytest.param(nk.operator.Ising, id="numba")]
    if jax.process_count() < 2
    else []
    + [
        pytest.param(nk.operator.IsingJax, id="jax"),
    ],
)
@pytest.mark.parametrize(
    "qgt",
    [
        pytest.param(nk.optimizer.qgt.QGTJacobianPyTree, id="pytree"),
        pytest.param(nk.optimizer.qgt.QGTJacobianDense, id="dense"),
        pytest.param(nk.optimizer.qgt.QGTOnTheFly, id="onthefly"),
    ],
)
@pytest.mark.parametrize(
    "chunk_size",
    [None, 64],
)
@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_vmc(Op, qgt, chunk_size):
    vs, g, _ = _setup(16)
    vs.chunk_size = chunk_size
    # initially the params are only on the first device of each process
    # but they will be broadcast on the first invocation, so here we only check its fully addressable
    for l in jax.tree_util.tree_leaves(vs.variables):
        assert l.is_fully_replicated

    ha = Op(hilbert=vs.hilbert, graph=g, h=1.0)
    opt = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(qgt(holomorphic=True), diag_shift=0.01)
    gs = nk.VMC(ha, opt, variational_state=vs, preconditioner=sr)
    gs.run(5)

    for l in jax.tree_util.tree_leaves(vs.variables):
        assert l.is_fully_replicated
        assert isinstance(l.sharding, PositionalSharding)
        assert l.sharding.device_set == set(jax.devices())


@pytest.mark.parametrize(
    "qgt",
    [
        pytest.param(nk.optimizer.qgt.QGTJacobianPyTree, id="pytree"),
        pytest.param(nk.optimizer.qgt.QGTJacobianDense, id="dense"),
    ],
)
@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_qgt_jacobian(qgt):
    n_samples = 1024
    vs, *_ = _setup(16)
    S = vs.quantum_geometric_tensor(qgt(holomorphic=True))
    for l in jax.tree_util.tree_leaves(S.O):
        assert l.shape[0] == n_samples
        assert isinstance(l.sharding, PositionalSharding)
        assert l.sharding.shape[0] == jax.device_count()
        assert l.sharding.device_set == set(jax.devices())
    v = vs.parameters
    res = S @ v

    for l in jax.tree_util.tree_leaves(res):
        assert l.is_fully_replicated
        assert isinstance(l.sharding, PositionalSharding)
        assert l.sharding.device_set == set(jax.devices())


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_qgt_onthefly():
    vs, *_ = _setup(16)
    S = vs.quantum_geometric_tensor(nk.optimizer.qgt.QGTOnTheFly(holomorphic=True))
    # TODO maybe check S._mat_vec tree leaves
    v = vs.parameters
    res = S @ v

    for l in jax.tree_util.tree_leaves(res):
        assert l.is_fully_replicated
        assert isinstance(l.sharding, PositionalSharding)
        assert l.sharding.device_set == set(jax.devices())


@pytest.mark.parametrize(
    "Op",
    [pytest.param(nk.operator.Ising, id="numba")]
    if jax.process_count() < 2
    else []
    + [
        pytest.param(nk.operator.IsingJax, id="jax"),
    ],
)
@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_operators(Op):
    vs, g, *_ = _setup(16)
    ha = Op(hilbert=vs.hilbert, graph=g, h=1.0)
    x = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(vs.samples, 0, 2)

    assert x.sharding.shape == (jax.device_count(), 1)
    xp, mels = ha.get_conn_padded(x)

    n_conn = xp.shape[1]
    assert xp.shape == (x.shape[0], n_conn, x.shape[-1])
    assert mels.shape == (x.shape[0], n_conn)

    assert isinstance(xp.sharding, PositionalSharding)
    assert isinstance(mels.sharding, PositionalSharding)

    assert xp.sharding.shape == (jax.device_count(), 1, 1)
    assert mels.sharding.shape == (jax.device_count(), 1)

    assert xp.sharding.device_set == set(jax.devices())
    assert mels.sharding.device_set == set(jax.devices())


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
@pytest.mark.parametrize(
    "chunk_size",
    [None, 64],
)
def test_fullsumstate(chunk_size):
    L = 12
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ma = nk.models.RBM(alpha=1, param_dtype=np.complex128)
    vs = nk.vqs.FullSumState(hi, ma, chunk_size=chunk_size)
    ha = nk.operator.IsingJax(hilbert=vs.hilbert, graph=g, h=1.0)
    opt = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(
        nk.optimizer.qgt.QGTOnTheFly(holomorphic=True), diag_shift=0.01
    )
    gs = nk.VMC(ha, opt, variational_state=vs, preconditioner=sr)
    gs.run(5)


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
@pytest.mark.parametrize(
    "chunk_size",
    [None, 64],
)
def test_exactsampler(chunk_size):
    L = 12
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ma = nk.models.RBM(alpha=1, param_dtype=np.complex128)
    sa = nk.sampler.ExactSampler(hi, dtype=np.int8)
    vs = nk.vqs.MCState(sa, ma, n_samples=1024, chunk_size=chunk_size)

    pos_sharding = jax.sharding.PositionalSharding(jax.devices())
    assert vs.samples.sharding.is_equivalent_to(pos_sharding.reshape(1, -1, 1), 3)

    ha = nk.operator.IsingJax(hilbert=vs.hilbert, graph=g, h=1.0)
    opt = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(
        nk.optimizer.qgt.QGTOnTheFly(holomorphic=True), diag_shift=0.01
    )
    gs = nk.VMC(ha, opt, variational_state=vs, preconditioner=sr)
    gs.run(5)


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_autoreg():
    L = 32
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ha = nk.operator.IsingJax(hilbert=hi, graph=g, h=1)
    ma = nk.models.FastARNNConv1D(hilbert=hi, layers=3, features=16, kernel_size=3)
    sa = nk.sampler.ARDirectSampler(hi)
    opt = nk.optimizer.Sgd(learning_rate=0.1)
    sr = nk.optimizer.SR(diag_shift=0.01)
    vs = nk.vqs.MCState(sa, ma, n_samples=256)
    pos_sharding = jax.sharding.PositionalSharding(jax.devices())
    assert vs.samples.sharding.is_equivalent_to(pos_sharding.reshape(1, -1, 1), 3)
    gs = nk.VMC(ha, opt, variational_state=vs, preconditioner=sr)
    gs.run(n_iter=5)


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
@pytest.mark.parametrize(
    "logger",
    [
        pytest.param(
            (
                nk.logging.RuntimeLog(),
                lambda logger: logger.serialize("_test_runtimelog.json"),
            ),
            id="RuntimeLog",
        ),
        pytest.param(
            (
                nk.logging.JsonLog("_test_jsonlog", save_params_every=1, write_every=1),
                lambda logger: None,
            ),
            id="JsonLog",
        ),
        pytest.param(
            (nk.logging.StateLog("_test_statelog", save_every=1), lambda logger: None),
            id="StateLog",
        ),
        pytest.param(
            (
                nkx.logging.HDF5Log(
                    "_test_hdf5log", save_params=True, save_params_every=1
                ),
                lambda logger: None,
            ),
            id="HDF5Log",
        ),
    ],
)
def test_loggers(logger):
    vs, _, ha = _setup(12)
    logger, out_fun = logger
    opt = nk.optimizer.Sgd(learning_rate=0.05)
    gs = nk.VMC(ha, opt, variational_state=vs)
    gs.run(10, out=logger)
    out_fun(logger)


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_serialization():
    vs, _, ha = _setup(12)

    b = serialization.to_bytes(vs)
    vs = serialization.from_bytes(vs, b)

    opt = nk.optimizer.Sgd(learning_rate=0.05)
    gs = nk.VMC(ha, opt, variational_state=vs)
    gs.run(1)

    b = serialization.to_bytes(gs.state)
    vs2 = serialization.from_bytes(gs.state, b)
    gs2 = nk.VMC(ha, opt, variational_state=vs2)
    gs2.run(1)


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
@pytest.mark.parametrize("ode_jit", [False, True])
def test_timeevolution(ode_jit):
    nk.config.update("netket_experimental_disable_ode_jit", not ode_jit)
    L = 8
    vs, _, ha = _setup(L)
    Sx = sum([nk.operator.spin.sigmax(ha.hilbert, i) for i in range(L)])
    Sx = Sx.to_pauli_strings().to_jax_operator()
    integrator = nkx.dynamics.Euler(dt=0.001)
    te = nkx.TDVP(
        ha,
        variational_state=vs,
        integrator=integrator,
        t0=0.0,
        qgt=nk.optimizer.qgt.QGTOnTheFly(holomorphic=True, diag_shift=1e-4),
        error_norm="qgt",
    )
    te.run(T=0.005, obs={"Sx": Sx}, show_progress=True)
    te2 = nkx.driver.TDVPSchmitt(
        ha,
        variational_state=vs,
        integrator=integrator,
        t0=0.0,
        error_norm="qgt",
        holomorphic=True,
    )
    te2.run(T=0.005, obs={"Sx": Sx}, show_progress=True)


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
def test_srt():
    vs, _, ha = _setup(12, alpha=2)
    vs.n_samples = 64
    opt = nk.optimizer.Sgd(learning_rate=0.05)
    gs = nkx.driver.VMC_SRt(
        ha,
        opt,
        variational_state=vs,
        diag_shift=0.1,
        jacobian_mode="complex",
    )
    gs.run(2)
