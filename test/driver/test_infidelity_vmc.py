import netket as nk
import jax.numpy as jnp
import jax
import numpy as np

import pytest


seed = 123
seed_target = 456


def _setup(useExactSampler=True):
    N = 3
    hi = nk.hilbert.Spin(0.5, N)

    ma = nk.models.RBM(alpha=1)
    n_samples = 8192

    H = nk.operator.IsingJax(hilbert=hi, graph=nk.graph.Chain(N), J=-1.0, h=1.0)

    if useExactSampler:
        sa = nk.sampler.ExactSampler(hilbert=hi)
        vs = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            seed=seed,
        )
        vs_target = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            seed=seed_target,
        )

    else:
        sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
        vs = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            n_discard_per_chain=1e3,
            seed=seed,
        )
        vs_target = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            seed=seed_target,
        )

    vs_exact = nk.vqs.FullSumState(
        hilbert=hi,
        model=ma,
        seed=seed,
    )
    vs_exact_target = nk.vqs.FullSumState(
        hilbert=hi,
        model=ma,
        seed=seed_target,
    )

    return vs, vs_target, vs_exact, vs_exact_target, H


def I_exact_fun(params, vs, vs_target, U=None):
    params_old = vs.parameters
    vs.parameters = params
    state = vs.to_array()
    vs.parameters = params_old

    if U is not None:
        state_target = U @ vs_target.to_array()
        state_target /= jnp.linalg.norm(state_target)

    else:
        state_target = vs_target.to_array()

    I = 1 - jnp.absolute(state.conj() @ state_target) ** 2

    return I


@pytest.mark.parametrize(
    "useExactSampler",
    [
        pytest.param(True, id="ExactSampler"),
        pytest.param(False, id="MetropolisSampler"),
    ],
)
@pytest.mark.parametrize(
    "useOperator",
    [
        pytest.param(True, id="useOperator"),
        pytest.param(False, id="useOperator"),
    ],
)
def test_MCState(useExactSampler, useOperator):
    vs, vs_target, _, _, H = _setup(useExactSampler)

    optimizer = nk.optimizer.Sgd(learning_rate=0.1)
    diag_shift = 0.001

    if useOperator:
        driver = nk.driver.Infidelity_SR(
            target_state=vs_target,
            optimizer=optimizer,
            diag_shift=diag_shift,
            variational_state=vs,
        )
        driver.run(n_iter=200)
        I_exact = I_exact_fun(vs.parameters, vs, vs_target)
    else:
        driver = nk.driver.Infidelity_SR(
            target_state=vs_target,
            optimizer=optimizer,
            diag_shift=diag_shift,
            variational_state=vs,
            operator=H,
        )
        driver.run(n_iter=200)
        I_exact = I_exact_fun(vs.parameters, vs, vs_target, U=H)

    assert I_exact < 1e-12


@pytest.mark.parametrize(
    "fullsumTarget",
    [
        pytest.param(True, id="FullSumState target"),
        pytest.param(False, id="MCState  target"),
    ],
)
@pytest.mark.parametrize(
    "useOperator",
    [
        pytest.param(True, id="useOperator"),
        pytest.param(False, id="useOperator"),
    ],
)
def test_FullSumState(fullsumTarget, useOperator):
    _, vs_target_mc, vs_exact, vs_exact_target, H = _setup(useExactSampler=True)

    # Use FullSumState as the variational state
    vs = vs_exact

    # Set the target state based on the parameter
    vs_target = vs_exact_target if fullsumTarget else vs_target_mc

    optimizer = nk.optimizer.Sgd(learning_rate=0.1)
    diag_shift = 0.001

    if useOperator:
        driver = nk.driver.Infidelity_SR(
            target_state=vs_target,
            optimizer=optimizer,
            diag_shift=diag_shift,
            variational_state=vs,
        )
        driver.run(n_iter=200)
        I_exact = I_exact_fun(vs.parameters, vs, vs_target)
    else:
        driver = nk.driver.Infidelity_SR(
            target_state=vs_target,
            optimizer=optimizer,
            diag_shift=diag_shift,
            variational_state=vs,
            operator=H,
        )
        driver.run(n_iter=200)
        I_exact = I_exact_fun(vs.parameters, vs, vs_target, U=H)

    assert I_exact < 1e-12


@pytest.mark.parametrize(
    "useOperator",
    [
        pytest.param(True, id="useOperator"),
        pytest.param(False, id="useOperator"),
    ],
)
def test_infidelity_sr_onthefly_vs_sr(useOperator):
    vs, vs_target, _, _, H = _setup(useExactSampler=True)
    n_iters = 5

    kwargs = {}
    if not useOperator:
        kwargs["operator"] = H

    driver_onthefly = nk.driver.Infidelity_SR(
        target_state=vs_target,
        optimizer=nk.optimizer.Sgd(learning_rate=0.1),
        diag_shift=0.001,
        variational_state=vs,
        use_ntk=False,
        on_the_fly=True,
        linear_solver=nk.optimizer.solver.cholesky,
        **kwargs,
    )
    driver_onthefly.run(n_iter=n_iters)

    vs_ref, vs_target_ref, _, _, H_ref = _setup(useExactSampler=True)
    if not useOperator:
        kwargs_ref = {"operator": H_ref}
    else:
        kwargs_ref = {}

    driver_ref = nk.driver.Infidelity_SR(
        target_state=vs_target_ref,
        optimizer=nk.optimizer.Sgd(learning_rate=0.1),
        diag_shift=0.001,
        variational_state=vs_ref,
        use_ntk=False,
        on_the_fly=False,
        linear_solver=nk.optimizer.solver.cholesky,
        **kwargs_ref,
    )
    driver_ref.run(n_iter=n_iters)

    jax.tree_util.tree_map(np.testing.assert_allclose, vs.parameters, vs_ref.parameters)


@pytest.mark.parametrize(
    "useOperator",
    [
        pytest.param(True, id="useOperator"),
        pytest.param(False, id="useOperator"),
    ],
)
def test_infidelity_fullsum_sr_onthefly_vs_sr(useOperator):
    _, _, vs_exact, vs_exact_target, H = _setup(useExactSampler=True)
    n_iters = 5

    kwargs = {}
    if not useOperator:
        kwargs["operator"] = H

    driver_onthefly = nk.driver.Infidelity_SR(
        target_state=vs_exact_target,
        optimizer=nk.optimizer.Sgd(learning_rate=0.1),
        diag_shift=0.001,
        variational_state=vs_exact,
        use_ntk=False,
        on_the_fly=True,
        linear_solver=nk.optimizer.solver.cholesky,
        **kwargs,
    )
    driver_onthefly.run(n_iter=n_iters)

    _, _, vs_exact_ref, vs_exact_target_ref, H_ref = _setup(useExactSampler=True)
    if not useOperator:
        kwargs_ref = {"operator": H_ref}
    else:
        kwargs_ref = {}

    driver_ref = nk.driver.Infidelity_SR(
        target_state=vs_exact_target_ref,
        optimizer=nk.optimizer.Sgd(learning_rate=0.1),
        diag_shift=0.001,
        variational_state=vs_exact_ref,
        use_ntk=False,
        on_the_fly=False,
        linear_solver=nk.optimizer.solver.cholesky,
        **kwargs_ref,
    )
    driver_ref.run(n_iter=n_iters)

    jax.tree_util.tree_map(
        np.testing.assert_allclose, vs_exact.parameters, vs_exact_ref.parameters
    )
