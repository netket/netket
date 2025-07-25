import netket as nk
import netket.experimental as nkx
import numpy as np
import jax.numpy as jnp

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
    vs, vs_target, vs_exact, vs_exact_target, H = _setup(useExactSampler)

    if useOperator:
        I_op = nkx.observable.InfidelityOperator(target_state=vs_target, operator=H)
        I_exact = I_exact_fun(vs.parameters, vs, vs_target, U=H)

    else:
        I_op = nkx.observable.InfidelityOperator(target_state=vs_target)
        I_exact = I_exact_fun(vs.parameters, vs, vs_target)

    I_stats = vs.expect(I_op)

    I_mean = I_stats.mean
    I_err = 5 * I_stats.error_of_mean

    np.testing.assert_allclose(I_exact.real, I_mean.real, atol=I_err)


@pytest.mark.parametrize(
    "useOperator",
    [
        pytest.param(True, id="useOperator"),
        pytest.param(False, id="useOperator"),
    ],
)
def test_FullSumState(useOperator):
    vs, vs_target, vs_exact, vs_exact_target, H = _setup()

    if useOperator:
        I_op = nkx.observable.InfidelityOperator(target_state=vs_target, operator=H)
        I_exact = I_exact_fun(vs.parameters, vs, vs_target, U=H)
    else:
        I_op = nkx.observable.InfidelityOperator(target_state=vs_target)
        I_exact = I_exact_fun(vs.parameters, vs, vs_target)

    I_stats = vs_exact.expect(I_op)

    I_mean = I_stats.mean

    np.testing.assert_allclose(I_exact.real, I_mean.real, atol=1e-6)
