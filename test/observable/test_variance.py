import netket as nk
import netket.experimental as nkx
import numpy as np

import pytest

from ..variational.finite_diff import central_diff_grad, same_derivatives
from .. import common

seed = 123


def _setup(useExactSampler=True):
    N = 3
    hi = nk.hilbert.Spin(0.5, N)

    ma = nk.models.RBM(alpha=1)
    n_samples = 8192

    if useExactSampler:
        sa = nk.sampler.ExactSampler(hilbert=hi)
        vs = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            seed=seed,
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

    vs_exact = nk.vqs.FullSumState(
        hilbert=hi,
        model=ma,
        seed=seed,
    )

    H = nk.operator.IsingJax(hi, graph=nk.graph.Chain(N), h=1, J=-1)
    H2 = H @ H

    return vs, vs_exact, H, H2


def var_exact_fun(params, vs, H, H2):
    params_old = vs.parameters
    vs.parameters = params
    state = vs.to_array()
    vs.parameters = params_old

    return state.conj() @ (H2 @ state) - (state.conj() @ (H @ state)) ** 2


@common.xfailif_mpi  # broken in recent jax versions
@pytest.mark.parametrize(
    "useExactSampler",
    [
        pytest.param(True, id="ExactSampler"),
        pytest.param(False, id="MetropolisSampler"),
    ],
)
@pytest.mark.parametrize(
    "use_Oloc_squared",
    [
        pytest.param(True, id="UseOloc2"),
        pytest.param(False, id="NotUseOloc2"),
    ],
)
def test_MCState(useExactSampler, use_Oloc_squared):
    vs, vs_exact, H, H2 = _setup(useExactSampler)
    var_op = nkx.observable.VarianceObservable(H, use_Oloc_squared=use_Oloc_squared)

    params, unravel = nk.jax.tree_ravel(vs.parameters)

    var_stats1 = vs.expect(var_op)
    var_stats, var_grad = vs.expect_and_grad(var_op)
    var_grad, _ = nk.jax.tree_ravel(var_grad)

    var_exact = var_exact_fun(vs.parameters, vs, H, H2)

    def _var_exact_fun(params, vstate, H, H2):
        return var_exact_fun(unravel(params), vstate, H, H2)

    var_grad_exact = central_diff_grad(_var_exact_fun, params, 1.0e-5, vs, H, H2)

    var_mean1 = var_stats1.mean
    var_err1 = 5 * var_stats1.error_of_mean
    var_mean = var_stats.mean
    var_err = 5 * var_stats.error_of_mean

    np.testing.assert_allclose(var_exact.real, var_mean1.real, atol=var_err1)
    np.testing.assert_allclose(var_exact.real, var_mean.real, atol=var_err)

    same_derivatives(var_grad, var_grad_exact, abs_eps=var_err, rel_eps=var_err)


@pytest.mark.parametrize(
    "use_Oloc_squared",
    [
        pytest.param(True, id="UseOloc2"),
        pytest.param(False, id="NotUseOloc2"),
    ],
)
def test_FullSumState(use_Oloc_squared):
    err = 1e-3
    vs, vs_exact, H, H2 = _setup()
    var_op = nkx.observable.VarianceObservable(H, use_Oloc_squared=use_Oloc_squared)

    params, unravel = nk.jax.tree_ravel(vs_exact.parameters)

    var_stats1 = vs_exact.expect(var_op)
    var_stats, var_grad = vs_exact.expect_and_grad(var_op)
    var_grad, _ = nk.jax.tree_ravel(var_grad)

    var_exact = var_exact_fun(vs_exact.parameters, vs_exact, H, H2)

    def _var_exact_fun(params, vstate, H, H2):
        return var_exact_fun(unravel(params), vstate, H, H2)

    var_grad_exact = central_diff_grad(_var_exact_fun, params, 1.0e-5, vs_exact, H, H2)

    var_mean1 = var_stats1.mean
    var_mean = var_stats.mean

    np.testing.assert_allclose(var_exact.real, var_mean1.real, atol=1e-6)
    np.testing.assert_allclose(var_exact.real, var_mean.real, atol=1e-6)

    same_derivatives(var_grad, var_grad_exact, abs_eps=err, rel_eps=err)
