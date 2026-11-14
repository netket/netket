import jax
import jax.numpy as jnp
import numpy as np
import pytest

import netket as nk
from netket.experimental.operator import ParticleNumberConservingFermioperator2nd

from .. import common


@pytest.mark.parametrize(
    "operator_cls",
    [
        pytest.param(nk.operator.LocalOperatorJax, id="jax"),
        pytest.param(
            nk.operator.LocalOperatorNumba,
            id="numba",
            marks=common.skipif_multiple_devices(
                "numba operators do not support several devices"
            ),
        ),
    ],
)
def test_adjoint_matmul_changes_sector(operator_cls):
    # Reproducer of #2322: the state is |1>, and the amplitudes the network
    # assigns to |0> and |2>, outside of the constraint, must not matter.
    hi = nk.hilbert.Fock(n_max=2, N=1, n_particles=1)
    vs = nk.vqs.MCState(
        nk.sampler.ExactSampler(hi), nk.models.RBM(alpha=1), n_samples=64, seed=0
    )
    vs.parameters = jax.tree_util.tree_map(jnp.zeros_like, vs.parameters)

    a = nk.operator.boson.destroy(hi, 0, cls=operator_cls)
    np.testing.assert_allclose(vs.expect(a.H @ a).mean, 1.0)


def _two_site_logpsi(variables, x, **kwargs):
    theta, phase, eta = variables["params"]["coefficients"]
    return (theta + 1j * phase) * x[..., 1] + eta * (jnp.sum(x, axis=-1) - 1)


def test_adjoint_matmul_physical_gradient_and_sector_independence():
    # On the physical sector psi = |10> + exp(theta + i*phase)|01>.
    # eta changes only the amplitudes outside that sector.
    hi = nk.hilbert.Fock(n_max=1, N=2, n_particles=1)
    theta, phase = 0.3, 0.4
    a = sum(nk.operator.boson.destroy(hi, i) for i in range(2)) / np.sqrt(2)
    op = a.H @ a
    expected = 0.5 + np.cos(phase) / (2 * np.cosh(theta))
    expected_grad = np.array(
        [
            -np.cos(phase) * np.sinh(theta) / (2 * np.cosh(theta) ** 2),
            -np.sin(phase) / (2 * np.cosh(theta)),
            0.0,
        ]
    )

    mc_results = []
    for eta in (0.0, 0.7):
        variables = {"params": {"coefficients": jnp.array([theta, phase, eta])}}
        exact_state = nk.vqs.FullSumState(
            hi, apply_fun=_two_site_logpsi, variables=variables
        )
        exact_stats, exact_grad = exact_state.expect_and_grad(op)
        np.testing.assert_allclose(exact_stats.mean, expected, atol=1e-12)
        np.testing.assert_allclose(
            exact_grad["coefficients"], expected_grad, atol=1e-12
        )

        # The same seed samples the same physical distribution for either eta.
        # The MC value and gradient must therefore be unchanged, sample by sample.
        mc_state = nk.vqs.MCState(
            nk.sampler.ExactSampler(hi),
            apply_fun=_two_site_logpsi,
            variables=variables,
            n_samples=128,
            seed=0,
        )
        stats, grad = mc_state.expect_and_grad(op)
        np.testing.assert_allclose(grad["coefficients"][2], 0.0, atol=1e-12)
        mc_results.append((stats.mean, grad["coefficients"]))

    np.testing.assert_allclose(mc_results[0][0], mc_results[1][0], atol=1e-12)
    np.testing.assert_allclose(mc_results[0][1], mc_results[1][1], atol=1e-12)


def test_unconstrained_adjoint_matmul_keeps_squared():
    hi = nk.hilbert.Fock(n_max=2, N=1)
    a = nk.operator.boson.destroy(hi, 0)
    assert isinstance(a.H @ a, nk.operator.Squared)


def test_variance_oloc_squared_constrained():
    # Squared(H) must keep the |H_loc|² estimator on constrained spaces, as this
    # operator cannot form H @ H.
    hi = nk.hilbert.SpinOrbitalFermions(4, n_fermions=2)
    cdag, c = nk.operator.fermion.create, nk.operator.fermion.destroy
    ha = sum(cdag(hi, i) @ c(hi, (i + 1) % 4) for i in range(4))
    ha = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(
        ha + ha.H.collect()
    )
    vs = nk.vqs.MCState(
        nk.sampler.ExactSampler(hi), nk.models.RBM(alpha=1), n_samples=1024, seed=0
    )

    psi = vs.to_array()
    h_psi = ha.to_dense() @ psi
    exact = (np.vdot(h_psi, h_psi) - np.vdot(psi, h_psi) ** 2).real

    var_op = nk.observable.VarianceObservable(ha, use_Oloc_squared=True)
    # Both APIs must continue to support the explicitly requested fast estimator.
    vs.expect_and_grad(var_op)
    # expect() propagates the joint uncertainty of <H> and <H^2>.
    stats = vs.expect(var_op)
    np.testing.assert_allclose(stats.mean.real, exact, atol=5 * stats.error_of_mean)
