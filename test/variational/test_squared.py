import jax
import jax.numpy as jnp
import numpy as np
import pytest

import netket as nk
from netket.experimental.operator import ParticleNumberConservingFermioperator2nd


@pytest.mark.parametrize(
    "operator_cls", [nk.operator.LocalOperatorNumba, nk.operator.LocalOperatorJax]
)
@pytest.mark.parametrize("visible_bias", [0.0, np.log(2.0)])
def test_adjoint_matmul_changes_sector(operator_cls, visible_bias):
    # Reproducer of #2322: the state is |1>, and the amplitudes the network
    # assigns to |0> and |2>, outside of the constraint, must not matter.
    if (
        nk.config.netket_sharding
        and jax.device_count() > 1
        and operator_cls is nk.operator.LocalOperatorNumba
    ):
        pytest.skip("Numba operators do not support sharding")

    hi = nk.hilbert.Fock(n_max=2, N=1, n_particles=1)
    vs = nk.vqs.MCState(
        nk.sampler.ExactSampler(hi), nk.models.RBM(alpha=1), n_samples=64, seed=0
    )
    parameters = jax.tree_util.tree_map(jnp.zeros_like, vs.parameters)
    parameters["visible_bias"] = jnp.full_like(parameters["visible_bias"], visible_bias)
    vs.parameters = parameters

    a = nk.operator.boson.destroy(hi, 0, cls=operator_cls)
    stats, gradient = vs.expect_and_grad(a.H @ a)

    np.testing.assert_allclose(vs.expect(a.H @ a).mean, 1.0)
    np.testing.assert_allclose(stats.mean, 1.0)
    for leaf in jax.tree_util.tree_leaves(gradient):
        np.testing.assert_allclose(leaf, 0.0, atol=1e-12)


def _fermion_setup():
    hi = nk.hilbert.SpinOrbitalFermions(4, n_fermions=2)
    cdag, c = nk.operator.fermion.create, nk.operator.fermion.destroy
    ha = sum(cdag(hi, i) @ c(hi, (i + 1) % 4) for i in range(4))
    ha = ha + ha.H.collect()
    ha = ha + 0.5 * nk.operator.fermion.number(hi, 0)
    ha = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(ha)
    vs = nk.vqs.MCState(
        nk.sampler.ExactSampler(hi), nk.models.RBM(alpha=1), n_samples=1024, seed=0
    )
    return hi, ha, vs


def test_squared_constrained_uses_oloc_squared():
    # Squared(H) on a constrained space keeps the |H_loc|² estimator, which is
    # exact as H preserves the constraint, even if H @ H cannot be formed.
    _, ha, vs = _fermion_setup()

    squared = nk.operator.Squared(ha)
    np.testing.assert_allclose(
        vs.local_estimators(squared).data, np.abs(vs.local_estimators(ha).data) ** 2
    )

    psi = vs.to_array()
    h_dense = ha.to_dense()
    stats, _ = vs.expect_and_grad(squared)
    exact = np.vdot(h_dense @ psi, h_dense @ psi).real
    np.testing.assert_allclose(stats.mean.real, exact, atol=5 * stats.error_of_mean)


def test_variance_oloc_squared_constrained():
    _, ha, vs = _fermion_setup()
    var_op = nk.observable.VarianceObservable(ha, use_Oloc_squared=True)

    psi = vs.to_array()
    h_dense = ha.to_dense()
    h_psi = h_dense @ psi
    exact = (np.vdot(h_psi, h_psi) - np.vdot(psi, h_psi) ** 2).real

    stats = vs.expect(var_op)
    np.testing.assert_allclose(stats.mean.real, exact, atol=5 * stats.error_of_mean)
    stats, _ = vs.expect_and_grad(var_op)
    np.testing.assert_allclose(stats.mean.real, exact, atol=5 * stats.error_of_mean)
