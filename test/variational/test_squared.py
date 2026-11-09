import jax
import jax.numpy as jnp
import numpy as np

import netket as nk
from netket.experimental.operator import ParticleNumberConservingFermioperator2nd


def test_adjoint_matmul_changes_sector():
    # Reproducer of #2322: the state is |1>, and the amplitudes the network
    # assigns to |0> and |2>, outside of the constraint, must not matter.
    hi = nk.hilbert.Fock(n_max=2, N=1, n_particles=1)
    vs = nk.vqs.MCState(
        nk.sampler.ExactSampler(hi), nk.models.RBM(alpha=1), n_samples=64, seed=0
    )
    vs.parameters = jax.tree_util.tree_map(jnp.zeros_like, vs.parameters)

    a = nk.operator.boson.destroy(hi, 0, cls=nk.operator.LocalOperatorJax)
    np.testing.assert_allclose(vs.expect(a.H @ a).mean, 1.0)


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
    stats, _ = vs.expect_and_grad(var_op)
    np.testing.assert_allclose(stats.mean.real, exact, atol=5 * stats.error_of_mean)
