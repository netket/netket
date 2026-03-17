from netket.jax import expect

import jax
import jax.numpy as jnp

import numpy as np
import pytest

import netket as nk
import netket.experimental as nkx

from test.common_mesh import with_meshes


def test_expect():
    def log_pdf(pars, σ):
        return jnp.abs(pars[σ]) ** 2

    def expected_fun(pars, σ, H):
        return H[σ] @ pars / pars[σ]

    key = jax.random.key(3)
    N = 10
    Ns = jax.device_count() * 20
    psi = jax.random.normal(key, (N,))
    H = jax.random.normal(key, (N, N))
    x = jax.random.choice(key, N, (Ns,), replace=True, p=jnp.abs(psi) ** 2)

    res, _ = expect(log_pdf, expected_fun, psi, x, H)

    assert res.shape == ()
    np.testing.assert_allclose(expected_fun(psi, x, H).mean(), res, atol=1e-8)

    res, _ = expect(log_pdf, expected_fun, psi, x, H, chunk_size=10)
    np.testing.assert_allclose(expected_fun(psi, x, H).mean(), res, atol=1e-8)

    grad_psi = jax.grad(lambda p, x: expect(log_pdf, expected_fun, p, x, H)[0])(psi, x)
    grad_psi_chunked = jax.grad(
        lambda p, x: expect(log_pdf, expected_fun, p, x, H, chunk_size=10)[0]
    )(psi, x)
    np.testing.assert_allclose(grad_psi, grad_psi_chunked, atol=1e-8)

    # TODO: check analytical gradient as well
    # grad_analytic = (H[x]@psi) / psi[x]
    # print(grad_psi.shape, grad_analytic.shape)
    # np.testing.assert_allclose(grad_psi, grad_analytic, atol=1e-8)


@with_meshes(auto=[((2,), ("S",))])
@pytest.mark.parametrize(
    "operator_kind",
    ["FermionOperator2nd", "PNC", "PNC_spin", "FermiHubbardJax"],
)
@pytest.mark.parametrize("chunk_size", [8])
def test_expect_fermionic_operators_chunked_sharding_regression(
    mesh, operator_kind, chunk_size
):
    g = nk.graph.Hypercube(length=2, n_dim=2, pbc=False)
    hi = nk.hilbert.SpinOrbitalFermions(g.n_nodes, s=1 / 2, n_fermions_per_spin=(2, 2))

    t = 1.0
    U = 0.01
    ham_generic = 0.0
    for sz in (1, -1):
        for u, v in g.edges():
            ham_generic += -t * nk.operator.fermion.create(
                hi, u, sz=sz
            ) @ nk.operator.fermion.destroy(
                hi, v, sz=sz
            ) - t * nk.operator.fermion.create(
                hi, v, sz=sz
            ) @ nk.operator.fermion.destroy(
                hi, u, sz=sz
            )
    for u in g.nodes():
        ham_generic += (
            U
            * nk.operator.fermion.number(hi, u, sz=1)
            @ nk.operator.fermion.number(hi, u, sz=-1)
        )

    if operator_kind == "FermionOperator2nd":
        ham = ham_generic
    elif operator_kind == "PNC":
        ham = nkx.operator.ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(
            ham_generic
        )
    elif operator_kind == "PNC_spin":
        ham = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(
            ham_generic
        )
    elif operator_kind == "FermiHubbardJax":
        ham = nk.operator.FermiHubbardJax(hilbert=hi, graph=g, t=t, U=U)
    else:
        raise ValueError(f"Unknown operator kind {operator_kind}.")

    sa = nk.sampler.MetropolisFermionHop(
        hi, graph=g, n_chains=8, sweep_size=8, spin_symmetric=True
    )
    ma = nk.models.Slater2nd(hi)
    vs = nk.vqs.MCState(sa, ma, n_discard_per_chain=2, n_samples=16)
    vs.chunk_size = chunk_size

    res = vs.expect(ham)
    assert jnp.isfinite(res.mean)


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
@with_meshes(auto=[((2,), ("S",))])
def test_apply_chunked_pvary_argnums_regression(mesh):
    meta = nk.jax.COOArray(
        jnp.array([[1], [3]], dtype=jnp.int32),
        jnp.array([3, 7], dtype=jnp.int32),
        (4,),
        fill_value=jnp.array(0, dtype=jnp.int32),
    )
    x = jnp.array([[1], [3], [1], [3]], dtype=jnp.int32)
    pars = jnp.array(2, dtype=jnp.int32)

    def f(pars, x, meta):
        return pars * jax.vmap(lambda xi: meta[(xi,)])(x)

    chunked_default = nk.jax.apply_chunked(
        f,
        in_axes=(None, 0, None),
        chunk_size=2,
        axis_0_is_sharded=True,
    )
    chunked_selective = nk.jax.apply_chunked(
        f,
        in_axes=(None, 0, None),
        chunk_size=2,
        axis_0_is_sharded=True,
        pvary_argnums=(0,),
    )

    with pytest.raises(ValueError, match="pvary"):
        chunked_default(pars, x, meta)

    y = chunked_selective(pars, x, meta)
    np.testing.assert_array_equal(y, np.array([[6], [14], [6], [14]], dtype=np.int32))
