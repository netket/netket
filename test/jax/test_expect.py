from netket.jax import expect

import jax
import jax.numpy as jnp

import numpy as np


def test_expect():
    def log_pdf(pars, σ):
        return jnp.abs(pars[σ]) ** 2

    def expected_fun(pars, σ, H):
        return H[σ] @ pars / pars[σ]

    key = jax.random.key(3)
    N = 10
    Ns = 30
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
