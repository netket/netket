import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

from netket.jax import vjp_new as nkvjp

from test.common_mesh import with_explicit_meshes


def model(pars, xs):
    W, b = pars
    return jnp.sum(xs @ W + b, axis=-1)


@with_explicit_meshes(
    [
        None,
        ((2,), ("S",)),
        ((1, 2), ("S", "P")),
        ((2, 2), ("S", "P")),
    ]
)
@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(None, id="bs=None"),
        pytest.param(4, id="bs=4"),
        pytest.param(4, id="bs=7"),
    ],
)
def test_vjp_new(mesh, batch_size):
    key = jax.random.PRNGKey(0)
    M, N = 6, 100
    W = jax.random.normal(key, (M, M))  # our “parameters”
    b = jax.random.normal(key, ())  # our “parameters”
    pars = (W, b)  # model parameters
    xs = jax.random.normal(key, (N, M))  # N samples of dimension M
    v = jax.random.normal(key, (N,))  # cotangent vector of length N

    if not mesh.empty:
        if "S" in mesh.axis_names:
            xs = reshard(xs, P("S", None))
            v = reshard(v, P("S"))
        if "P" in mesh.axis_names:
            W = reshard(W, P("P"))

    # 4) Full‐dataset VJP (no batching)
    fwd_full, vjp_fun_full = jax.vjp(model, pars, xs)
    (grad_pars_full, grad_xs_full) = vjp_fun_full(v)

    # 5) Chunk‐and‐reduce VJP
    vjp_fun_nk = nkvjp(model, pars, xs, batch_size=batch_size, batch_argnums=1)
    (grad_pars_nk, grad_xs_nk) = vjp_fun_nk(v)
    jax.tree.map(np.testing.assert_allclose, grad_pars_full, grad_pars_nk)
    jax.tree.map(np.testing.assert_allclose, grad_xs_full, grad_xs_nk)

    vjp_fun_nk = nkvjp(
        model, pars, xs, return_forward=True, batch_size=batch_size, batch_argnums=1
    )
    fwd_nk, (grad_pars_nk, grad_xs_nk) = vjp_fun_nk(v)
    np.testing.assert_allclose(fwd_full, fwd_nk)
    jax.tree.map(np.testing.assert_allclose, grad_pars_full, grad_pars_nk)
    jax.tree.map(np.testing.assert_allclose, grad_xs_full, grad_xs_nk)
