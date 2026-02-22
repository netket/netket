from functools import partial
import pytest

import numpy as np

import jax

from netket.jax.lax import reduce as nkreduce


def model(pars, xs):
    W, b = pars
    return xs @ W + b


@pytest.mark.parametrize(
    "stack", [pytest.param(False, id="no_stack"), pytest.param(True, id="stack")]
)
def test_reduce_vjp(stack):
    def chunk_grad(W, batch):
        xs_chunk, v_chunk = batch
        # vjp on the linear model w.r.t. W
        val, vjp_fun = jax.vjp(lambda W_: model(W_, xs_chunk), W)
        (gW,) = vjp_fun(v_chunk)  # gradient wrt W only
        if stack:
            return val, gW
        else:
            return gW

    key = jax.random.PRNGKey(0)
    M, N = 5, 100
    W = jax.random.normal(key, (M,))  # our “parameters”
    b = jax.random.normal(key, ())  # our “parameters”
    pars = (W, b)  # model parameters
    xs = jax.random.normal(key, (N, M))  # N samples of dimension M
    v = jax.random.normal(key, (N,))  # cotangent vector of length N

    # 4) Full‐dataset VJP (no batching)
    fwd_full, vjp_fun_full = jax.vjp(lambda W_: model(W_, xs), pars)
    (grad_full,) = vjp_fun_full(v)

    stack_outnums = (0,) if stack else ()

    # 5) Chunk‐and‐reduce VJP
    data = (xs, v)
    grad_reduce = nkreduce(partial(chunk_grad, pars), data, stack_outnums=stack_outnums)
    if stack:
        fwd_reduce, grad_reduce = grad_reduce
        np.testing.assert_allclose(fwd_full, fwd_reduce)

    jax.tree.map(np.testing.assert_allclose, grad_full, grad_reduce)

    grad_batch = nkreduce(
        partial(chunk_grad, pars), data, batch_size=4, stack_outnums=stack_outnums
    )
    if stack:
        fwd_batch, grad_batch = grad_batch
        np.testing.assert_allclose(fwd_full, fwd_batch)

    jax.tree.map(np.testing.assert_allclose, grad_full, grad_batch)
