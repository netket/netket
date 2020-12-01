import pytest
from jax.config import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from jax.scipy.sparse.linalg import cg
from netket.optimizer.jax._sr_onthefly import *
from netket.optimizer.jax.stochastic_reconfiguration import _jax_cg_solve_onthefly

# TODO more sophisitcated example?


@partial(jax.vmap, in_axes=(None, 0))
def f(params, x):
    return (
        params["a"][0] * x[0]
        + params["b"] * x[1]
        + params["c"] * (x[0] * x[1])
        + jnp.sin(x[1] * params["a"][1])
    )


n_samp = 25
samples = jnp.array(np.random.random((n_samp, 2)))

params = {
    "a": jnp.array([1.0 + 1.2j, -4.0 + -0.7j]),
    "b": jnp.array(2.0),
    "c": jnp.array(-0.55 + 4.33j),
}

v = {
    "a": jnp.array([0.7 + 1.1j, -3.9 + 3.5j]),
    "b": jnp.array(0.3),
    "c": jnp.array(-0.74 + 3j),
}

grad = {
    "a": jnp.array([0.01 + 23.0j, 1.99 + -7.2j]),
    "b": jnp.array(-0.231),
    "c": jnp.array(1.22 - 0.45j),
}

vprime = jnp.array(np.random.random(n_samp) + 1j * np.random.random(n_samp))


def flatten(x):
    x_flat, _ = jax.flatten_util.ravel_pytree(x)
    return x_flat


def toreal(x):
    if jnp.iscomplexobj(x):
        # workaround for
        # NotImplementedError: Transpose rule (for reverse-mode differentiation) for 'imag' not implemented
        return jnp.array(
            [x.real, (-1j * x).real]
        )  # need to use sth which jax thinks its a leaf
    else:
        return x


def tree_toreal(x):
    return jax.tree_map(toreal, x)


def tree_toreal_flat(x):
    return flatten(tree_toreal(x))


# invert the trafo using linear_transpose (AD)
def reassemble_complex(x, fun=tree_toreal_flat, target=params):
    # target: some tree with the shape and types we want
    _lt = jax.linear_transpose(fun, target)
    # jax gradient is actually the conjugated one, so we need to fix it:
    res = jax.tree_map(jax.lax.conj, _lt(x)[0])
    # also fix the dtypes:
    return jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        res,
        target,
    )


def f_real_flat(p, samples):
    return f(reassemble_complex(p), samples)


def f_real_flat_scalar(params, x):
    return f_real_flat(params, jnp.expand_dims(x, 0))[0]


# same as in nk.machine.Jax R->C
@partial(jax.vmap, in_axes=(None, 0))
def grads_real(params, x):
    r = jax.grad(lambda pars, v: f_real_flat_scalar(pars, v).real)(params, x)
    i = jax.grad(lambda pars, v: f_real_flat_scalar(pars, v).imag)(params, x)
    return jax.lax.complex(r, i)


params_real_flat = tree_toreal_flat(params)
grad_real_flat = tree_toreal_flat(grad)
v_real_flat = tree_toreal_flat(v)

ok_real = grads_real(params_real_flat, samples)
okmean_real = ok_real.mean(axis=0)
dok_real = ok_real - okmean_real
S_real = (dok_real.conjugate().transpose() @ dok_real / n_samp).real


def tree_allclose(t1, t2):
    t = jax.tree_multimap(jnp.allclose, t1, t2)
    return all(jax.tree_util.tree_flatten(t)[0])


def tree_conj(t):
    return jax.tree_map(jax.lax.conj, t)


def test_reassemble_complex():
    assert tree_allclose(params, reassemble_complex(tree_toreal_flat(params)))


# O_vjp and O_jvp are actually conjugated compared to ok
# (however the final result after matvec is still correct)
# this way we can avoid two conjugations


def test_vjp():
    a = O_vjp(samples, params, vprime, f)
    e = reassemble_complex((vprime @ ok_real).real)
    assert tree_allclose(tree_conj(a), e)


def test_obar():
    a = Obar(samples, params, f)
    e = reassemble_complex(okmean_real.real)
    assert tree_allclose(tree_conj(a), e)


def test_jvp():
    a = O_jvp(samples, params, v, f)
    e = ok_real @ v_real_flat
    assert tree_allclose(a, e)


def test_odagov():
    a = odagov(samples, params, v, f)
    e = reassemble_complex(
        (ok_real.conjugate().transpose() @ ok_real @ v_real_flat).real
    )
    assert tree_allclose(a, e)


def test_odagdeltaov():
    a = odagdeltaov(samples, params, v, f, factor=1.0 / n_samp)
    e = reassemble_complex(S_real @ v_real_flat)
    assert tree_allclose(a, e)


def test_matvec():
    diag_shift = 0.01
    a = mat_vec(v, f, params, samples, diag_shift, n_samp)
    e = reassemble_complex(S_real @ v_real_flat + diag_shift * v_real_flat)
    assert tree_allclose(a, e)


def test_cg():
    # also tests if matvec can be jitted and be differentiated with AD
    diag_shift = 0.001
    sparse_tol = 1.0e-5
    sparse_maxiter = None
    a = _jax_cg_solve_onthefly(
        v, f, params, samples, grad, diag_shift, n_samp, sparse_tol, sparse_maxiter
    )

    def mv_real(v):
        return S_real @ v + diag_shift * v

    e = reassemble_complex(
        cg(
            mv_real,
            grad_real_flat,
            x0=v_real_flat,
            tol=sparse_tol,
            maxiter=sparse_maxiter,
        )[0]
    )
    assert tree_allclose(a, e)
