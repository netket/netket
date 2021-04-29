import pytest
import netket.legacy as nk

if not nk.utils.jax_available:
    pytest.skip("skipping jax-only SR-onthefly tests", allow_module_level=True)

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from jax.scipy.sparse.linalg import cg
from netket.legacy.optimizer.jax._sr_onthefly import *
from netket.legacy.optimizer.jax.stochastic_reconfiguration import (
    _jax_cg_solve_onthefly,
)

pytestmark = pytest.mark.legacy


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

v_tilde = jnp.array(np.random.random(n_samp) + 1j * np.random.random(n_samp))


def flatten(x):
    x_flat, _ = jax.flatten_util.ravel_pytree(x)
    return x_flat


def toreal(x):
    if jnp.iscomplexobj(x):
        return jnp.array([x.real, x.imag])
    else:
        return x


def tree_toreal(x):
    return jax.tree_map(toreal, x)


def tree_toreal_flat(x):
    return flatten(tree_toreal(x))


# invert the transformation tree_toreal_flat using linear_transpose (AD)
def reassemble_complex(x, fun=tree_toreal_flat, target=params):
    # target: a tree with the expected shape and types of the result
    (res,) = jax.linear_transpose(fun, target)(x)
    res = tree_conj(res)
    # fix the dtypes:
    return tree_cast(res, target)


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


def test_reassemble_complex():
    assert tree_allclose(params, reassemble_complex(tree_toreal_flat(params)))


def test_vjp():
    actual = O_vjp(samples, params, v_tilde, f)
    expected = tree_conj(reassemble_complex((v_tilde @ ok_real).real))
    assert tree_allclose(actual, expected)


def test_mean():
    actual = O_mean(samples, params, f)
    expected = tree_conj(reassemble_complex(okmean_real.real))
    assert tree_allclose(actual, expected)


def test_OH_w():
    actual = OH_w(samples, params, v_tilde, f)
    expected = reassemble_complex((ok_real.conjugate().transpose() @ v_tilde).real)
    assert tree_allclose(actual, expected)


def test_jvp():
    actual = O_jvp(samples, params, v, f)
    expected = ok_real @ v_real_flat
    assert tree_allclose(actual, expected)


def test_Odagger_DeltaO_v():
    actual = Odagger_DeltaO_v(samples, params, v, f)
    expected = reassemble_complex(S_real @ v_real_flat)
    assert tree_allclose(actual, expected)


def test_matvec():
    diag_shift = 0.01
    actual = mat_vec(v, f, params, samples, diag_shift)
    expected = reassemble_complex(S_real @ v_real_flat + diag_shift * v_real_flat)
    assert tree_allclose(actual, expected)


def test_matvec_linear_transpose():
    w = v
    (actual,) = jax.linear_transpose(
        lambda v_: mat_vec(v_, f, params, samples, 0.0), v
    )(w)
    # use that S is hermitian:
    # S^T = (O^H O)^T = O^T O* = (O^H O)* = S*
    # S^T w = S* w = (S w*)*
    expected = tree_conj(mat_vec(tree_conj(w), f, params, samples, 0.0))
    # (expected,) = jax.linear_transpose(lambda v_: reassemble_complex(S_real @ tree_toreal_flat(v_)), v)(v)
    assert tree_allclose(actual, expected)


def test_cg():
    # also tests if matvec can be jitted and be differentiated with AD
    diag_shift = 0.001
    sparse_tol = 1.0e-5
    sparse_maxiter = None
    actual = _jax_cg_solve_onthefly(
        v, f, params, samples, grad, diag_shift, sparse_tol, sparse_maxiter
    )

    def mv_real(v):
        return S_real @ v + diag_shift * v

    expected = reassemble_complex(
        cg(
            mv_real,
            grad_real_flat,
            x0=v_real_flat,
            tol=sparse_tol,
            maxiter=sparse_maxiter,
        )[0]
    )
    assert tree_allclose(actual, expected)
