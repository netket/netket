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


samples = jnp.array(np.random.random((10, 2)))
n_samp = samples.shape[0]

params = {
    "a": jnp.array([1.0, -4.0]),
    "b": jnp.array(2.0),
    "c": jnp.array(-0.55 + 4.33j),
}

v = {"a": jnp.array([0.7, -3.9]), "b": jnp.array(0.3), "c": jnp.array(-0.74 + 3j)}

grad = {
    "a": jnp.array([0.01, 1.99]),
    "b": jnp.array(-0.231),
    "c": jnp.array(1.22 - 0.45j),
}

vprime = jnp.array(
    np.random.random(samples.shape[0]) + 1j * np.random.random(samples.shape[0])
)

# promotes types automatically
params_flat, conv = jax.flatten_util.ravel_pytree(params)


def flatten(x):
    x_flat, _ = jax.flatten_util.ravel_pytree(x)
    return x_flat


v_flat = flatten(v)
grad_flat = flatten(grad)


def f_flat(params_flat, x):
    return f(conv(params_flat), x)


def f_flat_scalar(params, x):
    return f_flat(params, jnp.expand_dims(x, 0))[0]


ok = jax.vmap(jax.grad(f_flat_scalar, argnums=0, holomorphic=True), in_axes=(None, 0))(
    params_flat, samples
).conjugate()  # natural gradient
okmean = ok.mean(axis=0)
dok = ok - okmean
S = dok.conjugate().transpose() @ dok / n_samp

# need to use isreal (and not isrealobj) here to get an array of all true or all false for each leaf when flattening
# so that real_ind has size n_params, and not just n_leaves
real_ind = flatten(jax.tree_map(jax.numpy.isreal, params))

assert real_ind.shape == params_flat.shape


def setzero_imag_part_of_real_params(x):
    # workaround for
    # NotImplementedError: Transpose rule (for reverse-mode differentiation) for 'imag' not implemented
    return jax.ops.index_add(x, real_ind, -1j * (-1j * x[real_ind]).real)


# flatten params to a vector of real params


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


params_real_flat = tree_toreal_flat(params)

# invert the trafo using linear_transpose (AD)
def reassemble_complex(x, fun=tree_toreal_flat, target=params):
    # target: some tree with the shape and types we want
    _lt = jax.linear_transpose(fun, target)
    # jax gradient is actually the conjugated one, so we need to fix it:
    return jax.tree_map(jax.lax.conj, _lt(x)[0])


def f_real_flat(p, samples):
    return f(reassemble_complex(p), samples)


grad_real_flat = tree_toreal_flat(grad)
v_real_flat = tree_toreal_flat(v)


def f_real_flat_scalar(params, x):
    return f_real_flat(params, jnp.expand_dims(x, 0))[0]


def _rea(x):
    re, im = x
    return re + 1j * im


@partial(jax.vmap, in_axes=(None, 0))
def grads_real(params, x):
    r = jax.grad(lambda pars, v: f_real_flat_scalar(pars, v).real)(params_real_flat, x)
    i = jax.grad(lambda pars, v: f_real_flat_scalar(pars, v).imag)(params_real_flat, x)
    return _rea((r, i))


ok_real = grads_real(params_real_flat, samples)
okmean_real = ok_real.mean(axis=0)
dok_real = ok_real - okmean_real
S_real = dok_real.conjugate().transpose() @ dok_real / n_samp


def test_f_flat():
    a = f(params, samples)
    b = f_flat(params_flat, samples)
    assert jnp.allclose(a, b)


def test_vjp():
    a = O_vjp(samples, params_flat, vprime, f_flat)
    b = flatten(O_vjp(samples, params, vprime, f))
    e = setzero_imag_part_of_real_params(vprime @ ok)
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)


def test_obar():
    a = flatten(Obar(samples, params, f))
    b = Obar(samples, params_flat, f_flat)
    e = okmean
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)


def test_jvp():
    a = O_jvp(samples, params, v, f)
    b = O_jvp(samples, params_flat, v_flat, f_flat)
    e = ok @ v_flat
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)


def test_odagov():
    a = flatten(odagov(samples, params, v, f))
    b = odagov(samples, params_flat, v_flat, f_flat)
    e = setzero_imag_part_of_real_params(ok.transpose().conjugate() @ (ok @ v_flat))
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)


def test_odagdeltaov():
    a = flatten(odagdeltaov(samples, params, v, f))
    b = odagdeltaov(samples, params_flat, v_flat, f_flat)
    # differnt calculation, but same result since additional terms are equal to zero
    e = setzero_imag_part_of_real_params(dok.transpose().conjugate() @ (dok @ v_flat))
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)


def test_matvec():
    diag_shift = 0.01
    a = flatten(mat_vec(v, f, params, samples, diag_shift, n_samp))
    b = mat_vec(v_flat, f_flat, params_flat, samples, diag_shift, n_samp)
    e = setzero_imag_part_of_real_params(
        dok.transpose().conjugate() @ (dok @ v_flat / n_samp) + diag_shift * v_flat
    )
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)


def test_cg():
    # also tests if matvec can be jitted and be differentiated with AD
    diag_shift = 0.001
    sparse_tol = 1.0e-5
    sparse_maxiter = None
    a = flatten(
        _jax_cg_solve_onthefly(
            v, f, params, samples, grad, diag_shift, n_samp, sparse_tol, sparse_maxiter
        )
    )
    b = _jax_cg_solve_onthefly(
        v_flat,
        f_flat,
        params_flat,
        samples,
        grad_flat,
        diag_shift,
        n_samp,
        sparse_tol,
        sparse_maxiter,
    )

    def mv(v):
        return setzero_imag_part_of_real_params(S @ v + diag_shift * v)

    e = setzero_imag_part_of_real_params(
        cg(mv, grad_flat, x0=v_flat, tol=sparse_tol, maxiter=sparse_maxiter)[0]
    )

    def mv_real(v):
        # _compose_result_real also takes real here
        return (S_real @ v + diag_shift * v).real

    e2 = flatten(
        reassemble_complex(
            cg(
                mv_real,
                grad_real_flat,
                x0=v_real_flat,
                tol=sparse_tol,
                maxiter=sparse_maxiter,
            )[0]
        )
    )
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)
    assert jnp.allclose(e, e2)
