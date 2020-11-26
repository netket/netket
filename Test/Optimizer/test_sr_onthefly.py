import pytest
import jax
import jax.numpy as jnp
import numpy as np
from netket.optimizer.jax._sr_onthefly import *

# TODO more sophisitcated example?

@partial(jax.vmap, in_axes=(None,0))
def f(params, x):
    return params['a'][0]*x[0]+params['b']*x[1]+params['c']*(x[0]*x[1])+jnp.sin(x[1]*params['a'][1])

samples = jnp.array(np.random.random((10, 2)))
params = jax.tree_map(jnp.array, {'a':[1., -4.], 'b':2., 'c':-0.55+4.33j})
v = jax.tree_map(jnp.array, {'a':[0.7, -3.9], 'b':0.3, 'c':-0.74+3j})
vprime = jnp.array(np.random.random(samples.shape[0])+1j*np.random.random(samples.shape[0]))

params_flat, conv = jax.flatten_util.ravel_pytree(params)  # promotes types automatically
v_flat, _ = jax.flatten_util.ravel_pytree(v)

def f_flat(params_flat, x):
    return f(conv(params_flat), x)

def flatten(x):
    x_flat, _ = jax.flatten_util.ravel_pytree(x)
    return x_flat

def f_flat_scalar(params, x):
    return f_flat(params, jnp.expand_dims(x, 0))[0]

ok = jax.vmap(jax.grad(f_flat_scalar, argnums=0, holomorphic=True), in_axes=(None, 0))(params_flat, samples).conjugate() # natural gradient
okmean = ok.mean(axis=0)
dok = ok - okmean

def setzero_real_params(x):
    real_ind = flatten(jax.tree_map(jax.numpy.isrealobj, params))
    return jax.ops.index_add(x, real_ind, -1j*x[real_ind].imag)




def test_f_flat():
    a = f(params, samples)
    b = f_flat(params_flat, samples)
    assert jnp.allclose(a,b)

def test_vjp():
    a = O_vjp(samples, params_flat, vprime, f_flat)
    b = flatten(O_vjp(samples, params, vprime, f))
    e = setzero_real_params(vprime @ ok)
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
    e = ok  @ v_flat
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)

def test_odagov():
    a = flatten(odagov(samples, params, v, f))
    b = odagov(samples, params_flat, v_flat, f_flat)
    e = setzero_real_params(ok.transpose().conjugate() @ (ok @ v_flat))
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)

def test_odagdeltaov():
    a = flatten(odagdeltaov(samples, params, v, f))
    b = odagdeltaov(samples, params_flat, v_flat, f_flat)
    # differnt calculation, but same result since additional terms are equal to zero
    e = setzero_real_params(dok.transpose().conjugate() @ (dok @ v_flat))
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)

def test_matvec():
    n_samp = samples.shape[0]
    diag_shift = 0.01
    a = flatten(mat_vec(v, f, params, samples, diag_shift, n_samp))
    b = mat_vec(v_flat, f_flat, params_flat, samples, diag_shift, n_samp)
    e = setzero_real_params(dok.transpose().conjugate() @ (dok @ v_flat/n_samp) + diag_shift * v_flat)
    assert jnp.allclose(a, e)
    assert jnp.allclose(b, e)
