import jax
import jax.numpy as jnp
from functools import partial


# onthefly SR
# since we cant store O if #samples x #params is too large

def O_jvp(x, theta, v, vlogwf):
    _, res = jax.jvp(lambda p: vlogwf(p,x), (theta,), (v,))
    return res


def O_vjp(x, theta, v, vlogwf, return_vjp_fun=False, vjp_fun=None):
    if vjp_fun is None:
        _, vjp_fun = jax.vjp(vlogwf, theta, x)
    res, _ = vjp_fun(v)
    if return_vjp_fun:
        return res, vjp_fun
    else:
        return res


# calculate \langle O \rangle
def Obar(samples, theta, vlogwf, **kwargs):
    # TODO better way to get dtype
    dtype = vlogwf(theta, samples[:1])[0].dtype
    v = jnp.ones(samples.shape[0], dtype=dtype)/samples.shape[0]
    return O_vjp(samples, theta, v, vlogwf, **kwargs)


# calculate O^\dagger O v
# @partial(jax.jit, static_argnums=3)
def odagov(samples, theta, v, vlogwf):
    vprime = O_jvp(samples, theta, v, vlogwf)
    res = O_vjp(samples, theta, vprime.conjugate(), vlogwf)
    return jax.tree_map(jax.lax.conj, res)  # return res.conjugate()


# calculate O^\dagger \Delta O v
# where \Delta O = O-\langle O \rangle
# optional: pass jvp_fun to be reused
# TODO vjp_fun and jit??
# @partial(jax.jit, static_argnums=3)
def odagdeltaov(samples, theta, v, vlogwf, vjp_fun=None, factor=1.):
    # reuse vjp_fun from O_mean below for O_vjp
    O_mean, vjp_fun = Obar(samples, theta, vlogwf, return_vjp_fun=True, vjp_fun=vjp_fun)
    vprime = O_jvp(samples, theta, v, vlogwf) # is an array of size n_samples
    # TODO tree_dot would be nice
    omeanv = jax.tree_util.tree_reduce(jax.lax.add, jax.tree_map(jax.numpy.sum, jax.tree_multimap(jax.lax.mul, O_mean, v)))  # omeanv = O_mean.dot(v); is a scalar
    vprime = vprime - jax.lax.broadcast(omeanv, vprime.shape)  # vprime -= omeanv (elementwise)
    vprime = jax.lax.mul(vprime, jax.lax.broadcast(jnp.array(factor, dtype=vprime.dtype), vprime.shape))  # vprime *= factor (elementwise)
    res = O_vjp(samples, theta, vprime.conjugate(), vlogwf, vjp_fun=vjp_fun)
    return jax.tree_map(jax.lax.conj, res)  # return res.conjugate()
