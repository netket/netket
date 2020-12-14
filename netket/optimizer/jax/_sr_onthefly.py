import jax
import jax.numpy as jnp
from functools import partial
from netket.stats import sum_inplace as _sum_inplace
from netket.utils import n_nodes


# onthefly SR
# since we cant store O if #samples x #params is too large


def O_jvp(x, theta, v, forward_fn):
    _, res = jax.jvp(lambda p: forward_fn(p, x), (theta,), (v,))
    return res


def O_vjp(x, theta, v, forward_fn, return_vjp_fun=False, vjp_fun=None):
    if vjp_fun is None:
        _, vjp_fun = jax.vjp(forward_fn, theta, x)
    res, _ = vjp_fun(v)
    if return_vjp_fun:
        return res, vjp_fun
    else:
        return res


# calculate \langle O \rangle
def Obar(samples, theta, forward_fn, **kwargs):
    # TODO better way to get dtype
    dtype = forward_fn(theta, samples[:1])[0].dtype
    v = jax.lax.broadcast( jnp.array(1./(samples.shape[0]*n_nodes), dtype=dtype) , (samples.shape[0],))
    return O_vjp(samples, theta, v, forward_fn, **kwargs)


# calculate O^\dagger O v
def odagov(samples, theta, v, forward_fn):
    vprime = O_jvp(samples, theta, v, forward_fn)
    res = O_vjp(samples, theta, vprime.conjugate(), forward_fn)
    return jax.tree_map(jax.lax.conj, res)  # return res.conjugate()


# calculate O^\dagger \Delta O v
# where \Delta O = O-\langle O \rangle
# optional: pass jvp_fun to be reused
def odagdeltaov(samples, theta, v, forward_fn, vjp_fun=None):
    # reuse vjp_fun from O_mean below for O_vjp
    O_mean, vjp_fun = Obar(
        samples, theta, forward_fn, return_vjp_fun=True, vjp_fun=vjp_fun
    )
    vprime = O_jvp(
        samples, theta, v, forward_fn
    )  # is an array of size n_samples; each MPI rank has its own slice
    # TODO tree_dot would be nice
    # here we use jax.numpy.add to automatically promote nonhomogeneous parameters to the larger type

    # omeanv = O_mean.dot(v); is a scalar
    omeanv = jax.tree_util.tree_reduce(
        jax.numpy.add,
        jax.tree_map(jax.numpy.sum, jax.tree_multimap(jax.lax.mul, O_mean, v)),
    )
    omeanv = _sum_inplace(omeanv)  # MPI Allreduce w/ MPI_SUM

    # vprime -= omeanv (elementwise)
    vprime = vprime - jax.lax.broadcast(omeanv, vprime.shape)
    # vprime /= n_samp (elementwise)
    vprime = jax.lax.mul(
        vprime,
        jax.lax.broadcast(jnp.array(1.0 / (samples.shape[0]*n_nodes), dtype=vprime.dtype), vprime.shape),
    )

    res = O_vjp(samples, theta, vprime.conjugate(), forward_fn, vjp_fun=vjp_fun)
    res = jax.tree_map(jax.lax.conj, res)  # res = res.conjugate()
    # convert back the parameters which were promoted earlier:
    # astype alone would also work, however this raises ComplexWarning when casting complex to real, so we take the real where needed first
    res = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        res,
        v,
    )
    res = jax.tree_map(_sum_inplace, res)  # MPI Allreduce w/ MPI_SUM
    return res


def mat_vec(v, forward_fn, params, samples, diag_shift):
    # all the leaves of v need to be arrays
    # when using MPI:
    #      each rank has its own slice of samples
    res = odagdeltaov(samples, params, v, forward_fn)
    # add diagonal shift:
    shiftv = jax.tree_map(
        lambda x: jax.lax.mul(
            x, jax.lax.broadcast(jnp.array(diag_shift, dtype=x.dtype), x.shape)
        ),
        v,
    )
    # res += diag_shift * v
    res = jax.tree_multimap(jax.lax.add, res, shiftv)
    return res
