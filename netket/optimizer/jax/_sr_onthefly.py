import jax
import jax.numpy as jnp
from functools import partial
from netket.stats import sum_inplace as _sum_inplace
from netket.utils import n_nodes


# TODO ...


def tree_conj(t):
    r"""
    conjugate all complex leaves
    The real leaves are left untouched.

    t: pytree
    """
    return jax.tree_map(lambda x: jax.lax.conj(x) if jnp.iscomplexobj(x) else x, t)


def tree_dot(a, b):
    r"""
    compute the dot product of a and b
    TODO ...

    a, b: pytrees with the same treedef
    """
    res = jax.tree_util.tree_reduce(
        jax.numpy.add, jax.tree_map(jax.numpy.sum, jax.tree_multimap(jax.lax.mul, a, b))
    )
    # convert shape from () to (1,)
    return jnp.expand_dims(res, 0)


def tree_cast(x, target):
    r"""
    Cast each leaf of x to the dtype of the corresponding leaf in target.
    The imaginary part of complex leaves which are cast to real is discarded

    x: a pytree with arrays as leaves
    target: a pytree with the same treedef as x where only the dtypes of the leaves are accessed
    """
    # astype alone would also work, however that raises ComplexWarning when casting complex to real
    # therefore the real is taken first where needed
    return jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        x,
        target,
    )


def tree_axpy(a, x, y):
    r"""
    compute a * x + y

    a: scalar
    x, y: pytrees with the same treedef
    """
    return jax.tree_multimap(lambda x_, y_: a * x_ + y_, x, y)


def O_jvp(x, theta, v, forward_fn):
    # TODO apply the transpose of sum_inplace (allreduce) to v here
    # in order to get correct transposition with MPI
    _, res = jax.jvp(lambda p: forward_fn(p, x), (theta,), (v,))
    return res


def O_vjp(x, theta, v, forward_fn, return_vjp_fun=False, vjp_fun=None, allreduce=True):
    if vjp_fun is None:
        _, vjp_fun = jax.vjp(forward_fn, theta, x)
    res, _ = vjp_fun(v)

    if allreduce:
        res = jax.tree_map(_sum_inplace, res)

    if return_vjp_fun:
        return res, vjp_fun
    else:
        return res


def O_mean(samples, theta, forward_fn, **kwargs):
    r"""
    compute \langle O \rangle

    TODO ...
    """
    dtype = forward_fn(theta, samples[:1])[0].dtype
    v = jnp.ones(samples.shape[0], dtype=dtype) * (1.0 / (samples.shape[0] * n_nodes))
    return O_vjp(samples, theta, v, forward_fn, **kwargs)


def Odagger_w(samples, theta, w, forward_fn, **kwargs):
    r"""
    compute  O^\dagger w

    TODO ...
    """
    # O^H w = (w^H O)^H
    # The transposition of the 1D arrays is omitted in the implementation:
    # (w^H O)^H -> (w* O)*

    # TODO The allreduce in O_vjp could be deferred until after the tree_cast
    # where the amount of data to be transferred would potentially be smaller
    res = tree_conj(O_vjp(samples, theta, w.conjugate(), forward_fn, **kwargs))
    # TODO ...
    return tree_cast(res, theta)


def Odagger_O_v(samples, theta, v, forward_fn):
    r"""
    compute  \langle O^\dagger O \rangle v

    TODO ...
    """
    v_tilde = O_jvp(samples, theta, v, forward_fn)
    v_tilde = v_tilde * (1.0 / (samples.shape[0] * n_nodes))
    return Odagger_w(samples, theta, v_tilde, forward_fn)


def Odagger_DeltaO_v(samples, theta, v, forward_fn, vjp_fun=None):
    r"""
    compute \langle O^\dagger \DeltaO \rangle v
    where \DeltaO = O - \langle O \rangle

    optional: pass jvp_fun to be reused

    TODO ...
    """

    # here the allreduce is deferred until after the dot product,
    # where only scalars instead of vectors have to be summed
    # the vjp_fun is returned so that it can be reused for O_vjp below
    omean, vjp_fun = O_mean(
        samples,
        theta,
        forward_fn,
        return_vjp_fun=True,
        vjp_fun=vjp_fun,
        allreduce=False,
    )
    omeanv = tree_dot(omean, v)  # omeanv = omean.dot(v); is a scalar
    omeanv = _sum_inplace(omeanv)  # MPI Allreduce w/ MPI_SUM

    # v_tilde is an array of size n_samples; each MPI rank has its own slice
    v_tilde = O_jvp(samples, theta, v, forward_fn)
    # v_tilde -= omeanv (elementwise):
    v_tilde = v_tilde - omeanv
    # v_tilde /= n_samples (elementwise):
    v_tilde = v_tilde * (1.0 / (samples.shape[0] * n_nodes))

    return Odagger_w(samples, theta, v_tilde, forward_fn, vjp_fun=vjp_fun)


# TODO allow passing vjp_fun from e.g. a preceding gradient calculation with the same samples
# TODO optionally return vjp_fun so that it can be reused in subsequent calls
def mat_vec(v, forward_fn, params, samples, diag_shift):
    r"""
    compute (S + diag_shift) v
    where S = \langle O^\dagger \DeltaO \rangle
    \DeltaO = O - \langle O \rangle
    TODO ...

    v: a pytree with the same structure as params
    forward_fn(params, x): a vectorised function returning the logarithm of the wavefunction for each configuration in x
    params: pytree of parameters with arrays as leaves
    samples: an array of samples (when using MPI each rank has its own slice of samples)
    diag_shift: a scalar diagonal shift
    """

    res = Odagger_DeltaO_v(samples, params, v, forward_fn)
    # add diagonal shift:
    res = tree_axpy(diag_shift, v, res)  # res += diag_shift * v
    return res
