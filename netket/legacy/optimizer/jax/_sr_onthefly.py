import jax
import jax.numpy as jnp
from functools import partial
from netket.stats import sum_inplace as _sum_inplace
from netket.utils import n_nodes

# Stochastic Reconfiguration with jvp and vjp

# Here O (Oks) is the jacobian (derivatives w.r.t. params) of the vectorised (in x) log wavefunction (forward_fn) evaluated at all samples.
# instead of computing (and storing) the full jacobian matrix jvp and vjp are used to implement the matrix vector multiplications with it.
# Expectation values are then just the mean over the leading dimension.


def tree_conj(t):
    r"""
    conjugate all complex leaves
    The real leaves are left untouched.

    t: pytree
    """
    return jax.tree_map(lambda x: jax.lax.conj(x) if jnp.iscomplexobj(x) else x, t)


def tree_dot(a, b):
    r"""
    compute the dot product of of the flattened arrays of a and b (without actually flattening)

    a, b: pytrees with the same treedef
    """
    res = jax.tree_util.tree_reduce(
        jax.numpy.add, jax.tree_map(jax.numpy.sum, jax.tree_multimap(jax.lax.mul, a, b))
    )
    # convert shape from () to (1,)
    # this is needed for automatic broadcasting to work also when transposed with linear_transpose
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


def O_jvp(x, params, v, forward_fn):
    # TODO apply the transpose of sum_inplace (allreduce) to v here
    # in order to get correct transposition with MPI
    _, res = jax.jvp(lambda p: forward_fn(p, x), (params,), (v,))
    return res


def O_vjp(x, params, v, forward_fn, return_vjp_fun=False, vjp_fun=None, allreduce=True):
    if vjp_fun is None:
        _, vjp_fun = jax.vjp(forward_fn, params, x)
    res, _ = vjp_fun(v)

    if allreduce:
        res = jax.tree_map(_sum_inplace, res)

    if return_vjp_fun:
        return res, vjp_fun
    else:
        return res


def O_mean(samples, params, forward_fn, **kwargs):
    r"""
    compute \langle O \rangle
    i.e. the mean of the rows of the jacobian of forward_fn
    """
    dtype = jax.eval_shape(forward_fn, params, samples).dtype
    v = jnp.ones(samples.shape[0], dtype=dtype) * (1.0 / (samples.shape[0] * n_nodes))
    return O_vjp(samples, params, v, forward_fn, **kwargs)


def OH_w(samples, params, w, forward_fn, **kwargs):
    r"""
    compute  O^H w
    (where ^H is the hermitian transpose)
    """
    # O^H w = (w^H O)^H
    # The transposition of the 1D arrays is omitted in the implementation:
    # (w^H O)^H -> (w* O)*

    # TODO The allreduce in O_vjp could be deferred until after the tree_cast
    # where the amount of data to be transferred would potentially be smaller
    res = tree_conj(O_vjp(samples, params, w.conjugate(), forward_fn, **kwargs))
    #
    return tree_cast(res, params)


def Odagger_DeltaO_v(samples, params, v, forward_fn, vjp_fun=None):
    r"""
    compute \langle O^\dagger \DeltaO \rangle v
    where \DeltaO = O - \langle O \rangle

    optional: pass jvp_fun to be reused
    """

    # here the allreduce is deferred until after the dot product,
    # where only scalars instead of vectors have to be summed
    # the vjp_fun is returned so that it can be reused in OH_w below
    omean, vjp_fun = O_mean(
        samples,
        params,
        forward_fn,
        return_vjp_fun=True,
        vjp_fun=vjp_fun,
        allreduce=False,
    )
    omeanv = tree_dot(omean, v)  # omeanv = omean.dot(v); is a scalar
    omeanv = _sum_inplace(omeanv)  # MPI Allreduce w/ MPI_SUM

    # v_tilde is an array of size n_samples; each MPI rank has its own slice
    v_tilde = O_jvp(samples, params, v, forward_fn)
    # v_tilde -= omeanv (elementwise):
    v_tilde = v_tilde - omeanv
    # v_tilde /= n_samples (elementwise):
    v_tilde = v_tilde * (1.0 / (samples.shape[0] * n_nodes))

    return OH_w(samples, params, v_tilde, forward_fn, vjp_fun=vjp_fun)


# TODO allow passing vjp_fun from e.g. a preceding gradient calculation with the same samples
# and optionally return vjp_fun so that it can be reused in subsequent calls
# TODO block the computations (in the same way as done with MPI) if memory consumtion becomes an issue
def mat_vec(v, forward_fn, params, samples, diag_shift):
    r"""
    compute (S + diag_shift) v

    where the elements of S are given by
    S_kl = \langle O_k^\dagger \DeltaO_ \rangle
    \DeltaO_k = O_k - \langle O_k \rangle
    and O_k (operator) is derivative of the log wavefunction w.r.t parameter k
    The expectation values are calculated as mean over the samples

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
