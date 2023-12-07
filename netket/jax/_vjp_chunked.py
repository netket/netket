from functools import partial

import jax
from jax.tree_util import Partial

from netket.jax import (
    compose,
    scanmap,
    scan_append_reduce,
    vjp as nkvjp,
)
from netket.utils import HashablePartial
from netket.utils import config
from netket.jax.sharding import sharding_decorator

from ._scanmap import _multimap
from ._chunk_utils import _chunk as _tree_chunk, _unchunk as _tree_unchunk


def _trash_tuple_elements(t, nums=()):
    assert isinstance(t, tuple)
    if isinstance(nums, int):
        nums = (nums,)
    return tuple(r for i, r in enumerate(t) if i not in nums)


def _vjp(fun, cotangents, *primals, nondiff_argnums=(), conjugate=False):
    # we pass a closure to vjp, capturing the nondiff_argnums
    # this is necessary to avoid errors when using integer arguments
    # resulting in float0 tangents, which nkvjp tries to conjugate, resulting in an error
    # If we were to use the standard jax.vjp we could just trash the output at the end...

    diff_args = tuple(a for i, a in enumerate(primals) if i not in nondiff_argnums)
    nondiff_args = tuple(a for i, a in enumerate(primals) if i in nondiff_argnums)

    def _fun(nondiff_argnums, nondiff_args, *diff_args):
        n_args = len(nondiff_args) + len(diff_args)
        it_nondiff = iter(nondiff_args)
        it_diff = iter(diff_args)
        args = tuple(
            next(it_nondiff) if i in nondiff_argnums else next(it_diff)
            for i in range(n_args)
        )
        return fun(*args)

    y, vjp_fun = nkvjp(
        partial(_fun, nondiff_argnums, nondiff_args), *diff_args, conjugate=conjugate
    )
    res = vjp_fun(cotangents)
    return (y,) + res


def __vjp_fun_chunked(
    fun,
    primals,
    cotangents,
    chunk_argnums,
    nondiff_argnums,
    chunk_size,
    conjugate,
    _vjp,
    _append_cond_fun,
):
    append_cond = _append_cond_fun(primals, nondiff_argnums, chunk_argnums)
    scan_fun = partial(scan_append_reduce, append_cond=append_cond)
    primals = tuple(
        _tree_chunk(p, chunk_size) if i in chunk_argnums else p
        for i, p in enumerate(primals)
    )
    cotangents = _tree_chunk(cotangents, chunk_size)
    # cotangents, and whatever requested in primals; +2 since 0 is the function, and 1 is cotangents
    argnums = (1,) + tuple(map(lambda x: x + 2, chunk_argnums))
    res = scanmap(
        partial(_vjp, nondiff_argnums=nondiff_argnums, conjugate=conjugate),
        scan_fun=scan_fun,
        argnums=argnums,
    )(fun, cotangents, *primals)

    return _multimap(lambda c, l: _tree_unchunk(l) if c else l, append_cond, res)


def _gen_append_cond_vjp(primals, nondiff_argnums, chunk_argnums):
    diff_argnums = filter(lambda i: i not in nondiff_argnums, range(len(primals)))
    return tuple(map(lambda i: i in chunk_argnums, diff_argnums))


_gen_append_cond_value_vjp = compose(lambda t: (True,) + t, _gen_append_cond_vjp)

_vjp_fun_chunked = partial(
    __vjp_fun_chunked,
    _vjp=compose(lambda yr: yr[1:], _vjp),
    _append_cond_fun=_gen_append_cond_vjp,
)
_value_and_vjp_fun_chunked = compose(
    lambda yr: (yr[0], yr[1:]),
    partial(__vjp_fun_chunked, _vjp=_vjp, _append_cond_fun=_gen_append_cond_value_vjp),
)


def check_chunk_size(chunk_argnums, chunk_size, *primals):
    if chunk_size is None:
        return None
    else:
        n_elements = jax.tree_util.tree_leaves(primals[chunk_argnums[0]])[0].shape[0]
        # check that they are all the same size
        chunk_leaves = jax.tree_util.tree_leaves([primals[i] for i in chunk_argnums])
        if not all(map(lambda x: x.shape[0] == n_elements, chunk_leaves)):
            raise ValueError(
                "The chunked arguments have inconsistent leading array dimensions"
            )
        if chunk_size >= n_elements:
            return None
        else:
            return chunk_size


def _vjp_chunked(
    fun,
    has_aux,
    chunk_argnums,
    chunk_size,
    nondiff_argnums,
    return_forward,
    conjugate,
):
    assert chunk_size is not None

    if has_aux:
        raise NotImplementedError
    else:
        return HashablePartial(
            _value_and_vjp_fun_chunked if return_forward else _vjp_fun_chunked,
            fun,
            chunk_argnums=chunk_argnums,
            nondiff_argnums=nondiff_argnums,
            chunk_size=chunk_size,
            conjugate=conjugate,
        )


@partial(
    jax.jit,
    static_argnames=(
        "fun",
        "has_aux",
        "chunk_argnums",
        "chunk_size",
        "nondiff_argnums",
        "return_forward",
        "conjugate",
    ),
)
def vjp_chunked(
    fun,
    *primals,
    has_aux=False,
    chunk_argnums=(),
    chunk_size=None,
    nondiff_argnums=(),
    return_forward=False,
    conjugate=False,
):
    """calculate the vjp in small chunks for a function where the leading dimension of the output only depends on the leading dimension of some of the arguments

    .. note::
        If experimental sharing is activated, the chunk_argnums are assumed to be sharded (not replicated) among devices.

    Args:
        fun: Function to be differentiated. It must accept chunks of size chunk_size of the primals in chunk_argnums.
        primals:  A sequence of primal values at which the Jacobian of ``fun`` should be evaluated.
        has_aux: Optional, bool. Only False is implemented. Indicates whether ``fun`` returns a pair where the
           first element is considered the output of the mathematical function to be
           differentiated and the second element is auxiliary data. Default False.
        chunk_argnums: an integer or tuple of integers indicating the primals which should be chunked.
            The leading dimension of each of the primals indicated must be the same as the output of fun.
        chunk_size: an integer indicating the size of the chunks over which the vjp is computed.
            It must be a integer divisor of the primals specified in chunk_argnums.
        nondiff_argnums: an integer or tuple of integers indicating the primals which should not be differentiated with.
            Specifying the arguments which are not needed should increase performance.
        return_forward: whether the returned function should also return the output of the forward pass

    Returns:
        a function corresponding to the vjp_fun returned by an equivalent ``jax.vjp(fun, *primals)[1]``` call
        which computes the vjp in chunks (recomputing the forward pass every time on subsequent calls).
        If return_forward=True the vjp_fun returned returns a tuple containing the output of the forward pass and the vjp.


    Example:
        >>> import jax
        >>> from netket.jax import vjp_chunked
        >>> from functools import partial
        >>>
        >>> @partial(jax.vmap, in_axes=(None, 0))
        ... def f(p, x):
        ...     return jax.lax.log(p.dot(jax.lax.sin(x)))
        >>>
        >>> k = jax.random.split(jax.random.PRNGKey(123), 4)
        >>> p = jax.random.uniform(k[0], shape=(8,))
        >>> v = jax.random.uniform(k[1], shape=(8,))
        >>> X = jax.random.uniform(k[2], shape=(1024,8))
        >>> w = jax.random.uniform(k[3], shape=(1024,))
        >>>
        >>> vjp_fun_chunked = vjp_chunked(f, p, X, chunk_argnums=(1,), chunk_size=32, nondiff_argnums=1)
        >>> vjp_fun = jax.vjp(f, p, X)[1]
        >>>
        >>> vjp_fun_chunked(w)
        (Array([106.76358917, 113.3123931 , 101.95475061, 104.11138622,
                      111.95590131, 109.17531467, 108.97138052, 106.89249739],            dtype=float64),)
        >>> vjp_fun(w)[:1]
        (Array([106.76358917, 113.3123931 , 101.95475061, 104.11138622,
                      111.95590131, 109.17531467, 108.97138052, 106.89249739],            dtype=float64),)
    """

    if not isinstance(primals, (tuple, list)):
        raise TypeError(
            "primal arguments to vjp_chunked must be a tuple or list; "
            f"found {type(primals).__name__}."
        )

    if isinstance(chunk_argnums, int):
        chunk_argnums = (chunk_argnums,)

    if not all(map(lambda x: (0 <= x) and (x < len(primals)), chunk_argnums)):
        raise ValueError(
            "chunk_argnums must index primals. Got chunk_argnums={} but len(primals)={}".format(
                chunk_argnums, len(primals)
            )
        )
        # TODO also check they are unique?

    if isinstance(nondiff_argnums, int):
        nondiff_argnums = (nondiff_argnums,)

    if chunk_argnums == ():
        chunk_size = None

    ############################################################################
    # sharding

    if config.netket_experimental_sharding and chunk_size is not None:
        if return_forward:
            raise NotImplementedError

        # assume the chunk_argnums are also sharded
        # later we might introduce an extra arg for it
        sharded_argnums = chunk_argnums
        sharded_args = tuple(i in sharded_argnums for i in range(len(primals)))

        # for the output we need to consult nondiff_argnums, which are removed
        non_sharded_argnums = tuple(
            set(range(len(primals))).difference(sharded_argnums)
        )
        out_args = _gen_append_cond_vjp(primals, nondiff_argnums, non_sharded_argnums)
        red_ops = tuple(jax.lax.psum if c else False for c in out_args)

        # check the chunk_size is not larger than the shard per device
        chunk_size = sharding_decorator(
            partial(check_chunk_size, chunk_argnums, chunk_size),
            sharded_args_tree=sharded_args,
            reduction_op_tree=True,
        )(*primals)

        if chunk_size is not None:
            _vjpc = _vjp_chunked(
                fun,
                has_aux=has_aux,
                chunk_argnums=chunk_argnums,
                chunk_size=chunk_size,
                nondiff_argnums=nondiff_argnums,
                return_forward=return_forward,
                conjugate=conjugate,
            )
            vjp_fun_sh = Partial(
                sharding_decorator(
                    _vjpc,
                    sharded_args_tree=(sharded_args, True),
                    reduction_op_tree=red_ops,
                ),
                primals,
            )
            return vjp_fun_sh
        else:
            pass  # no chunking, continue below

    ############################################################################
    # no sharding (or sharded, but not chunking)

    # check the chunk_size is not larger than the arrays
    chunk_size = check_chunk_size(chunk_argnums, chunk_size, *primals)

    if chunk_size is None:
        y, vjp_fun = nkvjp(fun, *primals, conjugate=conjugate, has_aux=has_aux)
        if return_forward:

            def __vjp_fun(y, vjp_fun, cotangents):
                res = vjp_fun(cotangents)
                res = _trash_tuple_elements(res, nondiff_argnums)
                return y, res

            return Partial(__vjp_fun, y, vjp_fun)
        else:

            def __vjp_fun(vjp_fun, cotangents):
                res = vjp_fun(cotangents)
                res = _trash_tuple_elements(res, nondiff_argnums)
                return res

            return Partial(__vjp_fun, vjp_fun)
    else:
        _vjpc = _vjp_chunked(
            fun,
            has_aux=has_aux,
            chunk_argnums=chunk_argnums,
            chunk_size=chunk_size,
            nondiff_argnums=nondiff_argnums,
            return_forward=return_forward,
            conjugate=conjugate,
        )
        return Partial(_vjpc, primals)
