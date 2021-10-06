import jax

from jax.tree_util import Partial

from functools import partial

from netket.jax import batch, unbatch, compose, scanmap, scan_append_accum, vjp as nkvjp


from ._scanmap import _multimap


_tree_batch = lambda x, *args, **kwargs: jax.tree_map(
    lambda l: batch(l, *args, **kwargs), x
)
_tree_unbatch = partial(jax.tree_map, unbatch)


def _trash_tuple_elements(t, nums=()):
    assert isinstance(t, tuple)
    if isinstance(nums, int):
        nums = (nums,)
    return tuple(r for i, r in enumerate(t) if i not in nums)


def _vjp(fun, cotangents, *primals, nondiff_argnums=(), conjugate=False):
    y, vjp_fun = nkvjp(fun, *primals, conjugate=conjugate)
    res = vjp_fun(cotangents)
    # trash non-needed tuple elements  of the output here
    # since xla is probably not able to optimize it away through the scan/loop if its trashed at the end
    # TODO pass closure to vjp instead ?
    res = _trash_tuple_elements(res, nondiff_argnums)
    return (y,) + res


def __vjp_fun_batched(
    fun,
    primals,
    cotangents,
    batch_argnums,
    nondiff_argnums,
    batch_size,
    conjugate,
    _vjp,
    _append_cond_fun,
):

    append_cond = _append_cond_fun(primals, nondiff_argnums, batch_argnums)
    scan_fun = partial(scan_append_accum, append_cond=append_cond)
    primals = tuple(
        _tree_batch(p, batch_size) if i in batch_argnums else p
        for i, p in enumerate(primals)
    )
    cotangents = _tree_batch(cotangents, batch_size)
    # cotangents, and whatever requested in primals; +2 since 0 is the function, and 1 is cotangents
    argnums = (1,) + tuple(map(lambda x: x + 2, batch_argnums))
    res = scanmap(
        partial(_vjp, nondiff_argnums=nondiff_argnums, conjugate=conjugate),
        scan_fun=scan_fun,
        argnums=argnums,
    )(fun, cotangents, *primals)

    return _multimap(lambda c, l: _tree_unbatch(l) if c else l, append_cond, res)


def _gen_append_cond_vjp(primals, nondiff_argnums, batch_argnums):
    diff_argnums = filter(lambda i: i not in nondiff_argnums, range(len(primals)))
    return tuple(map(lambda i: i in batch_argnums, diff_argnums))


_gen_append_cond_value_vjp = compose(lambda t: (True,) + t, _gen_append_cond_vjp)

_vjp_fun_batched = partial(
    __vjp_fun_batched,
    _vjp=compose(lambda yr: yr[1:], _vjp),
    _append_cond_fun=_gen_append_cond_vjp,
)
_value_and_vjp_fun_batched = compose(
    lambda yr: (yr[0], yr[1:]),
    partial(__vjp_fun_batched, _vjp=_vjp, _append_cond_fun=_gen_append_cond_value_vjp),
)


def vjp_batched(
    fun,
    *primals,
    has_aux=False,
    batch_argnums=(),
    batch_size=0,
    nondiff_argnums=(),
    return_forward=False,
    conjugate=False,
):
    """calculate the vjp in small batches for a function where the leading dimension of the output only depends on the leading dimension of some of the arguments

    Args:
        fun: Function to be differentiated. It must accept batches of size batch_size of the primals in batch_argnums.
        primals:  A sequence of primal values at which the Jacobian of ``fun`` should be evaluated.
        has_aux: Optional, bool. Only False is implemented. Indicates whether ``fun`` returns a pair where the
           first element is considered the output of the mathematical function to be
           differentiated and the second element is auxiliary data. Default False.
        batch_argnums: an integer or tuple of integers indicating the primals which should be batched.
            The leading dimension of each of the primals indicated must be the same as the output of fun.
        batch_size: an integer indicating the size of the batches over which the vjp is computed.
            It must be a integer divisor of the primals specified in batch_argnums.
        nondiff_argnums: an integer or tuple of integers indicating the primals which should not be differentiated with.
            Specifying the arguments which are not needed should increase performance.
        return_forward: whether the returned function should also return the output of the forward pass
    Returns:
        a function corresponding to the vjp_fun returned by an equivalent ``jax.vjp(fun, *primals)[1]``` call
        which computes the vjp in batches (recomputing the forward pass every time on subsequent calls).
        If return_forward=True the vjp_fun returned returns a tuple containg the ouput of the forward pass and the vjp.

    Example:
        In [1]: import jax
           ...: from netket.jax import vjp_batched
           ...: from functools import partial

        In [2]: @partial(jax.vmap, in_axes=(None, 0))
           ...: def f(p, x):
           ...:     return jax.lax.log(p.dot(jax.lax.sin(x)))
           ...:

        In [3]: k = jax.random.split(jax.random.PRNGKey(123), 4)
           ...: p = jax.random.uniform(k[0], shape=(8,))
           ...: v = jax.random.uniform(k[1], shape=(8,))
           ...: X = jax.random.uniform(k[2], shape=(1024,8))
           ...: w = jax.random.uniform(k[3], shape=(1024,))

        In [4]: vjp_fun_batched = vjp_batched(f, p, X, batch_argnums=(1,), batch_size=32, nondiff_argnums=1)
           ...: vjp_fun = jax.vjp(f, p, X)[1]

        In [5]: vjp_fun_batched(w)
        Out[5]:
        (DeviceArray([106.76358917, 113.3123931 , 101.95475061, 104.11138622,
                      111.95590131, 109.17531467, 108.97138052, 106.89249739],            dtype=float64),)

        In [6]: vjp_fun(w)[:1]
           ...:
        Out[6]:
        (DeviceArray([106.76358917, 113.3123931 , 101.95475061, 104.11138622,
                      111.95590131, 109.17531467, 108.97138052, 106.89249739],            dtype=float64),)
    """

    if not isinstance(primals, (tuple, list)):
        raise TypeError(
            "primal arguments to vjp_batched must be a tuple or list; "
            f"found {type(primals).__name__}."
        )

    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)

    if not all(map(lambda x: (0 <= x) and (x < len(primals)), batch_argnums)):
        raise ValueError(
            "batch_argnums must index primals. Got batch_argnums={} but len(primals)={}".format(
                batch_argnums, len(primals)
            )
        )
        # TODO also check they are unique?

    if isinstance(nondiff_argnums, int):
        nondiff_argnums = (nondiff_argnums,)

    if batch_size == 0 or batch_argnums == ():

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
    if has_aux:
        raise NotImplementedError
        # fun = compose(lambda x_aux: x_aux[0], fun)
        # TODO in principle we could also return the aux of the fwd pass for every batch...

    _vjp_fun = _value_and_vjp_fun_batched if return_forward else _vjp_fun_batched

    return Partial(
        partial(
            _vjp_fun,
            fun,
            batch_argnums=batch_argnums,
            nondiff_argnums=nondiff_argnums,
            batch_size=batch_size,
            conjugate=conjugate,
        ),
        primals,
    )
