import jax

from jax.tree_util import Partial

from functools import partial

from netket.jax import (
    batch,
    unbatch,
    compose,
    scanmap,
    scan_append_accum,
)


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


def _vjp(fun, cotangents, *primals, nondiff_argnums=()):
    y, vjp_fun = jax.vjp(fun, *primals)
    res = vjp_fun(cotangents)
    # trash non-needed tuple elements  of the output here
    # since xla is probably not able to optimize it away through the scan/loop if its trashed at the end
    # TODO pass closure to vjp instead ?
    res = _trash_tuple_elements(res, nondiff_argnums)
    return y, *res


def __vjp_fun_batched(
    fun,
    primals,
    cotangents,
    batch_argnums,
    nondiff_argnums,
    batch_size,
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
        partial(_vjp, nondiff_argnums=nondiff_argnums),
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
    primals,
    has_aux=False,
    batch_argnums=(),
    batch_size=0,
    nondiff_argnums=(),
    return_forward=False,
):

    if not isinstance(primals, (tuple, list)):
        raise TypeError(
            "primal arguments to vjp_batched must be a tuple or list; "
            f"found {type(primals).__name__}."
        )

    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)

    if isinstance(nondiff_argnums, int):
        nondiff_argnums = (nondiff_argnums,)

    if batch_size == 0 or batch_argnums == ():
        raise NotImplementedError  # TODO call normal jax.vjp ?

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
        ),
        primals,
    )
