import jax
from functools import partial
from . import mpi_sum_jax


def _mpi_args_sum(*args):
    token = jax.lax.create_token()
    res = []
    for x in args:
        y, token = mpi_sum_jax(x, token=token)
        res.append(y)
    return tuple(res)


def _mpi_args_sum_impl(*args, transpose):
    if transpose:
        return args
    else:
        return _mpi_args_sum(*args)


def _mpi_args_sum_transpose_rule(tan_args, *x_args, transpose):
    return _mpi_args_sum_impl(*tan_args, transpose=not transpose)


def _mpi_args_sum_abstract_eval(*args, transpose):
    return map(lambda x: jax.abstract_arrays.ShapedArray(x.shape, x.dtype), args)


def _batch_rule(prim, batched_args, batch_dims, **params):
    return prim.bind(*batched_args, **params), batch_dims


mpi_args_sum_p = jax.core.Primitive("mpi_args_sum")
mpi_args_sum_p.multiple_results = True
mpi_args_sum_p.def_impl(_mpi_args_sum_impl)
mpi_args_sum_p.def_abstract_eval(_mpi_args_sum_abstract_eval)
jax.interpreters.ad.primitive_transposes[mpi_args_sum_p] = _mpi_args_sum_transpose_rule
jax.interpreters.xla.register_initial_style_primitive(mpi_args_sum_p)
jax.interpreters.mlir.register_lowering(
    mpi_args_sum_p,
    jax.interpreters.mlir.lower_fun(_mpi_args_sum_impl, multiple_results=True),
)
jax.interpreters.batching.primitive_batchers[mpi_args_sum_p] = partial(
    _batch_rule, mpi_args_sum_p
)


def mpi_tree_sum(x):
    leaves, treedef = jax.tree_util.tree_flatten(x)
    out = mpi_args_sum_p.bind(*leaves, transpose=False)
    return treedef.unflatten(out)
