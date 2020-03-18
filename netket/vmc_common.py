import sys

import netket as _nk


def info(obj, depth=None):
    if hasattr(obj, "info"):
        return obj.info(depth)
    else:
        return str(obj)


def make_optimizer_fn(arg, ma):
    """
    Utility function to create the optimizer step function for VMC drivers.

    It currently supports three kinds of inputs:

    1. A NetKet optimizer, i.e., a subclass of `netket.optimizer.Optimizer`.

    2. A 3-tuple (init, update, get) of optimizer functions as used by the JAX
       optimizer submodule (jax.experimental.optimizers).

       The update step p0 -> p1 with bare step dp is computed as
            x0 = init(p0)
            x1 = update(i, dp, x1)
            p1 = get(x1)

    3. A single function update with signature p1 = update(i, dp, p0) returning the
       updated parameter value.
    """
    if isinstance(arg, tuple) and len(arg) == 3:
        init, update, get = arg

        def optimize_fn(i, grad, p):
            x0 = init(p)
            x1 = update(i, grad, x0)
            return get(x1)

        desc = "JAX-like optimizer"
        return optimize_fn, desc

    elif issubclass(type(arg), _nk.optimizer.Optimizer):

        arg.init(ma.n_par, ma.is_holomorphic)

        def optimize_fn(_, grad, p):
            arg.update(grad, p)
            return p

        desc = info(arg)
        return optimize_fn, desc

    elif callable(arg):
        import inspect

        sig = inspect.signature(arg)
        if not len(sig.parameters) == 3:
            raise ValueError(
                "Expected netket.optimizer.Optimizer subclass, JAX optimizer, "
                + " or callable f(i, grad, p); got callable with signature {}".format(
                    sig
                )
            )
        desc = "{}{}".format(arg.__name__, sig)
        return arg, desc
    else:
        raise ValueError(
            "Expected netket.optimizer.Optimizer subclass, JAX optimizer, "
            + " or callable f(i, grad, p); got {}".format(arg)
        )


def tree_map(fun, tree):

    if tree is None:
        result = None

    elif "_C_netket" in str(type(tree)):
        result = fun(tree)

    elif type(tree) == list:
        result = []
        for val in tree:
            result.append(fun, tree_map(val))

    else:
        result = {}
        for key in tree:
            result[key] = tree_map(fun, tree[key])

    return result
