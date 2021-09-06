from netket.jax import compose


def unbatch(x):
    return x.reshape((-1,) + x.shape[2:])


def batch(x, batchsize=None):
    # batchsize=None -> add just a dummy batch dimension, same as np.expand_dims(x, 0)
    n = x.shape[0]
    if batchsize is None:
        batchsize = n
    else:
        assert (n % batchsize) == 0
    return x.reshape((n // batchsize, batchsize) + x.shape[1:])


def rebatch(x, batchsize):
    return batch(unbatch(x), batchsize)


def unbatch_output(f):
    return compose(unbatch, f)
    # return wraps(f)(compose(unbatch, f))


def unbatch_args(f, argnums):
    if isinstance(argnums, int):
        argnums = (argnums,)

    def _f(*args):
        args = (unbatch(a) if i in argnums else a for i, a in enumerate(args))
        return f(*args)

    return _f


def batch_args(f, argnums, src=None):
    if isinstance(argnums, int):
        argnums = (argnums,)

    def _f(*args):
        batchsize = None if src is None else args[src].shape[1]
        args = (batch(a, batchsize) if i in argnums else a for i, a in enumerate(args))
        return f(*args)

    return _f
