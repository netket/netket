import math
from functools import partial, wraps

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from netket.utils import config
from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError


def replicate_sharding(f):
    """
    Wrapper for python get_conn_padded to make it work with shared/global device arrays.
    Calls f on every shard, and puts the results back on the devices with the correct sharding.
    The input to f is assumed to have PositionalSharding (or equivalent) along a single batch axis.

    The resulting function cannot be used inside of jit.

    Args:
        f: a python get_conn_padded (which takes self, x and maps it to (xp,mels))
    """
    # ideally I would like to use a simple shard map with callback, however
    # for that we need to know the shape a priori which would require an extra n_conn call.

    if config.netket_experimental_sharding:

        @wraps(f)
        def _f(self, x):
            concrete_or_error(None, x, NumbaOperatorGetConnDuringTracingError, f)

            if isinstance(x, jax.Array) and len(x.devices()) > 1:  # sharded
                if isinstance(x.sharding, jax.sharding.GSPMDSharding):
                    # convert to positional sharding, to simplify reshape below
                    s = x.sharding
                    shard_shape = tuple(
                        (
                            2
                            * (
                                np.array(s.shard_shape(x.shape)) == np.array(x.shape)
                            ).astype(int)
                            - 1
                        ).tolist()
                    )
                    # TODO would like to take x.devices, but list(x.devices()) has reversed order
                    s_new = jax.sharding.PositionalSharding(jax.devices()).reshape(
                        shard_shape
                    )
                    assert s.is_equivalent_to(s_new, x.ndim)
                    x = jax.jit(_identity, out_shardings=s_new)(x)

                xp_mels_np = []
                n_conn_dev = []
                for s in x.addressable_shards:
                    xp, mels = f(self, s.data)
                    xp_mels_np.append((xp, mels))
                    n_conn_dev.append(
                        jax.device_put(
                            np.array(
                                [
                                    mels.shape[-1],
                                ]
                            ),
                            s.device,
                        )
                    )
                # numba might pad every x differently, so here we pad all to the common max over devices and all processes
                n_conn = jax.make_array_from_single_device_arrays(
                    (len(x.devices()),),
                    jax.sharding.PositionalSharding(list(x.devices())),
                    n_conn_dev,
                )
                n_conn_max = int(jax.jit(lambda x: x.max())(n_conn))
                xp_dev = []
                mels_dev = []
                for (xp, mels), s in zip(xp_mels_np, x.addressable_shards):
                    npad = n_conn_max - mels.shape[-1]
                    if npad > 0:
                        mels = np.pad(
                            mels, pad_width=((0, 0),) * (mels.ndim - 1) + ((0, npad),)
                        )
                        xp = np.pad(
                            xp,
                            pad_width=((0, 0),) * (mels.ndim - 1)
                            + ((0, npad),)
                            + ((0, 0),),
                        )
                        xp[..., -npad:, :] = xp[..., :1, :]
                    xp_dev.append(jax.device_put(xp, s.device))
                    mels_dev.append(jax.device_put(mels, s.device))
                shape = x.shape[:-1] + (n_conn_max,)
                xp = jax.make_array_from_single_device_arrays(
                    shape + x.shape[-1:],
                    x.sharding.reshape(
                        x.sharding.shape[:-1] + (1,) + x.sharding.shape[-1:]
                    ),
                    xp_dev,
                )
                mels = jax.make_array_from_single_device_arrays(
                    shape, x.sharding, mels_dev
                )
                return xp, mels
            elif isinstance(x, jax.Array):  # and len(x.devices()) == 1; single device
                return jax.device_put(f(self, x), device=x.device())
            else:
                return f(self, x)

        return _f
    else:
        return f


_identity = lambda x: x


def _prepare_mask(n, n_pad):
    return jnp.ones(n + n_pad, dtype=bool).at[-n_pad:].set(0)


def put_global(inp_data, axis=0, pad=False, pad_value=None):
    """
    distribute a local array equally along an axis to all (local and global) devices
    The size of the axis needs to be divisible by the number of devices.
    each process needs to have the whole array (parts not belonging to it can be filled with garbage)

    Args:
        inp_data: the full array (on every process)
        axis: (optional) axis alogn which to distribute

    Returns:
        a distributed jax.Array
    """
    if pad:
        n = inp_data.shape[0]
        # pad to the next multiple of device_count
        device_count = jax.device_count()
        n_pad = math.ceil(inp_data.shape[0] / device_count) * device_count - n
        inp_data = jnp.pad(inp_data, ((0, n_pad), (0, 0)))
        if pad_value is not None and n_pad > 0:
            inp_data = inp_data.at[-n_pad:].set(pad_value)

    shape = [
        1,
    ] * inp_data.ndim
    shape[axis] = -1
    sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(shape)
    out_data = jax.jit(_identity, out_shardings=sharding)(inp_data)
    if pad:
        if n_pad > 0:
            mask = jax.jit(
                _prepare_mask, out_shardings=sharding.reshape(-1), static_argnums=(0, 1)
            )(n, n_pad)
        else:
            mask = None
        return out_data, mask
    else:
        return out_data


def extract_replicated(t):
    """
    Extract the value of a fully replicated global device array.

    Args:
        t: a jax Array (or a pytree of jax Arrays)

    Returns:
        A locally adressable representation of t
    """

    def _extract_replicated(x):
        if isinstance(x, jax.Array) and not x.is_fully_addressable:
            assert x.is_fully_replicated
            return x.addressable_data(0)
        else:
            return x

    return jax.tree_map(_extract_replicated, t)


def gather(x):
    """
    make a sharded array fully replicated by gathering all parts on every device
    """
    return jax.jit(_identity, out_shardings=x.sharding.replicate())(x)


def broadcast(x):
    """
    broadcast an array to all devices. Input on different processes is assumed to be the same
    """
    return jax.jit(
        _identity,
        out_shardings=jax.sharding.PositionalSharding(jax.devices())
        .replicate()
        .reshape((1,) * x.ndim),
    )(x)


def sharding_decorator(f, sharded_args_tree, reduction_op_tree=False):
    """
    A decorator which wraps a function so that it is evaluated on every shard of the distributed arguments,
    and the output is either returned sharded, or reduced with a collective operation.

    Does nothing unless config.netket_experimental_sharding=True.
    Intended for netket internal use only, interface might change in the future depending on requirements.

    Args:
        f: a function
        sharded_args_tree: a tuple/pyrtree of True/False indicating that the input is:
            True: sharded on axis 0 (True)
            False: assumed to be replicated
            the args of f are flattened according to sharded_args_tree, so if an arg is a pytree it is assumed the whole tree
        reduction_op_tree: a tuple/pyrtree of reduction_op, where for each output:
            reduction_op is e.g. jax.lax.psum if it is to be reduced, then f_wrapped returns a replicated array
            reduction op is False if it is not to be reduced, then f_wrapped returns a sharded array
            reduction op is True if it is not an array/pytree, then it is returned as python object

    Returns :
        f_wrapped: wrapped version of f
    """

    if config.netket_experimental_sharding:
        sharded_args, args_treedef = jax.tree_util.tree_flatten(sharded_args_tree)
        reduction_op, out_treedef = jax.tree_util.tree_flatten(reduction_op_tree)

        @wraps(f)
        def _fun(*args):
            args = args_treedef.flatten_up_to(args)

            _sele = lambda cond, xs: tuple(x for c, x in zip(cond, xs) if c)
            _not = lambda t: tuple(not x for x in t)
            _sele2 = lambda cond, x, y: tuple(x if c else y for c in cond)

            # workaround for shard_map not supporting non-array args part 1/2
            nonarray_args = tuple(not hasattr(a, "dtype") for a in args)
            args = tuple(
                Partial(partial(lambda x: x, a)) if c else a
                for a, c in zip(args, nonarray_args)
            )

            mesh = Mesh(jax.devices(), axis_names=("i"))
            in_specs = _sele2(sharded_args, P("i"), P())
            out_specs = out_treedef.unflatten(_sele2(reduction_op, P(), P("i")))

            @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
            def _f(*args):
                # workaround for shard_map not supporting non-array args part 2/2
                args = tuple(a() if c else a for a, c in zip(args, nonarray_args))

                res = f(*args_treedef.unflatten(args))

                # apply reductions
                # _id = lambda x: x
                # _wrap = lambda x: Partial(lambda : x)
                def _sele_op(o):
                    if o is False:
                        return lambda x: x
                    if o is True:
                        return lambda x: Partial(partial(lambda x: x, x))
                    else:
                        return partial(jax.tree_map, partial(o, axis_name="i"))

                reductions = [_sele_op(o) for o in reduction_op]
                res = out_treedef.flatten_up_to(res)
                res = [f(r) for f, r in zip(reductions, res)]
                res = out_treedef.unflatten(res)
                return res

            res = _f(*args)
            res = out_treedef.flatten_up_to(res)
            res = [a() if c is True else a for a, c in zip(res, reduction_op)]
            res = out_treedef.unflatten(res)
            return res

        return _fun

    return f
