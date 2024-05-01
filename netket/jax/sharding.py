# Copyright 2023 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Internal utility functions to support jax sharding natively within netket.
All functions in here are not part of the public API, internal, and may change without warning.
"""

import math
from functools import partial, wraps
import warnings

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax.sharding import (
    Mesh,
    PartitionSpec as P,
    PositionalSharding,
)
from jax.experimental.shard_map import shard_map
from jax.util import safe_zip

from netket.utils import config, mpi
from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError


def _convert_gspmdsharding_to_positionalsharding(x):
    # try to convert gspmdsharding to positional sharding
    # necessary because gspmdsharding has no .replicate()
    # TODO
    s = x.sharding
    shard_shape = tuple(
        (
            2 * (np.array(s.shard_shape(x.shape)) == np.array(x.shape)).astype(int) - 1
        ).tolist()
    )
    # TODO would like to take x.devices, but list(x.devices()) has reversed order
    s_new = PositionalSharding(jax.devices()).reshape(shard_shape)
    assert s.is_equivalent_to(s_new, x.ndim)
    return jax.jit(_identity, out_shardings=s_new)(x)
    # TODO support gspmdsharding in numba wrapper and use this
    # return jax.jit(jax.lax.with_sharding_constraint, static_argnums=1)(x, s_new)


def replicate_sharding_decorator_for_get_conn_padded(f):
    """
    Wrapper for python get_conn_padded to make it work with shared/global device arrays.

    Calls f on every shard, and puts the results back on the devices with the correct sharding.
    The input to f is assumed to have PositionalSharding (or equivalent) along a single batch axis.

     .. note::
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
                if config.netket_experimental_sharding_numba_wrapper_warning:
                    warnings.warn(
                        "You are using the experimental wrapper for numba operators acting on a sharded input array. "
                        "Please consider rewriting your operator in jax."
                        "Some of the built-in netket operators can be converted into jax by calling .to_jax_operator()"
                        "If you have to use this wrapper and find that it does not work properly, "
                        "please open an issue at https://github.com/netket/netket/issues."
                        "To silence this warning, set the environment variable `NETKET_EXPERIMENTAL_SHARDING_NUMBA_WRAPPER_WARNING=0`",
                        stacklevel=2,
                    )

                if isinstance(x.sharding, jax.sharding.GSPMDSharding):
                    x = _convert_gspmdsharding_to_positionalsharding(x)

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
                    PositionalSharding(list(x.devices())),
                    n_conn_dev,
                )
                n_conn_max = int(jax.jit(lambda x: x.max())(n_conn))
                xp_dev = []
                mels_dev = []
                for (xp, mels), s in safe_zip(xp_mels_np, x.addressable_shards):
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


def distribute_to_devices_along_axis(
    inp_data, axis=0, pad=False, pad_value=None, devices=jax.devices()
):
    """
    Distribute a local array equally along an axis to multiple jax devices devices

     .. note:
        Does nothing if netket.config.netket_experimental_sharding=False.

     .. note:
        Each jax process needs to have the whole array (parts not belonging to it can be filled with garbage).

    Args:
        inp_data: the full array (on every process)
        axis: (optional) axis along which to distribute
        pad: If True: pad the input data along axis to the next multiple of the number of devices
              If False (default): no padding; the size of the axis in inp_data needs to be divisible by the number of devices.
        pad_value: value to pad with (optional, only used if pad=True)
        devices: (optional) list of jax devices. Defaults to all available devices

    Returns:
        out_data: a distributed jax.Array
        mask: a mask indicating wether a given element is part of the original data (True) of of the padding (False)
              only returned if pad=True
    """
    if config.netket_experimental_sharding:
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
        sharding = PositionalSharding(devices).reshape(shape)
        out_data = jax.jit(_identity, out_shardings=sharding)(inp_data)
        # TODO support gspmdsharding in numba wrapper and use this
        # out_data = jax.jit(jax.lax.with_sharding_constraint, static_argnums=1)(
        #     inp_data, sharding
        # )

        if pad:
            if n_pad > 0:
                mask = jax.jit(
                    _prepare_mask,
                    out_shardings=sharding.reshape(-1),
                    static_argnums=(0, 1),
                )(n, n_pad)
            else:
                mask = None
            return out_data, mask
        else:
            return out_data
    else:
        return inp_data


# TODO consider merging this with distribute_to_devices_along_axis
@jax.jit
def with_samples_sharding_constraint(x, shape=None):
    """
    ensure the input x is sharded along axis 0 on all devices
    works both outside and inside of jit
    """
    if config.netket_experimental_sharding and jax.device_count() > 1:
        x = jax.lax.with_sharding_constraint(
            x, PositionalSharding(jax.devices()).reshape((-1,) + (1,) * (x.ndim - 1))
        )
    return x


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
            if not x.is_fully_replicated:
                raise RuntimeError(
                    "Expected a fully replicated array, but found one that is not. You should gather the array first. If you are not developing custom logic, please open a bug report with a reproducer."
                )
            return x.addressable_data(0)
        else:
            return x

    return jax.tree_util.tree_map(_extract_replicated, t)


def gather(x):
    """
    Make a sharded array fully replicated by gathering all parts on every device.

    Args:
        x: potentially unreplicated jax.Array with PositionalSharding

    Returns:
        fully replicated array
    """

    if not isinstance(x, jax.Array):
        # TODO in the future we could chagne it to just return x unchanged
        # but for now we error if x is not a jax array to ensure gather is used correctly
        raise RuntimeError("gather can only be applied to a jax.Array")
    elif x.is_fully_replicated:  # includes SingleDeviceSharding
        return x
    elif isinstance(x.sharding, jax.sharding.GSPMDSharding):
        # x.sharding.device_set has arbitrary order
        # Hardcode all devices until I figure out a way to deduce the order from x
        out_shardings = PositionalSharding(jax.devices()).replicate()
        # out_shardings = x.sharding.get_replicated(jax.devices())
    elif isinstance(x.sharding, PositionalSharding):
        out_shardings = x.sharding.replicate()
    else:
        raise NotImplementedError(
            f"Gather is only compatible with PositionalSharding, but array has {x.sharding} Please open a feature request."
        )
    return jax.jit(_identity, out_shardings=out_shardings)(x)
    # TODO support gspmdsharding in numba wrapper and use this
    # return jax.jit(jax.lax.with_sharding_constraint, static_argnums=1)(x, out_shardings)


def sharding_decorator(f, sharded_args_tree, reduction_op_tree=False, **kwargs):
    """
    A decorator which wraps a function so that it is evaluated on every shard of the distributed arguments,
    and the output is either returned sharded, or can be reduced with a collective operation.

    This is essentially a fancy wrapper around jax.experimental.shard_map,
    meant to be used to wrap the `chunked` parts of netket (vmap_chunked, vjp_chunked, ...), so that the
    computations are computed in chunks on every devices shard (and not in chunks of the whole array).

    .. warning::
        Intended for netket internal use only, the interface might change in the future based on our requirements.

    .. note:
        if `netket.config.netket_experimental_sharding=False` it returns the unchanged original function

    Args:
        f: a function
        sharded_args_tree: a tuple / tuple of pyrtrees of length of the number of args in f
            containing True/False indicating that each input in the argumens of f is:
                True: sharded on axis 0 (True)
                False: assumed to be replicated
                'key': A single jax.random.key. It is split across devices so that the function executed on every device is passed a different key.
                       If the argument is already a sharded array of keys use True instead.
            the args of f are flattened according to sharded_args_tree, so if an arg is a pytree a single True/False is assumed the whole tree
        reduction_op_tree: a tuple/pyrtree of reduction_op, where for each output:
            reduction_op is e.g. jax.lax.psum if it is to be reduced, then f_wrapped returns a replicated array
            reduction op is False if it is not to be reduced, then f_wrapped returns a sharded array
            reduction op is True if it is not an array/pytree, then it is returned as python object

    Returns :
        f_wrapped: wrapped version of f


    Example:


        %env NETKET_EXPERIMENTAL_SHARDING=1

        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec as P
        from jax.experimental.shard_map import shard_map
        from jax.tree_util import Partial
        from functools import partial
        from netket import config
        from netket.jax.sharding import sharding_decorator

        assert config.netket_experimental_sharding is True
        assert jax.device_count() > 1

        def expensive_elementwise_function(x, c):
            return x + c

        def looped_computation(x, c=1, f=expensive_elementwise_function):
            y = jnp.zeros_like(x)
            for i in range(len(x)):
                y = y.at[i].set(f(x[i], c))
            return y

        x = jax.jit(jnp.ones, out_shardings=jax.sharding.PositionalSharding(jax.devices()), static_argnums=0)(jax.device_count()*5)
        c = jax.jit(jnp.ones, out_shardings=jax.sharding.PositionalSharding(jax.devices()).replicate(), static_argnums=0)(())

        # if we were to run `looped_computation(x)`` with the sharded x, it would formally be computed sequentially for all elements device per device,
        # if we  jit, i.e. `jax.jit(looped_computation)(x)`` the output sharding would just be replicated, jax just computes everything replicated on every device.
        # However we want to compute sequentially on every device in parallel

        # one way to do this is by reshaping with the number of devices, and moving the axes (and in general might require changing the function)
        x_per_device = x.reshape(len(x.devices()), -1)
        x_per_device = jnp.swapaxes(x_per_device, 0,1) # make the devices the second axis
        y_per_device = jax.jit(jax.vmap(looped_computation, in_axes=1, out_axes=1))(x_per_device)
        y_per_device = jnp.swapaxes(y_per_device, 0,1)  # output swap back
        y_flat = y_per_device.ravel()
        jax.debug.visualize_array_sharding(y_flat)

        # we can achieve the same in 1 line with sharding_decorator:
        y = jax.jit(sharding_decorator(looped_computation, sharded_args_tree=(True,)))(x)
        jax.debug.visualize_array_sharding(y)

        # it also supports a mixture of arrays with PositionalSharding along an axis and replicated sharding:
        y = jax.jit(sharding_decorator(looped_computation, sharded_args_tree=(True, False)))(x, c)
        jax.debug.visualize_array_sharding(y)

        #
        # furthermore sharding decorator supports reduction operations on the output:
        #
        def looped_computation2(x):
            y = looped_computation(x)
            return y.sum(axis=0)

        # again the manual version:
        y2_per_device = jax.jit(jax.vmap(looped_computation2, in_axes=1, out_axes=0))(x_per_device)
        # take sum by hand over devices
        y2_flat = y2_per_device.sum(axis=0)

        # and with sharding_decorator:
        y2 = jax.jit(sharding_decorator(looped_computation2, sharded_args_tree=(True,), reduction_op_tree=jax.lax.psum))(x)

        # which internally wraps the function like this:
        mesh = Mesh(jax.devices(), axis_names=("i"))
        in_specs = P("i")
        out_specs = P()
        @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
        def _f(x):
            res = looped_computation2(x)
            res = jax.lax.psum(res, axis_name="i")
            return res

        #
        # Furthermore it supports non-array outputs with the special choice of reduction_op True:
        #
        def looped_computation3(x):
            y = looped_computation(x)
            some_python_object = {1,2,3}
            return y, some_python_object
        # here we cannot jit, as the static some_python_object cannot be returned from a jitted function
        # this is meant for calling inside of another jitted function, where e.g. some computation depends on the object
        # but is not retuned at the end (or is with the same trick we use here)
        y3, my_python_object = sharding_decorator(looped_computation3, sharded_args_tree=(True,), reduction_op_tree=(False, True))(x)

        # Internally it does something like this:

        mesh = Mesh(jax.devices(), axis_names=("i"))
        in_specs = P("i")
        out_specs = P('i'), P()
        @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
        def _f(x):
            some_python_object = {1,2,3}
            return x, Partial(partial(lambda x: x, some_python_object))
        y, obj_wrapped = _f(x)
        obj = obj_wrapped()

        # Finally we note that for non-array inputs, which are not yet supported by the jax experimental shard_map we
        # (automatically) wrap them in the metadata of a Partial if they don't have a dtype attribute

        # In the following, for the sake of better understanding we describe how we would have to do this manually

        # If it is a function we could just wrap it in a Partial directly:
        # NB: for the Partial ith no .args it is irrelevant if we use True or False in sharded_args_tree, jax just sees it as an empty array

        y = sharding_decorator(looped_computation, sharded_args_tree=(True, False, False))(x, c, Partial(expensive_elementwise_function))

        # In general we hide it in a partial inside of a partial which we have to call inside of the function
        # the same trick as for reduction op True above

        # here we use Partial(partial(lambda x: x, obj)), iirc just a Partial(lambda x: obj)) didn't work in some cases

        def looped_computation4(x, obj_wrapped):
            obj = obj_wrapped()
            if obj:
                y = looped_computation(x)
            else:
                y = x
            return y
        obj = True
        obj_wrapped = Partial(partial(lambda x: x, obj))
        y4 = jax.jit(sharding_decorator(looped_computation4, sharded_args_tree=(True, False)))(x, obj_wrapped)
    """

    if config.netket_experimental_sharding:
        if not isinstance(sharded_args_tree, tuple):
            sharded_args_tree = (sharded_args_tree,)
        sharded_args, args_treedef = jax.tree_util.tree_flatten(sharded_args_tree)
        reduction_op, out_treedef = jax.tree_util.tree_flatten(reduction_op_tree)

        @wraps(f)
        def _fun(*args):
            args = args_treedef.flatten_up_to(args)

            _sele = lambda cond, xs: tuple(x for c, x in safe_zip(cond, xs) if c)
            _not = lambda t: tuple(not x for x in t)
            _sele2 = lambda cond, x, y: tuple(x if c else y for c in cond)

            # PRNGKey treatment 1/2
            args = tuple(
                jax.random.split(a, jax.device_count()) if c == "key" else a
                for a, c in safe_zip(args, sharded_args)
            )

            # workaround for shard_map not supporting non-array args part 1/2
            nonarray_args = tuple(not hasattr(a, "dtype") for a in args)
            args = tuple(
                Partial(partial(lambda x: x, a)) if c else a
                for a, c in safe_zip(args, nonarray_args)
            )

            mesh = Mesh(jax.devices(), axis_names=("i"))
            in_specs = _sele2(sharded_args, P("i"), P())
            out_specs = out_treedef.unflatten(_sele2(reduction_op, P(), P("i")))

            @partial(
                shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kwargs
            )
            def _f(*args):
                # workaround for shard_map not supporting non-array args part 2/2
                args = tuple(a() if c else a for a, c in safe_zip(args, nonarray_args))

                # PRNGKey treatment 2/2
                args = tuple(
                    a[0] if c == "key" else a for a, c in safe_zip(args, sharded_args)
                )

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
                        return partial(
                            jax.tree_util.tree_map, partial(o, axis_name="i")
                        )

                reductions = [_sele_op(o) for o in reduction_op]
                res = out_treedef.flatten_up_to(res)
                res = [f(r) for f, r in safe_zip(reductions, res)]
                res = out_treedef.unflatten(res)
                return res

            res = _f(*args)
            res = out_treedef.flatten_up_to(res)
            res = [a() if c is True else a for a, c in safe_zip(res, reduction_op)]
            res = out_treedef.unflatten(res)
            return res

        return _fun

    return f


def device_count_per_rank():
    """
    Helper functions which returns the number of jax devices netket will use for every
    MPI rank.

    Returns:
        jax.device_count() if config.netket_experimental_sharding is True, and 1 otherwise
    """
    if config.netket_experimental_sharding:
        if mpi.n_nodes > 1:
            # this should never be triggered as we disable mpi when sharding
            raise NotImplementedError("hybrid mpi and sharding is not not supported")
        return jax.device_count()
    else:  # mpi or serial
        return 1


def device_count():
    """
    Helper functions which returns the TOTAL number of jax devices netket will use.

    Returns:
        jax.device_count() if config.netket_experimental_sharding is True, and mpi.rank otherwise.
    """
    return mpi.n_nodes * device_count_per_rank()
