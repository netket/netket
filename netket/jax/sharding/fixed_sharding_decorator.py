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

from functools import partial, wraps
import contextlib


import jax
from jax.tree_util import Partial
from jax.sharding import PartitionSpec as P

from netket.utils import config

safe_zip = partial(zip, strict=True)


SHARD_MAP_STACK_LEVEL: int = 0
"""
A counter used to keep track of how many levels deep we are in shard_map.

This should not be modified directly, but controlled through the context manager
_increase_SHARD_MAP_STACK_LEVEL.
"""


@contextlib.contextmanager
def _increase_SHARD_MAP_STACK_LEVEL():
    """
    A context manager used by `sharding_decorator` to keep track of how many nested
    sharded function calls we are in.

    In pratice, this is used to ensure that only the outermost function is sharded, and following
    calls are not sharded.
    """
    global SHARD_MAP_STACK_LEVEL
    try:
        SHARD_MAP_STACK_LEVEL += 1
        yield
    except Exception as e:
        raise e
    finally:
        SHARD_MAP_STACK_LEVEL -= 1


def _get_SHARD_MAP_STACK_LEVEL():
    """
    Returns the current value of SHARD_MAP_STACK_LEVEL.
    This is used to check if we are inside a shard_map call.
    """
    global SHARD_MAP_STACK_LEVEL
    return SHARD_MAP_STACK_LEVEL


def sharding_decorator(
    f, sharded_args_tree, reduction_op_tree=False, pvary_args_tree=False, **kwargs
):
    """
    A decorator which wraps a function so that it is evaluated on every shard of the distributed arguments,
    and the output is either returned sharded, or can be reduced with a collective operation.

    This is essentially a fancy wrapper around jax.shard_map,
    meant to be used to wrap the `chunked` parts of netket (vmap_chunked, vjp_chunked, ...), so that the
    computations are computed in chunks on every devices shard (and not in chunks of the whole array).

    .. warning::
        Intended for netket internal use only, the interface might change in the future based on our requirements.

    .. note:
        if `netket.config.netket_experimental_sharding=False` it returns the unchanged original function

    .. note:
        If nested functions are decorated with this decorator, only the outermost one is sharded and the internal
        ones are ignored.

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
        pvary_args_tree: a tuple / tuple of pyrtrees of length of the number of args in f
            if True apply pvary to the argument, else do nothing
    Returns :
        f_wrapped: wrapped version of f


    Example:


        %env NETKET_EXPERIMENTAL_SHARDING=1

        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
        from jax.tree_util import Partial
        from functools import partial
        from netket import config
        from netket.jax.sharding import sharding_decorator

        mesh = Mesh(jax.devices(), 'S')

        assert config.netket_experimental_sharding is True
        assert jax.device_count() > 1

        def expensive_elementwise_function(x, c):
            return x + c

        def looped_computation(x, c=1, f=expensive_elementwise_function):
            y = jnp.zeros_like(x)
            for i in range(len(x)):
                y = y.at[i].set(f(x[i], c))
            return y

        x = jax.jit(jnp.ones, out_shardings=NamedSharding(mesh, P('S')), static_argnums=0)(jax.device_count()*5)
        c = jax.jit(jnp.ones, out_shardings=NamedSharding(mesh, P()).replicate(), static_argnums=0)(())

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

        # it also supports a mixture of arrays with NamedSharding along an axis and replicated sharding:
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
        mesh = Mesh(jax.devices(), axis_names=("S"))
        in_specs = P("S")
        out_specs = P()
        @partial(jax.shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
        def _f(x):
            res = looped_computation2(x)
            res = jax.lax.psum(res, axis_name="S")
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

        mesh = Mesh(jax.devices(), axis_names=("S"))
        in_specs = P("S")
        out_specs = P('S'), P()
        @partial(jax.shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
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
    if config.netket_experimental_sharding:  # type: ignore
        if not isinstance(sharded_args_tree, tuple):
            sharded_args_tree = (sharded_args_tree,)
        sharded_args, args_treedef = jax.tree_util.tree_flatten(sharded_args_tree)
        reduction_op, out_treedef = jax.tree_util.tree_flatten(reduction_op_tree)

        @wraps(f)
        def _fun(*args_orig):
            global SHARD_MAP_STACK_LEVEL
            # Jax does not support nested shard_map calls, so we bail out eaerly
            # if we are already inside of a shard map call
            if SHARD_MAP_STACK_LEVEL > 0:
                return f(*args_orig)

            args = args_treedef.flatten_up_to(args_orig)

            _sele = lambda cond, xs: tuple(x for c, x in safe_zip(cond, xs) if c)
            _not = lambda t: tuple(not x for x in t)
            _sele2 = lambda cond, x, y: tuple(x if c else y for c in cond)

            # PRNGKey treatment 1/2
            args = tuple(
                jax.random.split(a, jax.device_count()) if c == "key" else a
                for a, c in safe_zip(args, sharded_args)
            )

            # workaround for shard_map not supporting non-array args part 1/2
            nonarray_args = jax.tree.map(lambda a: not hasattr(a, "dtype"), args)
            args = jax.tree.map(
                lambda c, a: Partial(partial(lambda x: x, a)) if c else a,
                nonarray_args,
                args,
            )

            mesh = jax.sharding.get_abstract_mesh()
            in_specs = _sele2(sharded_args, P("S"), P())
            out_specs = out_treedef.unflatten(_sele2(reduction_op, P(), P("S")))

            @partial(
                jax.shard_map,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                axis_names={"S"},
                **kwargs,
            )
            def _f(*args):
                # workaround for shard_map not supporting non-array args part 2/2
                args = jax.tree.map(lambda c, a: a() if c else a, nonarray_args, args)

                # PRNGKey treatment 2/2
                args = tuple(
                    a[0] if c == "key" else a for a, c in safe_zip(args, sharded_args)
                )

                args = args_treedef.unflatten(args)

                # pvary
                args = jax.tree_util.tree_map(
                    lambda c, l: (
                        jax.tree_util.tree_map(partial(jax.lax.pvary, axis_name="S"), l)
                        if c
                        else l
                    ),
                    pvary_args_tree,
                    args,
                )

                res = f(*args)

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
                            jax.tree_util.tree_map, partial(o, axis_name="S")
                        )

                reductions = [_sele_op(o) for o in reduction_op]
                res = out_treedef.flatten_up_to(res)
                res = [f(r) for f, r in safe_zip(reductions, res)]
                res = out_treedef.unflatten(res)
                return res

            # We are sure that, so far, we have SHARD_MAP_STACK_LEVEL=0, so we can
            # safely call a @shard_map decorated function. The context manager
            # makes sure we only shard the outermost call.
            with _increase_SHARD_MAP_STACK_LEVEL():
                res = _f(*args)

            res = out_treedef.flatten_up_to(res)
            res = [a() if c is True else a for a, c in safe_zip(res, reduction_op)]
            res = out_treedef.unflatten(res)
            return res

        return _fun

    return f
