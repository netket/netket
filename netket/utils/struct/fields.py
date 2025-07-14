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
Common utilities to define fields and properties of netket dataclasses
and Pytrees.
"""

from functools import partial

import dataclasses
from dataclasses import MISSING

import jax

from .pytree_serialization_sharding import ShardedFieldSpec
from ..deprecation import warn_deprecation


def _cache_name(property_name):
    return "__" + property_name + "_cache"


def _raw_cache_name(cache_name):
    # removes leading __ and ending _cache
    return cache_name[2:-6]


## Our stuff
class _Uninitialized:
    """
    Sentinel value used to signal uninitialized values
    """

    def __repr__(self):
        return "Uninitialized"


Uninitialized = _Uninitialized()

jax.tree_util.register_pytree_node(
    _Uninitialized, lambda x: ((), Uninitialized), lambda *args: Uninitialized
)


def field(
    pytree_node: bool | None = None,
    static: bool = False,
    serialize: bool | None = None,
    serialize_name: str | None = None,
    cache: bool = False,
    sharded: bool | ShardedFieldSpec = False,
    **kwargs,
):
    """Mark a field of a dataclass or PyTree to be:

    Args:
        pytree_node: [DEPRECATED] a leaf node in the pytree representation of this dataclass.
            If False this must be hashable. Use 'static' instead.
        static: If True, this field is static (not a pytree node) and must be hashable.
            Defaults to False.
        serialize: If True the node is included in the serialization.
            In general you should not specify this. (Defaults to not static).
        serialize_name: If specified, it's the name under which this attribute is serialized.
            This can be used to change the runtime attribute name, but maintain some
            other name in the serialisation format.
        cache: If True this node is a cache and will be reset every time
            fields are modified.
        sharded: a boolan or specification object specifying whether this entry is sharded.
            Defaults to False. If True, a JAX-compatible sharding along axis 0 is assumed.
    """
    # Handle deprecation warning for pytree_node
    if pytree_node is not None:
        warn_deprecation(
            "The 'pytree_node' argument is deprecated. Use 'static=True' instead of 'pytree_node=False' "
            "and 'static=False' instead of 'pytree_node=True'."
        )
        # Convert pytree_node to static (opposite behavior)
        static = not pytree_node
    
    if serialize is None:
        serialize = not static
    if sharded is True:
        sharded = ShardedFieldSpec()

    metadata = {
        "pytree_node": not static,  # Keep for backward compatibility
        "static": static,
        "serialize": serialize,
        "cache": cache,
        "sharded": sharded,
    }
    if serialize_name is not None:
        metadata["serialize_name"] = serialize_name
    return dataclasses.field(
        metadata=metadata,
        **kwargs,
    )


def static_field(**kwargs):
    return field(static=True, **kwargs)


class CachedProperty:
    """Sentinel attribute wrapper to signal that a method is a property
    but must be cached.
    """

    def __init__(self, method, pytree_node=None, static=True):
        self.name = method.__name__
        self.cache_name = _cache_name(self.name)
        self.method = method
        
        # Handle deprecation warning for pytree_node
        if pytree_node is not None:
            warn_deprecation(
                "The 'pytree_node' argument in CachedProperty is deprecated. "
                "Use 'static=False' instead of 'pytree_node=True' "
                "and 'static=True' instead of 'pytree_node=False'."
            )
            static = not pytree_node
        
        self.static = static
        self.pytree_node = not static  # Keep for backward compatibility
        self.type = method.__annotations__.get("return", MISSING)
        self.doc = method.__doc__

        if self.type is MISSING:
            raise TypeError(
                f"Cached property {method} requires a return type annotation."
            )

    def __get__(self, obj, objtype=None):
        val = getattr(obj, self.cache_name, Uninitialized)
        if val is Uninitialized:
            val = self.method(obj)
            setattr(obj, self.cache_name, val)
        return val

    def __repr__(self):
        return (
            f"CachedProperty(name={self.name}, "
            f"type={self.type}, static={self.static})"
        )


def property_cached(fun=None, pytree_node=None, static=True):
    """Decorator to make the method behave as a property but cache the resulting value and
    clears it upon replace.

    Args:
        pytree_node: [DEPRECATED] a leaf node in the pytree representation of this dataclass.
            If False this must be hashable. Use 'static' instead.
        static: If True, this cached property is static (not a pytree node) and must be hashable.
            Defaults to True.
    """
    if fun is None:
        return partial(property_cached, pytree_node=pytree_node, static=static)

    return CachedProperty(fun, pytree_node=pytree_node, static=static)
