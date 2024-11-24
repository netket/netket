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
    pytree_node: bool = True,
    serialize: bool | None = None,
    serialize_name: str | None = None,
    cache: bool = False,
    sharded: bool | ShardedFieldSpec = False,
    **kwargs,
):
    """Mark a field of a dataclass or PyTree to be:

    Args:
        pytree_node: a leaf node in the pytree representation of this dataclass.
            If False this must be hashable
        serialize: If True the node is included in the serialization.
            In general you should not specify this. (Defaults to value of pytree_node).
        serialize_name: If specified, it's the name under which this attribute is serialized.
            This can be used to change the runtime attribute name, but maintain some
            other name in the serialisation format.
        cache: If True this node is a cache and will be reset every time
            fields are modified.
        sharded: a boolan or specification object specifying whether this entry is sharded.
            Defaults to False. If True, a MPI-compatible sharding along axis 0 is assumed.
    """
    if serialize is None:
        serialize = pytree_node
    if sharded is True:
        sharded = ShardedFieldSpec()

    metadata = {
        "pytree_node": pytree_node,
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
    return field(pytree_node=False, **kwargs)


class CachedProperty:
    """Sentinel attribute wrapper to signal that a method is a property
    but must be cached.
    """

    def __init__(self, method, pytree_node=False):
        self.name = method.__name__
        self.cache_name = _cache_name(self.name)
        self.method = method
        self.pytree_node = pytree_node
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
            f"type={self.type}, pytree_node={self.pytree_node})"
        )


def property_cached(fun=None, pytree_node=False):
    """Decorator to make the method behave as a property but cache the resulting value and
    clears it upon replace.

    Args:
        pytree_node: a leaf node in the pytree representation of this dataclass.
            If False this must be hashable
    """
    if fun is None:
        return partial(property_cached, pytree_node=pytree_node)

    return CachedProperty(fun, pytree_node=pytree_node)
