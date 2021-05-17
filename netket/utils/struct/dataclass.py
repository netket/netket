# Copyright 2021 The NetKet Authors - All rights reserved.
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

# Part of this code has been copy-pasted from the google/flax repository
# the copyright notice is reproduced below

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for defining custom classes that can be used with jax transformations.
"""

from typing import TypeVar
import sys
import inspect
from functools import partial

import dataclasses
from dataclasses import MISSING

import builtins

from flax import serialization

import jax

from .utils import _set_new_attribute, _create_fn, get_class_globals

## END

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


def field(pytree_node=True, serialize=True, cache=False, **kwargs):
    """Mark a field of a dataclass to be:

    Args:
        pytree_node: a leaf node in the pytree representation of this dataclass. If False this must be hashable
        serialize: If True the node is included in the serialization. In general you should not specify this.
        cache: If True this node is a cache and will be reset every time fields are modified.
    """
    return dataclasses.field(
        metadata={"pytree_node": pytree_node, "serialize": serialize, "cache": cache},
        **kwargs,
    )


class CachedProperty:
    """Sentinel attribute wrapper to signal that a method is a property
    but must be cached.
    """

    def __init__(self, method, pytree_node=False):
        self.method = method
        self.pytree_node = pytree_node


def property_cached(fun):
    """Decorator to make the method behave as a property but cache the resulting value and
    clears it upon replace.
    """
    # if fun is None:
    #    return partial(property_cached, pytree_node=pytree_node)

    return CachedProperty(fun, pytree_node=False)


def process_cached_properties(clz, globals={}):
    """Looks for all attributes in clz, if anyone is a CachedProperty instance,
    which is a sential wrapper for methods, then create a cached attribute using
    dataclass language, set them as pytree_node=False so they are untracked.
    """

    cached_props = {}
    self_name = "self"

    for name, field_info in clz.__dict__.items():
        if isinstance(field_info, CachedProperty):
            cached_props[name] = field_info

    _precompute_body_method = []
    for name, cp in cached_props.items():
        method = cp.method
        pytree_node = cp.pytree_node

        cache_name = "__" + name + "_cache"
        compute_name = "__" + name
        return_type = method.__annotations__.get("return", MISSING)

        _cache = field(
            pytree_node=pytree_node,
            serialize=False,
            default=Uninitialized,
            repr=False,
            hash=False,
            init=True,
            compare=False,
        )
        _set_new_attribute(clz, cache_name, _cache)
        _set_new_attribute(clz, compute_name, method)
        clz.__annotations__[cache_name] = return_type
        # create accessor method
        body_lines = [
            f"if {self_name}.{cache_name} is Uninitialized:",
            f"\tBUILTINS.object.__setattr__({self_name},{cache_name!r},self.{compute_name}())",
            f"",
            f"return {self_name}.{cache_name}",
        ]

        fun = _create_fn(
            name, [self_name], body_lines, return_type=return_type, globals=globals
        )
        fun.__doc__ = method.__doc__
        prop_fun = property(fun)
        setattr(clz, name, prop_fun)

        _precompute_body_method.append(f"{self_name}.{name}")

    if len(_precompute_body_method) == 0:
        _precompute_body_method.append("pass")

    fun = _create_fn(name, [self_name], _precompute_body_method, globals=globals)
    fun.__doc__ = "Precompute the value of all cached properties"
    setattr(clz, "_precompute_cached_properties", fun)


def attach_preprocess_init(data_clz, init_doc=MISSING):
    preprocess_method_name = "__pre_init__"
    dataclass_init_name = "__init_dataclass__"

    if not preprocess_method_name in data_clz.__dict__:

        def _preprocess_args_default(self, *args, **kwargs):
            if hasattr(super(data_clz, self), preprocess_method_name):
                args, kwargs = getattr(super(data_clz, self), preprocess_method_name)(
                    *args, **kwargs
                )

            return args, kwargs

        _set_new_attribute(
            data_clz, preprocess_method_name, _preprocess_args_default
        )  # lambda *args, **kwargs: args, kwargs)

    _set_new_attribute(data_clz, dataclass_init_name, data_clz.__init__)

    self_name = "self"
    body_lines = [
        f"if not __skip_preprocess:",
        f"\targs, kwargs = {self_name}.{preprocess_method_name}(*args, **kwargs)",
        f"{self_name}.{dataclass_init_name}(*args, **kwargs)",
    ]

    fun = _create_fn(
        "__init__",
        [self_name, "*args", "__skip_preprocess=False", "**kwargs"],
        body_lines,
    )
    if init_doc is MISSING:
        fun.__doc__ = getattr(data_clz, preprocess_method_name).__doc__
    else:
        fun.__doc__ = init_doc
    setattr(data_clz, "__init__", fun)


def dataclass(clz=None, *, init_doc=MISSING):
    """
    Decorator creating a NetKet-flavour dataclass.
    This behaves as a flax dataclass, that is a Frozen python dataclass, with a twist!
    See their documentation for standard behaviour.

    The new functionalities added by NetKet are:
     - it is possible to define a method `__pre_init__(*args, **kwargs) -> Tuple[Tuple,Dict]` that processes the arguments
       and keyword arguments provided to the dataclass constructor. This allows to deprecate argument
       names and add some logic to customize the constructors.
       This function should return a tuple of the edited `(args, kwargs)`. If inheriting from other classes it is reccomended
       (though not mandated) to call the same method in parent classes.
       The function should return arguments and keyword arguments that will match the standard dataclass constructor.
       The function can also not be called in some internal cases, so it should not be a strict requirement to execute it.

     - Cached Properties. It is possible to mark properties of a netket dataclass with `@property_cached`. This will make the
       property behave as a standard property, but it's value is cached and reset every time a dataclass is manipulated.
       Cached properties can be part of the flattened pytree or not. See :ref:`netket.utils.struct.property_cached` for more info.

    Optinal Args:
        init_doc: the docstring for the init method. Otherwise it's inherited from `__pre_init__`.

    """

    if clz is None:
        return partial(dataclass, init_doc=init_doc)

    # get globals of the class to put generated methods in there
    _globals = get_class_globals(clz)
    _globals["Uninitialized"] = Uninitialized
    # proces all cached properties
    process_cached_properties(clz, globals=_globals)
    # create the dataclass
    data_clz = dataclasses.dataclass(frozen=True)(clz)
    # attach the custom preprocessing of init arguments
    attach_preprocess_init(data_clz, init_doc=init_doc)

    # when replacing reset the cache fields
    cache_fields = []
    non_cache_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        is_cache = field_info.metadata.get("cache", False)
        if is_cache:
            cache_fields.append(name)
        else:
            non_cache_fields.append(name)

    def replace(self, **updates):
        """"Returns a new object replacing the specified fields with new values."""
        # reset cached fields
        for name in cache_fields:
            updates[name] = Uninitialized
        return dataclasses.replace(self, **updates)

    data_clz.replace = replace

    # flax stuff: identify states
    meta_fields = []
    data_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        is_pytree_node = field_info.metadata.get("pytree_node", True)
        if is_pytree_node:
            data_fields.append(name)
        else:
            meta_fields.append(name)

    # support for jax pytree flattening unflattening
    def iterate_clz(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta

    def clz_from_iterable(meta, data):
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)
        return data_clz(__skip_preprocess=True, **kwargs)

    jax.tree_util.register_pytree_node(data_clz, iterate_clz, clz_from_iterable)

    # flax serialization
    skip_serialize_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        if not field_info.metadata.get("serialize", True):
            skip_serialize_fields.append(name)

    def to_state_dict(x):
        state_dict = {
            name: serialization.to_state_dict(getattr(x, name))
            for name in data_fields
            if name not in skip_serialize_fields
        }
        return state_dict

    def from_state_dict(x, state):
        """Restore the state of a data class."""
        state = state.copy()  # copy the state so we can pop the restored fields.
        updates = {}
        for name in data_fields:
            if name not in skip_serialize_fields:
                if name not in state:
                    raise ValueError(
                        f"Missing field {name} in state dict while restoring"
                        f" an instance of {clz.__name__}"
                    )
                value = getattr(x, name)
                value_state = state.pop(name)
                updates[name] = serialization.from_state_dict(value, value_state)
        if state:
            names = ",".join(state.keys())
            raise ValueError(
                f'Unknown field(s) "{names}" in state dict while'
                f" restoring an instance of {clz.__name__}"
            )
        return x.replace(**updates)

    serialization.register_serialization_state(data_clz, to_state_dict, from_state_dict)

    return data_clz
