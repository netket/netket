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

from functools import partial

import dataclasses
from dataclasses import MISSING

from flax import serialization

import jax

from .utils import _set_new_attribute, _create_fn, get_class_globals

try:
    from dataclasses import _FIELDS
except ImportError:
    _FIELDS = "__dataclass_fields__"

_CACHES = "__dataclass_caches__"

PRECOMPUTE_CACHED_PROPERTY_NAME = "_precompute_cached_properties"
HASH_COMPUTE_NAME = "__dataclass_compute_hash__"
# The name of the function, that if it exists, is called before
# __init__ to preprocess the input arguments.
_PRE_INIT_NAME = "__pre_init__"
_DATACLASS_INIT_NAME = "__init_dataclass__"


def _cache_name(property_name):
    return "__" + property_name + "_cache"


def _hash_cache_name(class_name):
    return "__" + class_name + "_hash_cache"


def _compute_cache_name(property_name):
    return "__" + property_name


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
        pytree_node: a leaf node in the pytree representation of this dataclass.
            If False this must be hashable
        serialize: If True the node is included in the serialization.
            In general you should not specify this.
        cache: If True this node is a cache and will be reset every time
            fields are modified.
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
        self.name = method.__name__
        self.cache_name = _cache_name(self.name)
        self.method = method
        self.pytree_node = pytree_node
        self.type = method.__annotations__.get("return", MISSING)
        self.doc = method.__doc__

        if self.type is MISSING:
            raise TypeError(
                "Cached property {method} requires a return type annotation."
            )

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


def _set_annotation(clz, attr, typ):
    if "__annotations__" not in clz.__dict__:
        setattr(clz, "__annotations__", {})

    if not hasattr(clz, attr):
        raise ValueError(f"Setting annotation for nonexistent attribute {attr}")

    clz.__annotations__[attr] = typ


def process_cached_properties(clz, globals=None):
    """Looks for all attributes in clz, if anyone is a CachedProperty instance,
    which is a sential wrapper for methods, then create a cached attribute using
    dataclass language, set them as pytree_node=False so they are untracked.
    """

    if globals is None:
        globals = {}

    cached_props = {}
    self_name = "self"

    for name, field_info in clz.__dict__.items():
        if isinstance(field_info, CachedProperty):
            cached_props[name] = field_info

    # Convert a property to something like this
    # @cached_property
    # def myproperty(self) -> T
    #   return val
    #
    # becomes
    #
    # __myproperty_cache : T = UNINITIALIZED
    # @property
    # def myproperty(self) -> T
    #    if self.__myproperty_cache is Uninitialized:
    #        setattr(self, '__myproperty_cache', self.__myproperty())
    #    return self.__myproperty_cache

    # create the compute method
    for name, cp in cached_props.items():
        _set_new_attribute(clz, _compute_cache_name(name), cp.method)

    # Create the actual property accessor method
    for name, cp in cached_props.items():
        cache_name = _cache_name(name)
        compute_name = _compute_cache_name(name)

        # create accessor method
        body_lines = [
            f"if {self_name}.{cache_name} is Uninitialized:",
            f"\tBUILTINS.object.__setattr__({self_name},{cache_name!r},self.{compute_name}())",
            "",
            f"return {self_name}.{cache_name}",
        ]

        fun = _create_fn(
            name,
            [self_name],
            body_lines,
            return_type=cp.type,
            globals=globals,
            doc=cp.doc,
        )
        prop_fun = property(fun)
        setattr(clz, name, prop_fun)

    # merge caches among levels:
    for b in clz.__mro__[1:]:
        # Only process classes that have been processed by our
        # decorator.  That is, they have a _FIELDS attribute.
        for (name, cp) in getattr(b, _CACHES, {}).items():
            if name not in cached_props:
                cached_props[name] = cp

    # Create the cache member
    for name, cp in cached_props.items():
        cache_name = _cache_name(name)

        # Create the dataclass attribute
        _cache = field(
            pytree_node=cp.pytree_node,
            serialize=False,
            cache=True,
            default=Uninitialized,
            repr=False,
            hash=False,
            init=True,
            compare=False,
        )
        _set_new_attribute(clz, cache_name, _cache)
        _set_annotation(clz, cache_name, cp.type)

    # create precompute method
    _precompute_body_method = []
    for name in cached_props.keys():
        _precompute_body_method.append(f"{self_name}.{name}")

    # Create the precompute method
    if len(_precompute_body_method) == 0:
        _precompute_body_method.append("pass")

    fun = _create_fn(
        PRECOMPUTE_CACHED_PROPERTY_NAME,
        [self_name],
        _precompute_body_method,
        globals=globals,
        doc="Precompute the value of all cached properties",
    )
    setattr(clz, PRECOMPUTE_CACHED_PROPERTY_NAME, fun)

    setattr(clz, _CACHES, cached_props)


def purge_cache_fields(clz):
    """Removes the cache fields generated by netket dataclass
    from the dataclass mechanism.
    """
    flds = getattr(clz, _FIELDS, None)
    if flds is not None:
        caches = getattr(clz, _CACHES)
        for name, _ in caches.items():
            cname = _cache_name(name)
            if cname in flds:
                flds.pop(cname)


def attach_preprocess_init(data_clz, *, globals={}, init_doc=MISSING, cache_hash=False):

    # If there is no __pre_init__ method in the class, create a default
    # one calling pre init on super() if there is one.
    if _PRE_INIT_NAME not in data_clz.__dict__:

        def _preprocess_args_default(self, *args, **kwargs):
            if hasattr(super(data_clz, self), _PRE_INIT_NAME):
                args, kwargs = getattr(super(data_clz, self), _PRE_INIT_NAME)(
                    *args, **kwargs
                )

            return args, kwargs

        _set_new_attribute(data_clz, _PRE_INIT_NAME, _preprocess_args_default)

    # attach the current __init__ (generated by python's dataclass) to
    # __dataclass_init__
    _set_new_attribute(data_clz, _DATACLASS_INIT_NAME, data_clz.__init__)

    # Create a new init function calling __pre_init__ and then __dataclass_init__
    # If __precompute_cached_properties is True, then also precompute all cached
    # properties after having initialised the dataclass
    self_name = "self"
    body_lines = [
        "if not __skip_preprocess:",
        f"\targs, kwargs = {self_name}.{_PRE_INIT_NAME}(*args, **kwargs)",
        f"{self_name}.{_DATACLASS_INIT_NAME}(*args, **kwargs)",
        "if __precompute_cached_properties:",
        f"\t{self_name}.{PRECOMPUTE_CACHED_PROPERTY_NAME}()",
    ]

    # If requested, create the cache hash field
    if cache_hash:
        body_lines.append(
            f"BUILTINS.object.__setattr__({self_name},{_hash_cache_name(data_clz.__name__)!r},Uninitialized)"
        )

    fun = _create_fn(
        "__init__",
        [
            self_name,
            "*args",
            "__precompute_cached_properties=False",
            "__skip_preprocess=False",
            "**kwargs",
        ],
        body_lines,
        globals=globals,
    )
    if init_doc is MISSING:
        fun.__doc__ = getattr(data_clz, _PRE_INIT_NAME).__doc__
    else:
        fun.__doc__ = init_doc
    setattr(data_clz, "__init__", fun)


def replace_hash_method(data_clz, *, globals={}):
    """
    Replace __hash__ by a method that checks if it has already been computed
    and returns the cached value otherwise.
    """
    self_name = "self"
    hash_cache_name = _hash_cache_name(data_clz.__name__)

    body_lines = [
        f"if {self_name}.{hash_cache_name} is Uninitialized:",
        f"\tBUILTINS.object.__setattr__({self_name},{hash_cache_name!r},self.{HASH_COMPUTE_NAME}())",
        f"return {self_name}.{hash_cache_name}",
    ]

    fun = _create_fn(
        "__init__",
        [self_name],
        body_lines,
        globals=globals,
        doc="Return the cached hash. Computation performed in {HASH_COMPUTE_NAME}",
    )
    # move the __hash__ to __precompute__hash__
    setattr(data_clz, HASH_COMPUTE_NAME, data_clz.__hash__)
    # set the new hash function
    setattr(data_clz, "__hash__", fun)


def dataclass(clz=None, *, init_doc=MISSING, cache_hash=False, _frozen=True):
    """
    Decorator creating a NetKet-flavour dataclass.
    This behaves as a flax dataclass, that is a Frozen python dataclass, with a twist!
    See their documentation for standard behaviour.

    The new functionalities added by NetKet are:
     - it is possible to define a method `__pre_init__(*args, **kwargs) ->
       Tuple[Tuple,Dict]` that processes the arguments and keyword arguments provided
       to the dataclass constructor. This allows to deprecate argument names and add
       some logic to customize the constructors.
       This function should return a tuple of the edited `(args, kwargs)`. If
       inheriting from other classes it is recommended (though not mandated) to
       call the same method in parent classes. The function should return arguments and
       keyword arguments that will match the standard dataclass constructor.
       The function can also not be called in some internal cases, so it should not be
       a strict requirement to execute it.

     - Cached Properties. It is possible to mark properties of a netket dataclass with
       `@property_cached`. This will make the property behave as a standard property,
       but it's value is cached and reset every time a dataclass is manipulated.
       Cached properties can be part of the flattened pytree or not.
       See :ref:`netket.utils.struct.property_cached` for more info.

    Optional Args:
        init_doc: the docstring for the init method. Otherwise it's inherited
            from `__pre_init__`.
        cache_hash: If True the hash is computed only once and cached. Use if
            the computation is expensive.
        _frozen: (default True) controls whether the resulting class is frozen or not.
            If it is not frozen, extra care should be taken.
    """

    if clz is None:
        return partial(
            dataclass, init_doc=init_doc, cache_hash=cache_hash, _frozen=_frozen
        )

    # get globals of the class to put generated methods in there
    _globals = get_class_globals(clz)
    _globals["Uninitialized"] = Uninitialized
    # proces all cached properties
    process_cached_properties(clz, globals=_globals)
    # create the dataclass
    data_clz = dataclasses.dataclass(frozen=_frozen)(clz)
    purge_cache_fields(data_clz)
    # attach the custom preprocessing of init arguments
    attach_preprocess_init(
        data_clz, globals=_globals, init_doc=init_doc, cache_hash=cache_hash
    )
    if cache_hash:
        replace_hash_method(data_clz, globals=_globals)

    # flax stuff: identify states
    meta_fields = []
    data_fields = []
    for name, field_info in getattr(data_clz, _FIELDS, {}).items():
        is_pytree_node = field_info.metadata.get("pytree_node", True)
        if is_pytree_node:
            data_fields.append(name)
        else:
            meta_fields.append(name)

    # List the cache fields
    cache_fields = []
    for _, cp in getattr(data_clz, _CACHES, {}).items():
        cache_fields.append(cp.cache_name)
        # they count as struct fields
        if cp.pytree_node:
            data_fields.append(cp.cache_name)
        # they count as meta fields
        else:
            meta_fields.append(cp.cache_name)

    def replace(self, **updates):
        """Returns a new object replacing the specified fields with new values."""
        # reset cached fields
        for name in cache_fields:
            updates[name] = Uninitialized

        return dataclasses.replace(self, **updates, __skip_preprocess=True)

    data_clz.replace = replace

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
