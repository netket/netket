import dataclasses
import inspect
import typing as tp
from abc import ABCMeta
from copy import copy
from functools import partial
from types import MappingProxyType

import jax

from .fields import CachedProperty, _cache_name, _raw_cache_name, Uninitialized
from netket.utils import config

P = tp.TypeVar("P", bound="Pytree")

DATACLASS_USER_INIT_N_ARGS = "_pytree_n_args_max"
"""
variable name used by dataclasses inheriting from a pytree to
store the topmost non-dataclass class in a mro.
"""


class PytreeMeta(ABCMeta):
    """
    Metaclass for PyTrees, takes care of initializing and turning
    frozen PyTrees to immutable after __init__.
    """

    def __call__(cls: type[P], *args: tp.Any, **kwargs: tp.Any) -> P:
        obj: P = cls.__new__(cls, *args, **kwargs)
        obj.__dict__["_pytree__initializing"] = True
        try:
            obj.__init__(*args, **kwargs)
        finally:
            del obj.__dict__["_pytree__initializing"]

        vars_dict = vars(obj)
        if obj._pytree__class_dynamic_nodes:
            vars_dict["_pytree__node_fields"] = tuple(
                sorted(
                    field
                    for field in vars_dict
                    if field not in cls._pytree__static_fields
                )
            )
        else:
            vars_dict["_pytree__node_fields"] = cls._pytree__data_fields

        for field in obj._pytree__cachedprop_fields:
            vars_dict[field] = Uninitialized

        return obj


class Pytree(metaclass=PytreeMeta):
    """
    Astract Base class for jaw-aware PyTree classes.

    A class inheriting from PyTree can be passed to a jax function
    as a standard argument, and can contain both static and dynamic
    fields. Those will be correctly handled when flattening the
    PyTree.

    Static fields must be specified as class attributes, by specifying
    the :func:`nk.utils.struct.field(pytree_node=False)`.

    Example:
        Construct a PyTree with a 'constant' value

        >>> from netket.utils.struct import field, Pytree
        >>> import jax
        >>>
        >>> class MyPyTree(Pytree):
        ...     a: int = field(pytree_node=False)
        ...     b: jax.Array
        ...     def __init__(self, a, b):
        ...         self.a = a
        ...         self.b = b
        ...     def __repr__(self):
        ...         return f"MyPyTree(a={self.a}, b={self.b})"
        >>>
        >>> my_pytree = MyPyTree(1, jax.numpy.ones(2))
        >>> jax.jit(lambda x: print(x))(my_pytree)  # doctest:+ELLIPSIS
            MyPyTree(a=1, b=Traced...


    PyTree classes by default are not mutable, therefore they
    behave similarly to frozen dataclasses. If you want to
    make a PyTree mutable, you can specify the `mutable=True`
    argument in the class definition.

    Example:

        >>> from netket.utils.struct import field, Pytree
        >>>
        >>> class MyPyTree(Pytree, mutable=True):
        ...     a: int = field(pytree_node=False)
        ...     ...

    By default only the fields declared as class attributes
    can be set and/or modified after initialization. If you
    want to allow the creation of new fields after initialization,
    you can specify the `dynamic_nodes=True` argument in the
    class definition.

    PyTree classes can also be inherited by a netket dataclass,
    in which case the dataclass will be initialized with the
    fields of the PyTree. However, this behaviour is deprecated
    and will be removed in the future. We suggest you to remove
    the `@nk.utils.struct.dataclass` decorator and simply define
    an `__init__` method.
    """

    _pytree__initializing: bool
    _pytree__class_is_mutable: bool
    _pytree__static_fields: tuple[str, ...]
    _pytree__node_fields: tuple[str, ...]
    _pytree__setter_descriptors: frozenset[str]

    _pytree__cachedprop_fields: tuple[str, ...]

    def __init_subclass__(cls, mutable: bool = False, dynamic_nodes: bool = False):
        super().__init_subclass__()

        # gather class info
        class_vars = vars(cls)
        setter_descriptors = set()
        static_fields = _inherited_static_fields(cls)

        # add special static fields
        static_fields.add("_pytree__node_fields")

        # new
        data_fields = _inherited_data_fields(cls)
        cached_prop_fields = set()

        for field, value in class_vars.items():
            if isinstance(value, dataclasses.Field) and not value.metadata.get(
                "pytree_node", True
            ):
                static_fields.add(field)
            elif isinstance(value, CachedProperty):
                cached_prop_fields.add(field)
            elif isinstance(value, dataclasses.Field) and value.metadata.get(
                "pytree_node", True
            ):
                data_fields.add(field)

            # add setter descriptors
            if hasattr(value, "__set__"):
                setter_descriptors.add(field)
        for field in cached_prop_fields:
            # setattr(cls, _cache_name(field), Uninitialized)
            if class_vars[field].pytree_node:
                data_fields.add(_cache_name(field))
            else:
                static_fields.add(_cache_name(field))
        cached_prop_fields = cached_prop_fields.union(
            _inherited_cachedproperty_fields(cls)
        )

        # If no annotations in this class, skip, otherwise we'd process
        # parent's annotations twice
        if "__annotations__" in cls.__dict__:
            # fields that are only type annotations, feed them forward
            for field, _ in cls.__annotations__.items():
                if field not in static_fields and field not in data_fields:
                    data_fields.add(field)

        if mutable and len(cached_prop_fields) != 0:
            raise ValueError("cannot use cached properties with " "mutable pytrees.")

        if config.netket_sphinx_build:
            for k in static_fields:
                try:
                    delattr(cls, k)
                except AttributeError:
                    pass
            for k in data_fields:
                try:
                    delattr(cls, k)
                except AttributeError:
                    pass

        # new
        init_fields = tuple(sorted(data_fields.union(static_fields)))
        data_fields = tuple(sorted(data_fields))
        cached_prop_fields = tuple(sorted(cached_prop_fields))
        cached_prop_fields = tuple(_cache_name(f) for f in cached_prop_fields)

        static_fields = tuple(sorted(static_fields))

        # init class variables
        cls._pytree__initializing = False
        cls._pytree__class_is_mutable = mutable
        cls._pytree__static_fields = static_fields
        cls._pytree__setter_descriptors = frozenset(setter_descriptors)

        # new
        cls._pytree__class_dynamic_nodes = dynamic_nodes
        cls._pytree__data_fields = data_fields
        cls._pytree__cachedprop_fields = cached_prop_fields
        cls._pytree__init_fields = init_fields

        # TODO: clean up this in the future once minimal supported version is 0.4.7
        if (
            "flatten_func"
            in inspect.signature(jax.tree_util.register_pytree_with_keys).parameters
        ):
            jax.tree_util.register_pytree_with_keys(
                cls,
                partial(
                    cls._pytree__flatten,
                    with_key_paths=True,
                ),
                cls._pytree__unflatten,
                flatten_func=partial(
                    cls._pytree__flatten,
                    with_key_paths=False,
                ),
            )
        else:
            jax.tree_util.register_pytree_with_keys(
                cls,
                partial(
                    cls._pytree__flatten,
                    with_key_paths=True,
                ),
                cls._pytree__unflatten,
            )

        # flax serialization support
        from flax import serialization

        serialization.register_serialization_state(
            cls,
            partial(cls._to_flax_state_dict, cls._pytree__static_fields),
            partial(cls._from_flax_state_dict, cls._pytree__static_fields),
        )

    def __pre_init__(self, *args, **kwargs):
        # Default implementation of __pre_init__, used by netket's
        # dataclasses for preinitialisation shuffling of parameters.
        #
        # This is necessary for PyTrees that are subclassed by a dataclass
        # (like a user-implemented sampler using legacy logic).
        #
        # This class takes out all arguments and kw-arguments that are
        # directed to the PyTree from a processing and 'hides' them
        # in a proprietary kwargument for later manipulation.
        #
        # This is necessary so we call the dataclass init only with
        # the arguments that it needs.

        # process keyword arguments
        kwargs_dataclass = {}
        kwargs_pytree = {}
        for k, v in kwargs.items():
            if k in self.__dataclass_fields__.keys():
                kwargs_dataclass[k] = v
            else:
                kwargs_pytree[k] = v

        # process positional args. Identify max positional arguments of the
        # topmost user defined init method
        max_pytree_args = getattr(self, DATACLASS_USER_INIT_N_ARGS, len(args))
        n_args_pytree = min(len(args), max_pytree_args)

        # First n args are for the pytree initialiser (lower) and later
        # positional arguments are for the dataclass initializer
        args_pytree = args[:n_args_pytree]
        args_dataclass = args[n_args_pytree:]

        signature_pytree = (args_pytree, kwargs_pytree)
        kwargs_dataclass["__base_init_args"] = signature_pytree

        return args_dataclass, kwargs_dataclass

    def __post_init__(self):
        pass

    @classmethod
    def _pytree__flatten(
        cls,
        pytree: "Pytree",
        *,
        with_key_paths: bool,
    ) -> tuple[tuple[tp.Any, ...], tp.Mapping[str, tp.Any],]:
        all_vars = vars(pytree).copy()
        static = {k: all_vars.pop(k) for k in pytree._pytree__static_fields}

        if with_key_paths:
            node_values = tuple(
                (jax.tree_util.GetAttrKey(field), all_vars.pop(field))
                for field in pytree._pytree__node_fields
            )
        else:
            node_values = tuple(
                all_vars.pop(field) for field in pytree._pytree__node_fields
            )

        if all_vars:
            raise ValueError(
                f"Unexpected fields in {cls.__name__}: {', '.join(all_vars.keys())}."
                "You cannot add new fields to a Pytree after it has been initialized."
            )

        return node_values, MappingProxyType(static)

    @classmethod
    def _pytree__unflatten(
        cls: type[P],
        static_fields: tp.Mapping[str, tp.Any],
        node_values: tuple[tp.Any, ...],
    ) -> P:
        pytree = object.__new__(cls)
        pytree.__dict__.update(zip(static_fields["_pytree__node_fields"], node_values))
        pytree.__dict__.update(static_fields)
        return pytree

    @classmethod
    def _to_flax_state_dict(
        cls, static_field_names: tuple[str, ...], pytree: "Pytree"
    ) -> dict[str, tp.Any]:
        from flax import serialization

        state_dict = {
            name: serialization.to_state_dict(getattr(pytree, name))
            for name in pytree.__dict__
            if name not in static_field_names
        }
        return state_dict

    @classmethod
    def _from_flax_state_dict(
        cls,
        static_field_names: tuple[str, ...],
        pytree: P,
        state: dict[str, tp.Any],
    ) -> P:
        """Restore the state of a data class."""
        from flax import serialization

        state = state.copy()  # copy the state so we can pop the restored fields.
        updates = {}
        for name in pytree.__dict__:
            if name in static_field_names:
                continue
            if name not in state:
                raise ValueError(
                    f"Missing field {name} in state dict while restoring"
                    f" an instance of {type(pytree).__name__},"
                    f" at path {serialization.current_path()}"
                )
            value = getattr(pytree, name)
            value_state = state.pop(name)
            updates[name] = serialization.from_state_dict(value, value_state, name=name)
        if state:
            names = ",".join(state.keys())
            raise ValueError(
                f'Unknown field(s) "{names}" in state dict while'
                f" restoring an instance of {type(pytree).__name__}"
                f" at path {serialization.current_path()}"
            )
        return pytree.replace(**updates)

    def replace(self: P, **kwargs: tp.Any) -> P:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. If the object is a dataclass, `dataclasses.replace`
        will be used. Otherwise, a new object will be created with the same
        type as the original object.
        """
        if dataclasses.is_dataclass(self):
            return dataclasses.replace(self, **kwargs)

        unknown_keys = set(kwargs) - set(vars(self))
        if unknown_keys:
            raise ValueError(
                f"Trying to replace unknown fields {unknown_keys} "
                f"for '{type(self).__name__}'"
            )

        pytree = copy(self)
        pytree.__dict__.update(kwargs)
        return pytree

    if not tp.TYPE_CHECKING:

        def __setattr__(self: P, field: str, value: tp.Any):
            if self._pytree__initializing:
                if self._pytree__class_dynamic_nodes:
                    pass
                elif field not in self._pytree__init_fields:
                    raise AttributeError(
                        f"Cannot set field {field} in init that was not described "
                        "as a class attribute above."
                    )
            else:
                if field in self._pytree__setter_descriptors:
                    pass
                elif field in self._pytree__cachedprop_fields:
                    pass
                elif not hasattr(self, field):
                    raise AttributeError(
                        f"Cannot add new fields to {type(self)} after initialization"
                    )
                elif not self._pytree__class_is_mutable:
                    raise AttributeError(
                        f"{type(self)} is immutable, trying to update field {field}"
                    )

            object.__setattr__(self, field, value)


def _inherited_static_fields(cls: type) -> set[str]:
    """
    Returns the set of static fields of base classes
    of the input class
    """
    static_fields = set()
    for parent_class in cls.mro():
        if parent_class is not cls and parent_class is not Pytree:
            if issubclass(parent_class, Pytree):
                static_fields.update(parent_class._pytree__static_fields)
            elif dataclasses.is_dataclass(parent_class):
                for field in dataclasses.fields(parent_class):
                    if not field.metadata.get("pytree_node", True):
                        static_fields.add(field.name)
    return static_fields


def _inherited_data_fields(cls: type) -> set[str]:
    """
    Returns the set of data fields of base classes
    of the input class.
    """
    data_fields = set()
    for parent_class in cls.mro():
        if parent_class is not cls and parent_class is not Pytree:
            if issubclass(parent_class, Pytree):
                data_fields.update(parent_class._pytree__data_fields)
            elif dataclasses.is_dataclass(parent_class):
                for field in dataclasses.fields(parent_class):
                    if field.metadata.get("pytree_node", True):
                        data_fields.add(field.name)
    return data_fields


def _inherited_cachedproperty_fields(cls: type) -> set[str]:
    """
    Returns the set of cached properties of base classes
    of the input class.
    """
    cachedproperty_fields = set()
    for parent_class in cls.mro():
        if parent_class is not cls and parent_class is not Pytree:
            if issubclass(parent_class, Pytree):
                fields = tuple(
                    _raw_cache_name(f) for f in parent_class._pytree__cachedprop_fields
                )
                cachedproperty_fields.update(fields)
            elif dataclasses.is_dataclass(parent_class):
                for field in dataclasses.fields(parent_class):
                    if isinstance(field, CachedProperty):
                        cachedproperty_fields.add(field.name)
    return cachedproperty_fields
