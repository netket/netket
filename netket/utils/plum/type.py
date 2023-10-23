import abc
import sys
import typing
import warnings
from typing import Literal, get_args, get_origin

try:  # pragma: specific no cover 3.8 3.9
    from types import UnionType
except ImportError:  # pragma: specific no cover 3.10 3.11

    class UnionType:
        """Replacement for :class:`types.UnionType`."""


__all__ = [
    "PromisedType",
    "ModuleType",
    "type_mapping",
    "resolve_type_hint",
    "is_faithful",
]


class ResolvableType(type):
    """A resolvable type that will resolve to `type` after `type` has been delivered via
    :meth:`.ResolvableType.deliver`. Before then, it will resolve to itself.

    Args:
        name (str): Name of the type to be delivered.
    """

    def __init__(self, name):
        type.__init__(self, name, (), {})
        self._type = None

    def __new__(self, name):
        return type.__new__(self, name, (), {})

    def deliver(self, type):
        """Deliver the type.

        Args:
            type (type): Type to deliver.

        Returns:
            :class:`ResolvableType`: `self`.
        """
        self._type = type
        return self

    def resolve(self):
        """Resolve the type.

        Returns:
            type: If no type has been delivered, this will return itself. If a type
                `type` has been delivered via :meth:`.ResolvableType.deliver`, this will
                return that type.
        """
        if self._type is None:
            return self
        else:
            return self._type


class PromisedType(ResolvableType):
    """A type that is promised to be available when you will you need it.

    Args:
        name (str, optional): Name of the type that is promised. Defaults to
            `"SomeType"`.
    """

    def __init__(self, name="SomeType"):
        ResolvableType.__init__(self, f"PromisedType[{name}]")
        self._name = name

    def __new__(cls, name="SomeType"):
        return ResolvableType.__new__(cls, f"PromisedType[{name}]")


class ModuleType(ResolvableType):
    """A type from another module.

    Args:
        module (str): Module that the type lives in.
        name (str): Name of the type that is promised.
    """

    def __init__(self, module, name):
        if module in {"__builtin__", "__builtins__"}:
            module = "builtins"
        ResolvableType.__init__(self, f"ModuleType[{module}.{name}]")
        self._name = name
        self._module = module

    def __new__(cls, module, name):
        return ResolvableType.__new__(cls, f"ModuleType[{module}.{name}]")

    def retrieve(self):
        """Attempt to retrieve the type from the reference module.

        Returns:
            :class:`ModuleType`: `self`.
        """
        if self._type is None:
            if self._module in sys.modules:
                type = sys.modules[self._module]
                for name in self._name.split("."):
                    type = getattr(type, name)
                self.deliver(type)
        return self._type is not None


def _is_hint(x):
    """Check if an object is a type hint.

    Args:
        x (object): Object.

    Returns:
        bool: `True` if `x` is a type hint and `False` otherwise.
    """
    try:
        if x.__module__ == "builtins":  # pragma: specific no cover 3.8
            # Check if `x` is a subscripted built-in. We do this by checking the module
            # of the type of `x`.
            x = type(x)
        return x.__module__ in {
            "types",  # E.g., `tuple[int]`
            "typing",
            "collections.abc",  # E.g., `Callable`
        }
    except AttributeError:
        return False


def _hashable(x):
    """Check if an object is hashable.

    Args:
        x (object): Object to check.

    Returns:
        bool: `True` if `x` is hashable and `False` otherwise.
    """
    try:
        hash(x)
        return True
    except TypeError:
        return False


type_mapping = {}
"""dict: When running :func:`resolve_type_hint`, map keys in this dictionary to the
values."""


def resolve_type_hint(x):
    """Resolve all :class:`ResolvableType` in a type or type hint.

    Args:
        x (type or type hint): Type hint.

    Returns:
        type or type hint: `x`, but with all :class:`ResolvableType`\\s resolved.
    """
    if _hashable(x) and x in type_mapping:
        return resolve_type_hint(type_mapping[x])
    elif _is_hint(x):
        origin = get_origin(x)
        args = get_args(x)
        if args == ():
            # `origin` might not make sense here. For example, `get_origin(Any)` is
            # `None`. Since the hint wasn't subscripted, the right thing is to right the
            # hint itself.
            return x
        else:
            if origin is UnionType:  # pragma: specific no cover 3.8 3.9
                # The new union syntax was used.
                y = args[0]
                for arg in args[1:]:
                    y = y | arg
                return y
            else:
                # Do not resolve the arguments for `Literal`s.
                if origin != Literal:
                    args = resolve_type_hint(args)
                try:
                    return origin[args]
                except TypeError as e:  # pragma: specific no cover 3.9 3.10 3.11
                    # In Python 3.8, the origin might be a type that cannot be
                    # subscripted. As a workaround, we get the name of the type,
                    # capitalize it, and try to get it from `typing`. So far, this
                    # seems to have worked fine.
                    if sys.version_info.minor <= 8:
                        return getattr(typing, origin.__name__.capitalize())[args]
                    else:  # pragma: no cover
                        # This branch can never be reached.
                        raise e

    elif x is None:
        return x
    elif x is Ellipsis:
        return x

    elif isinstance(x, tuple):
        return tuple(resolve_type_hint(arg) for arg in x)
    elif isinstance(x, list):
        return list(resolve_type_hint(arg) for arg in x)
    elif isinstance(x, type):
        if isinstance(x, ResolvableType):
            if isinstance(x, ModuleType):
                if not x.retrieve():
                    # If the type could not be retrieved, then just return the
                    # wrapper. Namely, `x.resolve()` will then return `x`, which means
                    # that the below call will result in an infinite recursion.
                    return x
            return resolve_type_hint(x.resolve())
        else:
            return x

    else:
        warnings.warn(
            f"Could not resolve the type hint of `{x}`. "
            f"I have ended the resolution here to not make your code break, but some "
            f"types might not be working correctly. "
            f"Please open an issue at https://github.com/wesselb/plum.",
            stacklevel=2,
        )
        return x


def is_faithful(x):
    """Check whether a type hint is faithful.

    A type or type hint `t` is defined _faithful_ if, for all `x`, the following holds
    true::

        isinstance(x, x) == issubclass(type(x), t)

    You can control whether types are faithful or not by setting the attribute
    `__faithful__`::

        class UnfaithfulType:
            __faithful__ = False

    Args:
        x (type or type hint): Type hint.

    Returns:
        bool: Whether `x` is faithful or not.
    """
    return _is_faithful(resolve_type_hint(x))


def _is_faithful(x):
    if _is_hint(x):
        origin = get_origin(x)
        args = get_args(x)
        if args == ():
            # Unsubscripted type hints tend to be faithful. For example, `Any`, `List`,
            # `Tuple`, `Dict`, `Callable`, and `Generator` are. When we come across a
            # counter-example, we will refine this logic.
            return True
        else:
            if origin in {typing.Union, typing.Optional}:
                return all(is_faithful(arg) for arg in args)
            else:
                return False

    elif x is None:
        return True
    elif x == Ellipsis:
        return True

    elif isinstance(x, (tuple, list)):
        return all(is_faithful(arg) for arg in x)
    elif isinstance(x, type):
        if hasattr(x, "__faithful__"):
            return x.__faithful__
        else:
            # This is the fallback method. Check whether `__instancecheck__` is default
            # or not. If it is, assume that it is faithful.
            return type(x).__instancecheck__ in {
                type.__instancecheck__,
                abc.ABCMeta.__instancecheck__,
            }
    else:
        warnings.warn(
            f"Could not determine whether `{x}` is faithful or not. "
            f"I have concluded that the type is not faithful, so your code might run "
            f"with subpar performance. "
            f"Please open an issue at https://github.com/wesselb/plum.",
            stacklevel=2,
        )
        return False
