from beartype.door import TypeHint

from . import _is_bearable, function
from .dispatcher import Dispatcher
from .repr import repr_short
from .type import resolve_type_hint

__all__ = [
    "convert",
    "add_conversion_method",
    "conversion_method",
    "add_promotion_rule",
    "promote",
]

_dispatch = Dispatcher()


@_dispatch
def convert(obj, type_to):
    """Convert an object to a particular type.

    Args:
        obj (object): Object to convert.
        type_to (type): Type to convert to.

    Returns:
        object: `obj` converted to type `type_to`.
    """
    type_to = resolve_type_hint(type_to)
    # TODO: Can we implement this without using `type`?!
    return _convert.invoke(type(obj), type_to)(obj, type_to)


# Deliver `convert`.
function._promised_convert = convert


@_dispatch
def _convert(obj, type_to):
    if _is_bearable(obj, resolve_type_hint(type_to)):
        return obj
    else:
        raise TypeError(f"Cannot convert `{obj}` to `{repr_short(type_to)}`.")


def add_conversion_method(type_from, type_to, f):
    """Add a conversion method to convert an object from one type to another.

    Args:
        type_from (type): Type to convert from.
        type_to (type): Type to convert to.
        f (function): Function that converts an object of type `type_from` to
            type `type_to`.
    """

    @_convert.dispatch
    def perform_conversion(obj: type_from, _: type_to):
        return f(obj)


def conversion_method(type_from, type_to):
    """Decorator to add a conversion method to convert an object from one
    type to another.

    Args:
        type_from (type): Type to convert from.
        type_to (type): Type to convert to.
    """

    def add_method(f):
        add_conversion_method(type_from, type_to, f)

    return add_method


# Add some common conversion methods.
add_conversion_method(object, tuple, lambda x: (x,))
add_conversion_method(tuple, tuple, lambda x: x)
add_conversion_method(list, tuple, tuple)
add_conversion_method(object, list, lambda x: [x])
add_conversion_method(list, list, lambda x: x)
add_conversion_method(tuple, list, list)
add_conversion_method(bytes, str, lambda x: x.decode("utf-8", "replace"))


@_dispatch
def _promotion_rule(type1, type2):
    """Promotion rule.

    Args:
        type1 (type): First type to promote.
        type2 (type): Second type to promote.

    Returns:
        type: Type to convert to.
    """
    type1 = resolve_type_hint(type1)
    type2 = resolve_type_hint(type2)
    if TypeHint(type1) <= TypeHint(type2):
        return type2
    elif TypeHint(type2) <= TypeHint(type1):
        return type1
    else:
        raise TypeError(
            f"No promotion rule for `{repr_short(type1)}` and `{repr_short(type2)}`."
        )


@_dispatch
def add_promotion_rule(type1, type2, type_to):
    """Add a promotion rule.

    Args:
        type1 (type): First type to promote.
        type2 (type): Second type to promote.
        type_to (type): Type to convert to.
    """

    @_promotion_rule.dispatch
    def rule(t1: type1, t2: type2):
        return type_to

    # If the types are the same, the method will get overwritten.

    @_promotion_rule.dispatch
    def rule(t1: type2, t2: type1):  # noqa: F811
        return type_to


@_dispatch
def promote(obj1, obj2, *objs):
    """Promote objects to a common type.

    Args:
        \\*objs (object): Objects to convert.

    Returns:
        tuple: `objs`, but all converted to a common type.
    """
    # Convert to a single tuple.
    objs = (obj1, obj2) + objs

    # Get the types of the objects.
    # TODO: Can we implement this without calling `type`?!
    types = [type(obj) for obj in objs]

    def _promote_types(t0, t1):
        return resolve_type_hint(_promotion_rule.invoke(t0, t1)(t0, t1))

    # Find the common type.
    _promotion_rule._resolve_pending_registrations()
    common_type = _promote_types(types[0], types[1])
    for t in types[2:]:
        common_type = _promote_types(common_type, t)

    # Convert objects and return.
    return tuple(convert(obj, common_type) for obj in objs)


@_dispatch
def promote(obj: object):  # noqa: F811
    # Promote should always return a tuple to avoid edge cases.
    return (obj,)


@_dispatch
def promote():  # noqa: F811
    return ()
