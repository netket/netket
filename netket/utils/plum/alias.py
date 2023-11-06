"""This module monkey patches `__repr__` and `__str__` of `typing.Union` to control how
`typing.Unions` are displayed.

Example::

    >> plum.activate_union_aliases()

    >> IntOrFloat = typing.Union[int, float]

    >> IntOrFloat
    Union[int, float]

    >> plum.set_union_alias(IntOrFloat, "IntOrFloat")

    >> IntOrFloat
    typing.Union[IntOrFloat]

    >> typing.Union[int, float]
    typing.Union[IntOrFloat]

    >> typing.Union[int, float, str]
    typing.Union[IntOrFloat, str]

Note that `IntOrFloat` prints to `typing.Union[IntOrFloat]` rather than just
`IntOrFloat`. This is deliberate, with the goal of not breaking code that relies on
parsing how unions print.
"""

import typing
from functools import wraps
from typing import get_args

__all__ = ["activate_union_aliases", "deactivate_union_aliases", "set_union_alias"]

_union_type = type(typing.Union[int, float])
_original_repr = _union_type.__repr__
_original_str = _union_type.__str__


@wraps(_original_repr)
def _new_repr(self):
    """Print a `typing.Union`, replacing all aliased unions by their aliased names.

    Returns:
        str: Representation of a `typing.Union` taking into account union aliases.
    """
    args = get_args(self)
    args_set = set(args)

    # Find all aliased unions contained in this union.
    found_unions = []
    found_positions = []
    found_aliases = []
    for union, alias in reversed(_aliased_unions):
        union_set = set(union)
        if union_set <= args_set:
            found = False
            for i, arg in enumerate(args):
                if arg in union_set:
                    found_unions.append(union_set)
                    found_positions.append(i)
                    found_aliases.append(alias)
                    found = True
                    break
            if not found:  # pragma: no cover
                # This branch should never be reached.
                raise AssertionError(
                    "Could not identify union. This should never happen."
                )

    # Delete any unions that are contained in strictly bigger unions. We check for
    # strictly inequality because any union includes itself.
    for i in range(len(found_unions) - 1, -1, -1):
        for union in found_unions:
            if found_unions[i] < union:
                del found_unions[i]
                del found_positions[i]
                del found_aliases[i]
                break

    # Create a set with all arguments of all found unions.
    found_args = set()
    for union in found_unions:
        found_args |= union

    # Insert the aliases right before the first found argument. When we insert an
    # element, the positions of following insertions need to be appropriately
    # incremented.
    args = list(args)
    delta = 0
    # Sort by insertion position to ensure that all following insertions are at higher
    # indices. This makes the bookkeeping simple.
    for i, alias in sorted(zip(found_positions, found_aliases), key=lambda x: x[0]):
        args.insert(i + delta, alias)
        delta += 1

    # Filter all elements of unions that are aliased.
    new_args = ()
    for arg in args:
        if arg not in found_args:
            new_args += (arg,)
    args = new_args

    # Generate a string representation.
    args_repr = [a if isinstance(a, str) else typing._type_repr(a) for a in args]
    # Like `typing` does, print `Optional` whenever possible.
    if len(args) == 2:
        if args[0] is type(None):  # noqa: E721
            return f"typing.Optional[{args_repr[1]}]"
        elif args[1] is type(None):  # noqa: E721
            return f"typing.Optional[{args_repr[0]}]"
    # We would like to just print `args_repr[0]` whenever `len(args) == 1`, but
    # this might break code that parses how unions print.
    return "typing.Union[" + ", ".join(args_repr) + "]"


@wraps(_original_str)
def _new_str(self):
    """Does the same as :func:`_new_repr`.

    Returns:
        str: Representation of the `typing.Union` taking into account union aliases.
    """
    return _new_repr(self)


def activate_union_aliases():
    """When printing `typing.Union`s, replace all aliased unions by the aliased names.
    This monkey patches `__repr__` and `__str__` for `typing.Union`."""
    _union_type.__repr__ = _new_repr
    _union_type.__str__ = _new_str


def deactivate_union_aliases():
    """Undo what :func:`.alias.activate` did. This restores the original  `__repr__`
    and `__str__` for `typing.Union`."""
    _union_type.__repr__ = _original_repr
    _union_type.__str__ = _original_str


_aliased_unions = []


def set_union_alias(union, alias):
    """Change how a `typing.Union` is printed. This does not modify `union`.

    Args:
        union (type or type hint): A union.
        alias (str): How to print `union`.

    Returns:
        type or type hint: `union`.
    """
    if not isinstance(union, _union_type):
        args = (union,)
    else:
        args = get_args(union)
    for existing_union, existing_alias in _aliased_unions:
        if set(existing_union) == set(args) and alias != existing_alias:
            if isinstance(union, _union_type):
                union_str = _original_str(union)
            else:
                union_str = repr(union)
            raise RuntimeError(f"`{union_str}` already has alias `{existing_alias}`.")
    _aliased_unions.append((args, alias))
    return union
