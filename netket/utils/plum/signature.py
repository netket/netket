import inspect
import operator
import typing
from copy import copy
from typing import Callable, List, Tuple, Union

import beartype.door
from beartype.peps import resolve_pep563 as beartype_resolve_pep563
from rich.segment import Segment

from . import _is_bearable
from .repr import repr_short, rich_repr
from .type import is_faithful, resolve_type_hint
from .util import Comparable, Missing, TypeHint, multihash, wrap_lambda

__all__ = ["Signature", "append_default_args"]

OptionalType = Union[TypeHint, type(Missing)]


@rich_repr
class Signature(Comparable):
    """Object representing a call signature that may be used to dispatch a function
    call.

    This object differs structurally from the return value of :func:`inspect.signature`
    as it only contains information necessary for performing dispatch.

    For example, for the current implementation of Plum, which does not dispatch on
    keyword arguments, those are left out of this signature object. Similarly, return
    type information and argument names are not present.

    Attributes:
        types (tuple[:obj:`.TypeHint`, ...]): Types of the call signature.
        varargs (type or :class:`.util.Missing`): Type of the variable number of
            arguments.
        has_varargs (bool): Whether `varargs` is not :class:`.util.Missing`.
        precedence (int): Precedence.
        is_faithful (bool): Whether this signature only uses faithful types.
    """

    _default_varargs = Missing
    _default_precedence = 0

    __slots__ = ("types", "varargs", "precedence", "is_faithful")

    def __init__(
        self,
        *types: Tuple[TypeHint, ...],
        varargs: OptionalType = _default_varargs,
        precedence: int = _default_precedence,
    ):
        """Instantiate a signature, which contains exactly the information necessary for
        dispatch.

        Args:
            *types (:obj:`.TypeHint`): Types of the arguments.
            varargs (:obj:`.TypeHint`, optional): Type of the variable arguments.
            precedence (int, optional): Precedence. Defaults to `0`.
        """
        self.types = types
        self.varargs = varargs
        self.precedence = precedence

        types_are_faithful = all(is_faithful(t) for t in types)
        varargs_are_faithful = self.varargs is Missing or is_faithful(self.varargs)
        self.is_faithful = types_are_faithful and varargs_are_faithful

    @staticmethod
    def from_callable(f: Callable, precedence: int = 0) -> "Signature":
        """Construct a signature from a callable.

        Args:
            f (Callable): Callable.
            precedence (int, optional): Precedence. Defaults to 0.

        Returns:
            :class:`Signature`: Signature for `f`.
        """
        types, varargs = _extract_signature(f)
        return Signature(
            *types,
            varargs=varargs,
            precedence=precedence,
        )

    @property
    def has_varargs(self) -> bool:
        return self.varargs is not Missing

    def __copy__(self):
        cls = type(self)
        copy = cls.__new__(cls)

        copy.types = self.types
        copy.varargs = self.varargs
        copy.precedence = self.precedence
        copy.is_faithful = self.is_faithful
        return copy

    def __rich_console__(self, console, options):
        yield Segment("Signature(")
        show_comma = True
        if self.types:
            yield Segment(", ".join(map(repr_short, self.types)))
        if self.varargs != Signature._default_varargs:
            if show_comma:
                yield Segment(", ")
            yield Segment("varargs=" + repr_short(self.varargs))
        if self.precedence != Signature._default_precedence:
            if show_comma:
                yield Segment(", ")
            yield Segment("precedence=" + repr(self.precedence))
        yield Segment(")")

    def __eq__(self, other):
        if isinstance(other, Signature):
            return (
                self.types,
                self.varargs,
                self.precedence,
                self.is_faithful,
            ) == (
                other.types,
                other.varargs,
                other.precedence,
                other.is_faithful,
            )
        return False

    def __hash__(self):
        return multihash(Signature, *self.types, self.varargs)

    def expand_varargs(self, n: int) -> Tuple[TypeHint, ...]:
        """Expand variable arguments.

        Args:
            n (int): Desired number of types.

        Returns:
            tuple[type, ...]: Expanded types.
        """
        if self.has_varargs:
            expansion_size = max(n - len(self.types), 0)
            return self.types + (self.varargs,) * expansion_size
        else:
            return self.types

    def __le__(self, other) -> bool:
        """Checks if signature self is more specific of signature other.

        The logic of this method is based upon the PhD thesis of Jeff Bezanson.
        https://github.com/JeffBezanson/phdthesis/blob/master/main.pdf

        The relevant section is chapter 4.3, and the list of rules are found at
        Sec 4.3.2
        """

        # If both signatures have varargs, we interpret both varargs as set of
        # signatures, and we verify that at least 1 element in the set of A is
        # more specific than an element in the set of B, but that no element
        # in the set of B is more specific than the set of A.
        #
        # This implements Rule #3 for variadic elements of Sec 4.3.2
        if self.has_varargs and other.has_varargs:
            if len(self.types) == len(other.types):
                _self = Signature(*self.types)
                _other = Signature(*other.types)
            else:
                max_len = max(len(self.types), len(other.types))
                _self = Signature(*self.expand_varargs(max_len))
                _other = Signature(*other.expand_varargs(max_len))

            # If an element in set [[self]] is more specific than the smallest
            # element in set [[other]]
            if _self <= _other:
                # Check that no element of set [[other]] is more specific than
                # an element of set [[self]]
                varargs_comparison = beartype.door.TypeHint(
                    other.varargs
                ) < beartype.door.TypeHint(self.varargs)
                return not varargs_comparison
            else:
                # if no element in set [[self]] is more specific than set [[other]],
                # then self is not more specific than other
                return False

        # If the number of types of the signatures are unequal, then the signature
        # with the fewer number of types must be expanded using variable arguments.
        if not (
            len(self.types) == len(other.types)
            or (len(self.types) > len(other.types) and other.has_varargs)
            or (len(self.types) < len(other.types) and self.has_varargs)
        ):
            return False

        # Finally, expand the types and compare.
        self_types = self.expand_varargs(len(other.types))
        other_types = other.expand_varargs(len(self.types))
        is_more_specific = all(
            [
                beartype.door.TypeHint(x) <= beartype.door.TypeHint(y)
                for x, y in zip(self_types, other_types)
            ]
        )

        # If there are no varargs, we could just return `is_more_specific`, but if
        # there are varargs, the rules are more complex. In particular, this must
        # implement Rule #4 of Sec 4.3.2
        # (A vararg type is less speciﬁc than an otherwise equal non-vararg type.)
        if is_more_specific:
            # We are more specific, but equality might mean that one of the two
            # is a vararg, therefore smaller
            if self.has_varargs ^ other.has_varargs:
                # If only one signature has a vararg, check if the two signatures
                # are really equivalent
                equivalent = all(
                    [
                        beartype.door.TypeHint(x) == beartype.door.TypeHint(y)
                        for x, y in zip(self_types, other_types)
                    ]
                )
                if equivalent and self.has_varargs:
                    # Rule #4: the smallest is the one without varargs
                    return False
                else:
                    return True
            else:
                # None has varargs (both vararg case was already handled above) so
                # just return True
                return True
        else:
            # if we are not more specific, varargs don't matter
            return False

    def match(self, values) -> bool:
        """Check whether values match the signature.

        Args:
            values (tuple): Values.

        Returns:
            bool: `True` if `values` match this signature and `False` otherwise.
        """
        # `values` must either be exactly many as `self.types`. If there are more
        # `values`, then there must be variable arguments to cover the arguments.
        if not (
            len(self.types) == len(values)
            or (len(self.types) < len(values) and self.has_varargs)
        ):
            return False
        else:
            types = self.expand_varargs(len(values))
            return all(_is_bearable(v, t) for v, t in zip(values, types))

    def compute_distance(self, values: Tuple[object, ...]) -> int:
        """Computes the edit distance between this
        Signature and the
        """
        types = self.expand_varargs(len(values))
        # vararg_types = types[len(self.types):]

        distance = 0

        # count 1 for every extra or missingargument
        distance += abs(len(types) - len(values))

        # count 1 for every mismatching arg type
        for v, t in zip(values, types):
            if not _is_bearable(v, t):
                distance += 1

        return distance

    def compute_args_ok(self, values) -> List[bool]:
        types = self.expand_varargs(len(values))

        args_ok = []

        # count 1 for every mismatching arg type
        for v, t in zip(values, types):
            args_ok.append(_is_bearable(v, t))

        # all extra args are not ok
        for _ in range(len(args_ok), len(values)):
            args_ok.append(False)

        return args_ok


def inspect_signature(f) -> inspect.Signature:
    """Wrapper of :func:`inspect.signature` which adds support for certain non-function
    objects.

    Args:
        f (object): Function-like object.

    Returns:
        object: Signature.
    """
    if isinstance(f, operator.itemgetter):
        f = wrap_lambda(f)
    elif isinstance(f, operator.attrgetter):
        f = wrap_lambda(f)
    return inspect.signature(f)


def resolve_pep563(f: Callable):
    """Utility function to resolve PEP563-style annotations and make editable.

    This function mutates `f`.

    Args:
        f (Callable): Function whose annotations should be resolved.
    """
    if hasattr(f, "__annotations__"):
        beartype_resolve_pep563(f)  # This mutates `f`.
        # Override the `__annotations__` attribute, since `resolve_pep563` modifies
        # `f` too.
        for k, v in typing.get_type_hints(f).items():
            f.__annotations__[k] = v


def _extract_signature(f: Callable, precedence: int = 0) -> Signature:
    """Extract the signature from a function.

    Args:
        f (function): Function to extract signature from.
        precedence (int, optional): Precedence of the method.

    Returns:
        :class:`.Signature`: Signature.
    """
    resolve_pep563(f)

    # Extract specification.
    sig = inspect_signature(f)

    # Get types of arguments.
    types = []
    varargs = Missing
    for arg in sig.parameters:
        p = sig.parameters[arg]

        # Parse and resolve annotation.
        if p.annotation is inspect.Parameter.empty:
            annotation = typing.Any
        else:
            annotation = resolve_type_hint(p.annotation)

        # Stop once we have seen all positional parameter without a default value.
        if p.kind in {p.KEYWORD_ONLY, p.VAR_KEYWORD}:
            break

        if p.kind == p.VAR_POSITIONAL:
            # Parameter indicates variable arguments.
            varargs = annotation
        else:
            # Parameter is a regular positional parameter.
            types.append(annotation)

        # If there is a default parameter, make sure that it is of the annotated type.
        if p.default is not inspect.Parameter.empty:
            if not _is_bearable(p.default, annotation):
                raise TypeError(
                    f"Default value `{p.default}` is not an instance "
                    f"of the annotated type `{repr_short(annotation)}`."
                )

    return types, varargs


def append_default_args(signature: Signature, f: Callable) -> List[Signature]:
    """Returns a list of signatures of function `f`, where those signatures are derived
    from the input arguments of `f` by treating every non-keyword-only argument with a
    default value as a keyword-only argument turn by turn.

    Args:
        f (function): Function to extract default arguments from.
        signature (:class:`.signature.Signature`): Signature of `f` from which to
            remove default arguments.

    Returns:
        list[:class:`.signature.Signature`]: list of signatures excluding from 0 to all
        default arguments.
    """
    # Extract specification.
    f_signature = inspect_signature(f)

    signatures = [signature]

    arg_names = list(f_signature.parameters.keys())
    # We start at the end and, once we reach non-keyword-only arguments, delete the
    # argument with defaults values one by one. This generates a sequence of signatures,
    # which we return.
    arg_names.reverse()

    for arg in arg_names:
        p = f_signature.parameters[arg]

        # Ignore variable arguments and keyword arguments.
        if p.kind in {p.VAR_KEYWORD, p.KEYWORD_ONLY}:
            continue

        # Stop when non-variable arguments without a default are reached.
        if p.kind != p.VAR_POSITIONAL and p.default is inspect.Parameter.empty:
            break

        # Skip variable arguments. These will always be removed.
        if p.kind == p.VAR_POSITIONAL:
            continue

        signature_copy = copy(signatures[-1])

        # As specified over, these additional signatures should never have variable
        # arguments.
        signature_copy.varargs = Missing

        # Remove the last positional argument.
        signature_copy.types = signature_copy.types[:-1]

        signatures.append(signature_copy)

    return signatures
