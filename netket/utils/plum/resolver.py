import pydoc
import sys
from functools import wraps
from typing import Callable, Optional, Tuple, Union

from rich.text import Text

from .method import Method, MethodList
from .repr import rich_repr
from .signature import Signature
from .util import argsort

__all__ = ["AmbiguousLookupError", "NotFoundLookupError"]


@rich_repr(str=True)
class AmbiguousLookupError(LookupError):
    """A signature cannot be resolved due to ambiguity."""

    def __init__(
        self,
        fname: Optional[str],
        target: Union[Tuple[object, ...], Signature],
        methods: "MethodList",
    ):
        """Create a new NotFoundLookupError.

        Args:
            fname (Optional[str]): name (or qualified name) of the function
                that could not be resolved.
            target (Union[Tuple[object, ...], Signature]): target signature
                or arguments that could not be resolved.
            methods (MethodList): list of ambiguous methods.
        """
        self.fname = fname if fname is not None else "<function>"
        self.target = target
        self.methods = methods

    def __rich_console__(self, console, options):
        yield Text(f"{self.fname}{self.target} is ambiguous.")
        yield Text()
        yield Text("Valid matches are:")
        for m in self.methods:
            args_ok = m.signature.compute_args_ok(self.target)
            yield m._repr_signature_mismatch(args_ok)


@rich_repr(str=True)
class NotFoundLookupError(LookupError):
    """A signature cannot be resolved because no applicable method can be found.

    This error object is used to display the closest methods to the target signature.
    """

    def __init__(
        self,
        fname: Optional[str],
        target: Union[Tuple[object, ...], Signature],
        methods: "MethodList",
        *,
        max_suggestions: int = 3,
    ):
        """Create a new NotFoundLookupError.

        Args:
            fname (Optional[str]): name (or qualified name) of the function
                that could not be resolved.
            target (Union[Tuple[object, ...], Signature]): target signature
                or arguments that could not be resolved.
            methods (MethodList): list of methods that were considered.
            max_suggestions (int, optional): Maximum number of displayed signatures.
                Defaults to 3.
        """
        self.fname = fname if fname is not None else "<function>"
        self.target = target
        self.methods = methods

        self.max_suggestions = max_suggestions

    def __rich_console__(self, console, options):
        """
        Generate a string of the top `max_suggestions` methods
        and signatures that are closest to the given one.
        """
        yield Text(f"{self.fname}{self.target} could not be resolved.")

        if not isinstance(self.target, Signature):
            distances = []
            for method in self.methods:
                dist = method.signature.compute_distance(self.target)
                distances.append(dist)

            sort_method_ids = argsort(distances)

            # Take at most 3 hints
            sort_method_ids = sort_method_ids[: self.max_suggestions]

            distances = [distances[i] for i in sort_method_ids]
            methods = [self.methods[i] for i in sort_method_ids]

            # create the error message
            yield Text("\nClosest candidates are:")
            for m in methods:
                args_ok = m.signature.compute_args_ok(self.target)
                yield m._repr_signature_mismatch(args_ok)


def _change_function_name(f: Callable, name: str) -> Callable:
    """It is not always the case that `f.__name__` is writable. To solve this, first
    create a temporary function that wraps `f` and then change the name.

    Args:
        f (function): Function to change the name of.
        name (str): New name.

    Returns:
        function: Function that wraps `f` and has name `name`.
    """

    @wraps(f)
    def f_renamed(*args, **kw_args):
        return f(*args, **kw_args)  # pragma: no cover

    f_renamed.__name__ = name
    return f_renamed


def _document(f: Callable, f_name: Optional[str] = None) -> str:
    """Generate documentation for a function `f`.

    The generated documentation contains both the function definition and the
    docstring. The docstring is on the same level of indentation of the function
    definition. There will be no trailing newlines.

    If the package :mod:`sphinx` is not imported, then the function definition will
    be preceded by the string `<separator>`.

    If the package :mod:`sphinx` is imported, then the function definition will include
    a Sphinx directive to displays the function definition in a nice way.

    Args:
        f (function): Function.
        f_name (str): An optional string representing the function name, which will be
            overridden from implementation's docstring which might have a different
            name. If this is not specified, the name is not overriden.

    Returns:
        str: Documentation for `f`.
    """
    # Ensure that the implementation has the right name, because this name
    # will show up in the docstring.
    if f_name is not None and getattr(f, "__name__", None) != f_name:
        f = _change_function_name(f, f_name)

    # :class:`pydoc._PlainTextDoc` removes styling. This styling will display
    # erroneously in Sphinx.
    parts = pydoc._PlainTextDoc().document(f).rstrip().split("\n")

    # Separate out the function definition and the lines corresponding to the body.
    title = parts[0]
    body = parts[1:]

    # Remove indentation from every line of the body. This indentation defaults to
    # four spaces.
    body = [line[4:] for line in body]

    # If `sphinx` is imported, assume that we're building the documentation. In that
    # case, display the function definition in a nice way.
    if "sphinx" in sys.modules:
        title = ".. py:function:: " + title + "\n   :noindex:"
    else:
        title = "<separator>\n\n" + title
    title += "\n"  # Add a newline to separate the title from the body.

    # Ensure that there are no trailing newlines. This can happen if the body is empty.
    return "\n".join([title] + body).rstrip()


class Resolver:
    """Method resolver.

    Attributes:
        methods (list[:class:`.method.Method`]): Registered methods.
        is_faithful (bool): Whether all methods are faithful or not.
    """

    __slots__ = ("methods", "is_faithful", "function_name")

    def __init__(self, function_name: Optional[str] = None):
        self.function_name = function_name
        self.methods: MethodList = MethodList()
        self.is_faithful: bool = True

    def doc(self, exclude: Union[Callable, None] = None) -> str:
        """Concatenate the docstrings of all methods of this function. Remove duplicate
        docstrings before concatenating.

        Args:
            exclude (function, optional): Exclude this implementation from the
                concatenation.

        Returns:
            str: Concatenation of all docstrings.
        """
        # Generate all docstrings, possibly excluding `exclude`.
        docs = [
            _document(m.implementation, self.function_name)
            for m in self.methods
            if not (exclude and m.implementation == exclude)
        ]
        # This can yield duplicates, because of extra methods automatically generated by
        # :func:`.signature.append_default_args`. We remove these by simply only
        # keeping unique docstrings.
        unique_docs = []
        for d in docs:
            if d not in unique_docs:
                unique_docs.append(d)
        # The unique documentations have no trailing newlines, so separate them with
        # a newline.
        return "\n\n".join(unique_docs)

    def register(self, method: Method) -> None:
        """Register a new method.

        Args:
            method (:class:`.method.Method`): Method to add.
        """
        signature = method.signature

        existing = [m.signature == signature for m in self.methods]
        if any(existing):
            if sum(existing) != 1:
                raise AssertionError(
                    f"The added method `{method}` is equal to {sum(existing)} "
                    f"existing methods. This should never happen."
                )
            self.methods[existing.index(True)] = method
        else:
            self.methods.append(method)

        # Use a double negation for slightly better performance.
        self.is_faithful = not any(not s.signature.is_faithful for s in self.methods)

    def __len__(self) -> int:
        return len(self.methods)

    def resolve(self, target: Union[Tuple[object, ...], Signature]) -> Method:
        """Find the most specific signature that satisfies a target.

        Args:
            target (:class:`.signature.Signature` or tuple[object]): Target to resolve.
                Must be either a signature to be encompassed or a tuple of arguments.

        Returns:
            :class:`.signature.Signature`: The most specific signature satisfying
                `target`.
        """
        if isinstance(target, tuple):

            def check(m):
                # `target` are concrete arguments.
                return m.signature.match(target)

        else:

            def check(m):
                # `target` is a signature that must be encompassed.
                return target <= m.signature

        candidates = []
        for method in [m for m in self.methods if check(m)]:
            # If none of the candidates are comparable, then add the method as
            # a new candidate and continue.
            if not any(c.signature.is_comparable(method.signature) for c in candidates):
                candidates += [method]
                continue

            # The signature under consideration is comparable with at least one
            # of the candidates. First, filter any strictly more general candidates.
            new_candidates = [
                c for c in candidates if not method.signature < c.signature
            ]

            # If the signature under consideration is as specific as at least
            # one candidate, then and only then add it as a candidate.
            if any(method.signature <= c.signature for c in candidates):
                candidates = new_candidates + [method]
            else:
                candidates = new_candidates

        if len(candidates) == 0:
            # There is no matching signature.
            raise NotFoundLookupError(self.function_name, target, self.methods)

        elif len(candidates) == 1:
            # There is exactly one matching signature. Success!
            return candidates[0]
        else:
            # There are multiple matching signatures. Before raising an exception,
            # attempt to resolve the ambiguity using the precedence of the signatures.
            precedences = [c.signature.precedence for c in candidates]
            max_precendence = max(precedences)
            if sum([p == max_precendence for p in precedences]) == 1:
                return candidates[precedences.index(max_precendence)]
            else:
                # Could not resolve the ambiguity, so error.
                raise AmbiguousLookupError(self.function_name, target, candidates)
