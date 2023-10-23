import inspect
import typing
from typing import Callable, List, Optional

from rich.segment import Segment
from rich.text import Text

from .repr import repr_pyfunction, repr_type, rich_repr
from .signature import Signature, inspect_signature
from .type import resolve_type_hint
from .util import TypeHint

__all__ = ["Method", "extract_return_type"]


@rich_repr
class Method:
    """Method.

    Attributes:
        return_type (type): Return type.
        implementation (function or None): Implementation.
    """

    _default_return_type = typing.Any

    __slots__ = ("function_name", "implementation", "signature", "return_type")

    def __init__(
        self,
        implementation: Callable,
        signature: Signature,
        *,
        function_name: Optional[str] = None,
        return_type: Optional[TypeHint] = None,
    ):
        """Instantiate a method.

        Args:
            implementation (function): Callable implementing the method.
            signature (:class:`Signature`): Signature of the method.
            return_type (type, optional): Return type of the method. Can be left
                unspecified, in which case the correct type will be deduced from the
                signature.
            return_type (type, optional): Type of the return value. Defaults to
                :obj:`Any`.
        """
        if return_type is None:
            return_type = extract_return_type(implementation)
        if function_name is None:
            function_name = implementation.__name__

        self.implementation = implementation
        self.signature = signature
        self.function_name = function_name
        self.return_type = return_type

    def __copy__(self):
        return Method(
            self.implementation,
            self.signature,
            function_name=self.function_name,
            return_type=self.return_type,
        )

    def __eq__(self, other):
        if isinstance(other, Method):
            return (
                self.function_name,
                self.implementation,
                self.signature,
                self.return_type,
            ) == (
                other.function_name,
                other.implementation,
                other.signature,
                other.return_type,
            )
        return False

    def __hash__(self):
        s = (self.function_name, self.implementation, self.signature, self.return_type)
        return hash(s)

    def __str__(self):
        function_name = self.function_name
        signature = self.signature
        return_type = self.return_type
        impl = self.implementation
        return f"Method({function_name=}, {signature=}, {return_type=}, {impl=})"

    def __rich_console__(self, console, options):
        argnames, kwnames, kwvar_name = extract_argnames(self.implementation)

        sig = self.signature
        parts = []
        if sig.types:
            for nm, t in zip(argnames, sig.types):
                parts.append(Text(f"{nm}: ") + repr_type(t))
        if sig.varargs != Signature._default_varargs:
            parts.append(Text(f"*{argnames[-1]}: ") + repr_type(sig.varargs))

        if len(kwnames) > 0 or kwvar_name is not None:
            parts.append(Text("*"))
        for kwnm in kwnames:
            parts.append(Text(f"{kwnm}"))
        if kwvar_name is not None:
            parts.append(Text(f"**{kwvar_name}"))

        res = Text(self.function_name) + Text("(") + Text(", ").join(parts) + Text(")")
        if self.return_type != Method._default_return_type:
            res.append(" -> ")
            res.append(repr_type(self.return_type))
        if sig.precedence != Signature._default_precedence:
            res.append(f"\n    precedence={sig.precedence}")

        yield res
        yield Segment("    ")
        yield repr_pyfunction(self.implementation)

    def _repr_signature_mismatch(self, args_ok: List[bool]) -> str:
        """Special version of __repr__ but used when
        printing args mismatch (mainly in hints to possible
        similar signatures).

        Args:
            args_ok: a list of which arguments match this signature
                and which don't according to the resolver.
        """
        sig = self.signature

        argnames, kwnames, kwvar_name = extract_argnames(self.implementation)
        varargs_ok = all(args_ok[len(sig.types) :])

        parts = []
        if sig.types:
            for i, (nm, t) in enumerate(zip(argnames, sig.types)):
                is_ok = args_ok[i] if i < len(args_ok) else False
                arg_txt = Text(f"{nm}: ")
                type_txt = repr_type(t)
                if not is_ok:
                    type_txt.stylize("red")
                arg_txt.append(type_txt)
                parts.append(arg_txt)
        if sig.varargs != Signature._default_varargs:
            arg_txt = Text(f"*{argnames[-1]}: ")
            type_txt = repr_type(sig.varargs)
            if not varargs_ok:
                type_txt.stylize("red")
            arg_txt.append(type_txt)
            parts.append(arg_txt)

        if len(kwnames) > 0 or kwvar_name is not None:
            parts.append(Text("*"))
        for kwnm in kwnames:
            parts.append(Text(f"{kwnm}"))
        if kwvar_name is not None:
            parts.append(Text(f"**{kwvar_name}"))

        res = Text(self.function_name) + Text("(") + Text(", ").join(parts) + Text(")")
        if self.return_type != Method._default_return_type:
            res.append(" -> ")
            res.append(repr_type(self.return_type))
        if sig.precedence != Signature._default_precedence:
            res.append(f"\n    precedence={sig.precedence}")

        res.append("\n    ")
        res.append_text(repr_pyfunction(self.implementation))
        return res


@rich_repr
class MethodList(list):
    def __rich_console__(self, console, options):
        yield f"Method List with # {len(self)} methods:"
        for i, method in enumerate(self):
            yield Segment(f" [{i}] ")
            yield method


def extract_argnames(f: Callable, precedence: int = 0) -> Signature:
    """Extract the signature from a function.

    Args:
        f (function): Function to extract signature from.
        precedence (int, optional): Precedence of the method.

    Returns:
        :class:`.Signature`: Signature.
    """
    # Extract specification.
    sig = inspect_signature(f)

    # Get types of arguments.
    argnames = []
    kwnames = []
    kwvar_name = None
    for arg in sig.parameters:
        p = sig.parameters[arg]

        if p.kind in {p.KEYWORD_ONLY}:
            kwnames.append(p.name)
        elif p.kind in {p.VAR_KEYWORD}:
            kwvar_name = p.name
        else:
            argnames.append(p.name)

    return argnames, kwnames, kwvar_name


def extract_return_type(f: Callable) -> TypeHint:
    """Extract the return type from a function.

    Assumes that PEP563-style already have been resolved.

    Args:
        f (function): Function to extract return type from.

    Returns:
        :class:`TypeHint`: Return type annotation
    """

    # Extract specification.
    sig = inspect_signature(f)

    # Get possible return type.
    if sig.return_annotation is inspect.Parameter.empty:
        return_type = typing.Any
    else:
        return_type = resolve_type_hint(sig.return_annotation)

    return return_type
