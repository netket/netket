import inspect
import os
import sys
import types
import typing
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional

import rich
from rich.color import Color
from rich.style import Style
from rich.text import Text

__all__ = [
    "repr_short",
    "repr_type",
    "repr_source_path",
    "repr_pyfunction",
    "rich_repr",
]

path_style = Style(color=Color.from_ansi(7))
file_style = Style(bold=True, underline=True)

module_style = Style(color="grey66")
class_style = Style(bold=True)


def repr_type(typ) -> Text:
    """Returns a {class}`rich.Text` representation of a type.

    Does some light syntax highlighting mimicking Julia, boldening
    class names and coloring module names with a lighter color.

    Args:
        typ: A type or arbitrary object.
    """
    # TODO: remove version check when dropping support for Python 3.8
    if sys.version_info.minor > 8 and isinstance(typ, types.GenericAlias):
        return Text(repr(typ), style=class_style)

    if isinstance(typ, type):
        if typ.__module__ in ["builtins", "typing", "typing_extensions"]:
            return Text(typ.__qualname__, style=class_style)
        else:
            return Text(f"{typ.__module__}.", style=module_style) + Text(
                typ.__qualname__, style=class_style
            )
    if isinstance(typ, types.FunctionType):
        return Text(typ.__name__, style=module_style)

    return Text(repr(typ), style=class_style)


def repr_short(x) -> str:
    """Representation as a string, but in shorter form. This just calls
    :func:`typing._type_repr`.

    Args:
        x (object): Object.

    Returns:
        str: Shorter representation of `x`.
    """
    # :func:`typing._type_repr` is an internal function, but it should be available in
    # Python versions 3.8 through 3.11.
    return typing._type_repr(x)


def repr_source_path(function: Callable) -> Text:
    """Returns a {class}`rich.Text` object with an hyperlink
    to the function definition.

    Args:
        function: A callable

    Returns:
        A {class}`rich.Text` object with an hyperlink. If
        it fails to introspect returns an empty string.
    """
    try:
        fpath = inspect.getfile(function)
        fline = inspect.getsourcelines(function)[1]
        uri = f"file://{fpath}#{fline}"

        # compress the path
        home_path = os.path.expanduser("~")
        fpath = fpath.replace(home_path, "~")

        # underline file name
        fname = os.path.basename(fpath)
        if fname.endswith(".py"):
            fpath = (
                Text(os.path.dirname(fpath), style=path_style)
                + Text("/")
                + Text(fname, style=file_style)
            )
        else:
            fpath = Text(fpath, style=path_style)
        fpath.append_text(Text(f":{fline}"))
        fpath.stylize(f"link {uri}")
    except OSError:  # pragma: no cover
        fpath = Text()
    return fpath


def repr_pyfunction(function: Callable) -> Text:
    """Returns a {class}`rich.Text` object representing
    a function.

    Args:
        function: A callable

    Returns:
        A {class}`rich.Text` object with an hyperlink. If
        it fails to introspect returns an empty string.
    """
    res = Text(repr(function))
    res.append(" @ ")
    res.append_text(repr_source_path(function))
    return res


########################
# Rich class decorator #
########################


def __repr_from_rich__(self) -> str:
    """
    default __repr__ that calls __rich__
    """
    # print("calling __repr_from_rich__")
    console = rich.get_console()
    with console.capture() as capture:
        console.print(self, end="")
    res = capture.get()
    # print("got ", res)
    return res


def _repr_mimebundle_from_rich_(
    self, include: Iterable[str], exclude: Iterable[str], **kwargs: Any
) -> Dict[str, str]:
    from rich.jupyter import _render_segments

    console = rich.get_console()
    segments = list(console.render(self, console.options))  # type: ignore
    html = _render_segments(segments)
    text = console._render_buffer(segments)
    data = {"text/plain": text, "text/html": html}
    if include:
        data = {k: v for (k, v) in data.items() if k in include}
    if exclude:
        data = {k: v for (k, v) in data.items() if k not in exclude}
    return data


def rich_repr(clz: Optional[type] = None, str: bool = False):
    """
    Class decorator defining a __repr__ method that calls __rich__.

    This also sets `_repr_mimebundle_` for better rendering in
    jupyter.

    Args:
        clz: Class to decorate. If None, returns a decorator.
        str: If True, also defines __str__.

    Returns:
        The decorated class. If clz is None, returns a decorator.
    """
    if clz is None:
        return partial(rich_repr, str=str)
    clz.__repr__ = __repr_from_rich__
    clz._repr_mimebundle_ = _repr_mimebundle_from_rich_
    if str:
        clz.__str__ = __repr_from_rich__
    return clz
