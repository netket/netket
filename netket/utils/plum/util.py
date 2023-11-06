import abc
import sys
from typing import List

if sys.version_info.minor <= 8:  # pragma: specific no cover 3.9 3.10 3.11
    from typing import Callable
else:  # pragma: specific no cover 3.8
    from collections.abc import Callable

__all__ = [
    "Callable",
    "TypeHint",
    "Missing",
    "multihash",
    "Comparable",
    "wrap_lambda",
    "is_in_class",
    "get_class",
    "get_context",
    "argsort",
]

# We use this to indicate a reader that we expect a type hint. Using just `object` as a
# type hint is technically correct for `int | None` for example, but does not convey the
# intention to a reader. Furthermore, if later on, Python has a proper type for type
# hints, we can just replace it here.
TypeHint = object


class _MissingType(type):
    """The type of :class:`Missing`."""

    def __bool__(self):
        # For some reason, Sphinx does attempt to evaluate `bool(Missing)`. Let's try
        # to keep Sphinx working correctly by not raising an exception.
        if "sphinx" in sys.modules:
            return False
        else:
            raise TypeError("`Missing` has no boolean value.")


class Missing(metaclass=_MissingType):
    """A class that can be used to indicate that a value is missing. This class cannot
    be instantiated and has no boolean value."""

    def __init__(self):
        raise TypeError("`Missing` cannot be instantiated.")


def multihash(*args):
    """Multi-argument order-sensitive hash.

    Args:
        *args: Objects to hash.

    Returns:
        int: Hash.
    """
    return hash(args)


class Comparable(metaclass=abc.ABCMeta):
    """A mixin that makes instances of the class comparable.

    Requires the subclass to just implement `__le__`.
    """

    def __eq__(self, other):
        return self <= other <= self

    def __ne__(self, other):
        return not self == other

    @abc.abstractmethod
    def __le__(self, other):
        pass  # pragma: no cover

    def __lt__(self, other):
        return self <= other and not other <= self

    def __ge__(self, other):
        return other.__le__(self)

    def __gt__(self, other):
        return self >= other and not other >= self

    def is_comparable(self, other):
        """Check whether this object is comparable with another one.

        Args:
            other (object): Object to check comparability with.

        Returns:
            Whether the object is comparable with `other`.
        """
        return self < other or self == other or self > other


def wrap_lambda(f: Callable) -> Callable:
    """Wrap a callable in a lambda function.

    Args:
        f (Callable): Function to wrap.

    Returns:
        function: Wrapped version of `f`.
    """
    return lambda x: f(x)


def is_in_class(f: Callable) -> bool:
    """Check if a function is part of a class.

    Args:
        f (function): Function to check.

    Returns:
        bool: Whether `f` is part of a class.
    """
    parts = f.__qualname__.split(".")
    return len(parts) >= 2 and parts[-2] != "<locals>"


def _split_parts(f: Callable) -> List[str]:
    qualified_name = f.__module__ + "." + f.__qualname__
    return qualified_name.split(".")


def get_class(f: Callable) -> str:
    """Assuming that `f` is part of a class, get the fully qualified name of the
    class.

    Args:
        f (function): Method to get class name for.

    Returns:
        str: Fully qualified name of class.
    """
    parts = _split_parts(f)
    return ".".join(parts[:-1])


def get_context(f) -> str:
    """Get the fully qualified name of the context for `f`.

    If `f` is part of a class, then the context corresponds to the scope of the class.
    If `f` is not part of a class, then the context corresponds to the scope of the
    function.

    Args:
        f (function): Method to get context for.

    Returns:
        str: The context of `f`.
    """
    parts = _split_parts(f)
    if is_in_class(f):
        # Split off function name and class.
        return ".".join(parts[:-2])
    else:
        # Split off function name only.
        return ".".join(parts[:-1])


def argsort(indexable):
    """
    Returns the indices used to index into a list in order
    to return the sorted list.
    """
    return sorted(range(len(indexable)), key=indexable.__getitem__)
