import sys

if sys.version_info >= (3, 11):  # pragma: specific no cover 3.7 3.8 3.9 3.10
    from typing import get_overloads, overload
else:  # pragma: specific no cover 3.11
    from typing_extensions import get_overloads, overload


__all__ = ["overload", "get_overloads"]
