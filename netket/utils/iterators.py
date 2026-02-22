from typing import TypeVar
from collections.abc import Iterable

T = TypeVar("T")


def to_iterable(
    maybe_iterable: T | Iterable[T], none_is_empty: bool = False
) -> tuple[T, ...]:
    """
    Ensure the result is iterable. If the input is not iterable, it is wrapped into a tuple.

    Args:
        maybe_iterable: An object that may or may not be iterable.
        none_is_empty: If True, None is treated as an empty iterable. If False, None is treated as a non-iterable object and wrapped into a tuple.

    Returns:
        An iterable containing the input object, or the input object itself if it is already iterable.
    """
    if none_is_empty and maybe_iterable is None:
        return tuple()

    if not isinstance(maybe_iterable, Iterable):
        maybe_iterable = (maybe_iterable,)

    return tuple(maybe_iterable)


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))
