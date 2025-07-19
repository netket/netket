def to_iterable(maybe_iterable):
    """
    to_iterable(maybe_iterable)

    Ensure the result is iterable. If the input is not iterable, it is wrapped into a tuple.
    """
    if hasattr(maybe_iterable, "__iter__"):
        surely_iterable = maybe_iterable
    else:
        surely_iterable = (maybe_iterable,)

    return surely_iterable
