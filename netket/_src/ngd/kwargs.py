import inspect

from collections.abc import Callable
from functools import partial


def ensure_accepts_kwargs(f: Callable, kwarg_name: str):
    """
    Add a keyword argument to a function if it is not already present.

    This is useful to make a function compatible with a certain interface, even if it does not
    need the additional argument.

    In this case, we use it to make the linear solver functions compatible with the QGT interface.
    Some QGT solvers require the vector of local energies as an additional argument. The standard
    solver interface however only requires the matrix and the right-hand side (jacobian @ local_energies).

    Args:
        f: The function to concretize.
        kwarg_name: The name of the keyword argument to add.

    Returns:
        A new function that accepts the keyword argument `kwarg_name`.
    """
    sig = inspect.signature(f)
    params = sig.parameters

    if kwarg_name in params:
        return f
    else:
        to_concretize = f
        if isinstance(f, partial):
            to_concretize = f.func

        def concretized_func(*args, **kwargs):
            kwargs.pop(kwarg_name, None)
            return to_concretize(*args, **kwargs)

        if isinstance(f, partial):
            concretized_func = partial(concretized_func, *f.args, **f.keywords)

        return concretized_func
