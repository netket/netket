# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import functools
import inspect


_KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD


def partial_from_kwargs(
    func: Callable = None, *, exclusive_arg_names: tuple[tuple[str, ...], ...] = None
):
    """
    Wraps the decorated function such that if only the keyword arguments are specified
    a partial object wrapping the specified keyword arguments is returned.

    Args:
        fun: a function with keyword only arguments.
        exclusive_arg_names: a list of tuples of strings, where each tuple contains the names of
            arguments that are mutually exclusive. If any of the arguments in the tuple are specified
            together, an error is raised.
    """
    if func is None:
        return functools.partial(
            partial_from_kwargs, exclusive_arg_names=exclusive_arg_names
        )

    # Get the functions's Keyword only arguments and keyword maybe arguments
    sig = inspect.signature(func)
    kwargs_only = [
        par.name for par in sig.parameters.values() if par.kind == _KEYWORD_ONLY
    ]
    maybe_kwargs = [
        par.name
        for par in sig.parameters.values()
        if par.kind == _POSITIONAL_OR_KEYWORD
    ]
    has_varkw = any(par.kind == _VAR_KEYWORD for par in sig.parameters.values())

    if not (len(kwargs_only) > 0 or has_varkw):
        raise ValueError(
            """
            Cannot decorate with `partial_from_kwargs` a function without keyword-only arguments.

            partial_from_kwargs allows to create a partial when keyword only arguments are passed,
            but non-keyword only arguments are not treated. If no keyword only arguments are present,
            then this decorator would do nothing.
            """
        )

    # Construct a dictionary of invalid kwargs for every key. This is used to raise an early error.
    kwargs_exclusion_list = {}
    if exclusive_arg_names is not None:
        # Make it a list of tuples
        if isinstance(exclusive_arg_names[0], str):
            exclusive_arg_names = [exclusive_arg_names]
        for exclusion_set in exclusive_arg_names:
            exclusion_set = set(exclusion_set)
            for arg in exclusion_set:
                kwargs_exclusion_list[arg] = kwargs_exclusion_list.get(
                    arg, set()
                ).union(exclusion_set - {arg})

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # If positional arguments are passed, call the function directly
        if len(args) > 0:
            return func(*args, **kwargs)
        elif any(key in maybe_kwargs for key in kwargs.keys()):
            # Only support keyword only arguments
            return func(*args, **kwargs)
        else:
            if not has_varkw:
                for kwarg in kwargs:
                    if kwarg not in kwargs_only:
                        raise TypeError(
                            f"Unexpected keyword argument '{kwarg}' when calling"
                            f"{func}. Valid arguments are {kwargs_only}."
                        )

            # check that arguments are not mutually exclusive
            for kwarg in kwargs:
                if kwarg in kwargs_exclusion_list:
                    if any(arg in kwargs for arg in kwargs_exclusion_list[kwarg]):
                        raise ValueError(
                            f"Cannot specify both {kwarg} and {kwargs_exclusion_list[kwarg]} together."
                        )

            return functools.partial(func, **kwargs)

    return wrapper
