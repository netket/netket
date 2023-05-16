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

import warnings
import functools
import inspect

from textwrap import dedent

from .config_flags import config
from .version_check import module_version


def deprecated(reason=None, func_name=None):
    r"""
    This is a decorator which can be used to mark functions as deprecated. It
    will result in a warning being emitted when the function is used.
    """

    def decorator(func):
        object_type = "class" if inspect.isclass(func) else "function"
        message = "Call to deprecated {} {!r}".format(
            object_type, func_name or func.__name__
        )
        if reason is not None:
            message += f" ({dedent(reason)})"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def warn_deprecation(message):
    r"""
    This is a function that sends a deprecation warning to the user about a
    function that will is now deprecated and will be removed in a future
    major release.

    :param message: A mandatory message documenting the deprecation.
    """
    warnings.warn(dedent(message), category=FutureWarning, stacklevel=2)


def deprecated_new_name(func_name, reason=""):
    def deprecated_decorator(func):
        @functools.wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                dedent(
                    f"""

    {func.__name__} has been renamed to {func_name}. The old name is
    now deprecated and will be removed in the next minor version.

    Please update your code by chaing occurences of `{func.__name__}` with
    `{func_name}`.

    {dedent(reason)}

                    """
                ),
                category=FutureWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


_dep_msg = """

**DEPRECATION_WARNING:**
    The `dtype` argument to neural-network layers and models is deprecated
    throughout NetKet to maintain consistency with new releases of flax.
    Please use `param_dtype` instead.

    This warning will become an error in a future version of NetKet.

"""


def _dtype_deprecated(self):
    warn_deprecation(_dep_msg)
    return self.param_dtype


def deprecate_dtype(clz):
    """
    Decorator taking a class or a function returning an instance of a class
    and replacing the `dtype` argument with `param_dtype`.

    This decorator also adds a deprecated field `dtype` to the class, which
    returns `param_dtype`.
    """
    # Sphinx is terrible at understanding wrappers of classes, so we detect
    # this env variable set by us sphinx's conf.py, and we do not apply the
    # decorator if that's the case.
    if config.netket_sphinx_build:
        return clz

    # If it is a class, then add the deprecated dtype attribute returning
    # `param.dtype`
    # else, if this is a function returning a class, it will be set later
    # on.
    if inspect.isclass(clz):
        lazy = False
        if not hasattr(clz, "dtype"):
            clz.dtype = property(_dtype_deprecated)
    else:
        lazy = True

    @functools.wraps(clz)
    def helper(*args, **kws):
        # deprecated dtype argument
        if "dtype" in kws.keys():
            warn_deprecation(_dep_msg)
            dtype = kws.pop("dtype")
            kws["param_dtype"] = dtype
        res = clz(*args, **kws)
        if lazy:
            _clz = type(res)
            if not hasattr(_clz, "dtype"):
                _clz.dtype = property(_dtype_deprecated)
        return res

    return helper


# TODO: remove the switch when we support only jax >= 0.3.17


def pure_callback(callback, result_shape_dtypes, *args):
    if module_version("jax") >= (0, 3, 17):
        from jax import pure_callback

        return pure_callback(callback, result_shape_dtypes, *args)
    else:
        from jax.experimental.host_callback import call

        return call(callback, *args, result_shape=result_shape_dtypes)
