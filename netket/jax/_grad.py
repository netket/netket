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

from typing import Callable, Tuple, Any, Union, Sequence

import operator

import jax
from jax.util import safe_map

from .utils import is_complex, tree_leaf_iscomplex, eval_shape

map = safe_map


def _ensure_index(x: Any) -> Union[int, Tuple[int, ...]]:
    """Ensure x is either an index or a tuple of indices."""
    try:
        return operator.index(x)
    except TypeError:
        return tuple(map(operator.index, x))


def grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    allow_int: bool = False,
) -> Callable:
    """Creates a function which evaluates the gradient of ``fun``.

    Args:
      fun: Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, or standard Python containers.
        Argument arrays in the positions specified by ``argnums`` must be of
        inexact (i.e., floating-point or complex) type. It
        should return a scalar (which includes arrays with shape ``()`` but not
        arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
      holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
        holomorphic. If True, inputs and outputs must be complex. Default False.
      allow_int: Optional, bool. Whether to allow differentiating with
        respect to integer valued inputs. The gradient of an integer input will
        have a trivial vector-space dtype (float0). Default False.

    Returns:
      A function with the same arguments as ``fun``, that evaluates the gradient
      of ``fun``. If ``argnums`` is an integer then the gradient has the same
      shape and type as the positional argument indicated by that integer. If
      argnums is a tuple of integers, the gradient is a tuple of values with the
      same shapes and types as the corresponding arguments. If ``has_aux`` is True
      then a pair of (gradient, auxiliary_data) is returned.

    For example:

    >>> import jax
    >>>
    >>> grad_tanh = jax.grad(jax.numpy.tanh)
    >>> print(grad_tanh(0.2))
    0.9610429829661166
    """
    value_and_grad_f = value_and_grad(
        fun, argnums, has_aux=has_aux, allow_int=allow_int
    )

    def grad_f(*args, **kwargs):
        _, g = value_and_grad_f(*args, **kwargs)
        return g

    def grad_f_aux(*args, **kwargs):
        (_, aux), g = value_and_grad_f(*args, **kwargs)
        return g, aux

    return grad_f_aux if has_aux else grad_f


def value_and_grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    allow_int: bool = False,
) -> Callable[..., Tuple[Any, Any]]:
    """Create a function which evaluates both ``fun`` and the gradient of ``fun``.

    Args:
      fun: Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, or standard Python containers. It
        should return a scalar (which includes arrays with shape ``()`` but not
        arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
      allow_int: Optional, bool. Whether to allow differentiating with
        respect to integer valued inputs. The gradient of an integer input will
        have a trivial vector-space dtype (float0). Default False.

    Returns:
      A function with the same arguments as ``fun`` that evaluates both ``fun``
      and the gradient of ``fun`` and returns them as a pair (a two-element
      tuple). If ``argnums`` is an integer then the gradient has the same shape
      and type as the positional argument indicated by that integer. If argnums is
      a sequence of integers, the gradient is a tuple of values with the same
      shapes and types as the corresponding arguments.
    """

    argnums = _ensure_index(argnums)
    docstr = (
        "Value and gradient of {fun} with respect to positional "
        "argument(s) {argnums}. Takes the same arguments as {fun} but "
        "returns a two-element tuple where the first element is the value "
        "of {fun} and the second element is the gradient, which has the "
        "same shape as the arguments at positions {argnums}."
    )
    docstr = docstr.format(fun=fun, argnums=argnums)

    # @wraps(fun, docstr=docstr)
    def value_and_grad_f(*args, **kwargs):
        out_shape = eval_shape(fun, *args, has_aux=has_aux, **kwargs)

        _args_iterable = (argnums,) if isinstance(argnums, int) else argnums

        # only check if derivable arguments are complex
        if tree_leaf_iscomplex([args[i] for i in _args_iterable]):
            if is_complex(out_shape):  # C -> C
                return jax.value_and_grad(
                    fun,
                    argnums=argnums,
                    has_aux=has_aux,
                    allow_int=allow_int,
                    holomorphic=True,
                )(*args, **kwargs)
            else:  # C -> R
                raise RuntimeError("C->R function detected, but not supported.")
        else:
            if is_complex(out_shape):  # R -> C

                def grad_rc(*args, **kwargs):
                    if has_aux:

                        def real_fun(*args, **kwargs):
                            val, aux = fun(*args, **kwargs)
                            return val.real, aux

                        def imag_fun(*args, **kwargs):
                            val, aux = fun(*args, **kwargs)
                            return val.imag, aux

                        out_r, grad_r, aux = jax.value_and_grad(
                            real_fun, argnums=argnums, has_aux=True, allow_int=allow_int
                        )(*args, **kwargs)
                        out_j, grad_j, _ = jax.value_and_grad(
                            imag_fun, argnums=argnums, has_aux=True, allow_int=allow_int
                        )(*args, **kwargs)

                    else:
                        real_fun = lambda *args, **kwargs: fun(*args, **kwargs).real
                        imag_fun = lambda *args, **kwargs: fun(*args, **kwargs).imag

                        out_r, grad_r = jax.value_and_grad(
                            real_fun,
                            argnums=argnums,
                            has_aux=False,
                            allow_int=allow_int,
                        )(*args, **kwargs)
                        out_j, grad_j = jax.value_and_grad(
                            imag_fun,
                            argnums=argnums,
                            has_aux=False,
                            allow_int=allow_int,
                        )(*args, **kwargs)

                    out = out_r + 1j * out_j
                    grad = jax.tree_map(lambda re, im: re + 1j * im, grad_r, grad_j)

                    if has_aux:
                        return out, grad, aux
                    else:
                        return out, grad

                return grad_rc(*args, **kwargs)
            else:  # R -> R
                return jax.value_and_grad(
                    fun, argnums=argnums, has_aux=has_aux, allow_int=allow_int
                )(*args, **kwargs)

    return value_and_grad_f
