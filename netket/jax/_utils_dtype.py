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

import numpy as np

import jax
from jax import numpy as jnp

from netket.utils.numbers import dtype as _dtype


def is_complex_dtype(typ):
    """
    Returns True if typ is a complex dtype.

    This is almost equivalent to `jnp.iscomplexobj` but also handles types such as
    `float`, `complex` and `int`, which are used throughout netket.
    """
    return jnp.issubdtype(typ, jnp.complexfloating)


def is_real_dtype(typ):
    """
    Returns True if typ is a floating real dtype.

    This is almost equivalent to `jnp.isrealobj` but also handles types such as
    `float`, `complex` and `int`, which are used throughout netket.
    """
    return jnp.issubdtype(typ, jnp.floating)


# Return the type holding the real part of the input type
def dtype_real(typ):
    """
    If typ is a complex dtype returns the real counterpart of typ
    (eg complex64 -> float32, complex128 ->float64).
    Returns typ otherwise.
    """
    if np.issubdtype(typ, np.complexfloating):
        if typ == np.dtype("complex64"):
            return np.dtype("float32")
        elif typ == np.dtype("complex128"):
            return np.dtype("float64")
        else:
            raise TypeError(f"Unknown complex floating type {typ}")
    else:
        return typ


def dtype_complex(typ):
    """
    Return the complex dtype corresponding to the type passed in.
    If it is already complex, do nothing
    """
    if is_complex_dtype(typ):
        return typ
    elif typ == np.dtype("float32"):
        return np.dtype("complex64")
    elif typ == np.dtype("float64"):
        return np.dtype("complex128")
    else:
        raise TypeError(f"Unknown complex type for {typ}")


def maybe_promote_to_complex(*types):
    """
    Maybe promotes the first argument to it's complex counterpart given by
    dtype_complex(typ) if any of the arguments is complex
    """
    main_typ = types[0]

    for typ in types:
        if is_complex_dtype(typ):
            return dtype_complex(main_typ)
    else:
        return main_typ


def canonicalize_dtypes(*values, dtype=None):
    """
    Return the canonicalised result dtype of an operation combining several
    values, with a possible default dtype.

    Equivalent to

    .. code-block:: python

        if dtype is None:
            dtype = jnp.result_type(*[_dtype(x) for x in values])
        return jax.dtypes.canonicalize_dtype(dtype)

    Args:
        *values: all values to combine. Ignored if dtype is not None
        dtype: default value overriding values.
    """
    if dtype is None:
        dtype = jnp.result_type(*[_dtype(x) for x in values])
    # Fallback to x32 when x64 is disabled in JAX
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return jnp.dtype(dtype)


def _in_int_dtype_range(num, dtype):
    return np.iinfo(dtype).min <= num <= np.iinfo(dtype).max


_int_dtypes = [np.int8, np.int16, np.int32, np.int64]
_uint_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]


def bottom_int_dtype(*vals, dtype=None, allow_unsigned: bool = False):
    """
    Find the smallest integer dtype that contains the values

    If the dtype provided is floating, simply return it.

    Args:
        values: An arbitrary number of values
        dtype: default dtype to start from
        allow_unsigned: whether to allow unsigned dtypes
    """
    if dtype is None:
        dtype = canonicalize_dtypes(*vals, dtype=dtype)

    if np.issubdtype(dtype, np.floating):
        return jnp.dtype(dtype)

    # Check if all values are unsigned. If yes, work with uint types,
    # else check int types
    if any(v < 0 for v in vals) or not allow_unsigned:
        dtypes_choice = _int_dtypes
    # elif all(v in (0, 1) for v in vals):
    #    return np.dtype(bool)
    else:
        dtypes_choice = _uint_dtypes

    for dtyp in dtypes_choice:
        if all(_in_int_dtype_range(v, dtyp) for v in vals):
            return jnp.dtype(dtyp)
    raise ValueError(
        f"Some of the values in {vals} do not fit in any numpy integer dtype."
    )
