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
    return dtype
