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
"""
Tools to compare and hash floating point numbers safely.
"""

import numpy as np
from netket.utils.types import Array, Union


def comparable(
    x: Array, *, bin_density: int = 3326400, offset: float = 5413 / 15629
) -> Array:
    """
    Casts a floating point input to integer-indexed bins that are safe to compare
    or hash.

    Arguments:
        x: the float array to be converted
        bin_density: the inverse width of each bin. When binning rational numbers,
            it's best to use a multiple of all expected denominators `bin_density`.
            The default is :math:`3326400 = 2^6\times 3^3\times 5^2\times 7\times 11`.
        offset: constant offset added to `bin_density * x` before rounding. To minimise
            the chances of "interesting" numbers appearing on bin boundaries, it's
            best to use a rational number with a large prime denominator.
            The default is 5413/15629, both are primes.

    Returns:
        `x * bin_density + offset` rounded to an integer

    Example:

        >>> comparable([0.0, 0.3, 0.30000001, 1.3])
        array([      0,  997920,  997920, 4324320])
    """
    return np.asarray(np.rint(np.asarray(x) * bin_density + offset), dtype=int)


def comparable_periodic(
    x: Array,
    where: Union[Array, bool] = True,
    *,
    bin_density: int = 3326400,
    offset: float = 5413 / 15629,
) -> Array:
    """
    Casts the fractional parts of floating point input to integer-indexed bins
    that are safe to compare or hash.

    Arguments:
        x: the float array to be converted
        where: specifies whether the fractional part (True) or the full value (False)
            of the input is to be used. Must be broadcastable to `x.shape`.
            If False, the output is the same as that of `comparable`.
        bin_density: the inverse width of each bin. When binning rational numbers,
            it's best to use a multiple of all expected denominators `bin_density`.
            The default is :math:`3326400 = 2^6\times 3^3\times 5^2\times 7\times 11`.
        offset: constant offset added to `bin_density * x` before rounding. To minimse
            the chances of "interesting" numbers appearing on bin boundaries, it's
            best to use a rational number with a large prime denominator.
            The default is 5413/15629, both are primes.

    Returns:
        [`x` or frac(`x`)]` * bin_density + offset` rounded to an integer

    Example:

        >>> comparable_periodic([0.0, 0.3, 0.30000001, 1.3], where = [[True], [False]])
        array([[      0,  997920,  997920,  997920],
               [      0,  997920,  997920, 4324320]])
    """
    bins = np.asarray(np.rint(np.asarray(x) * bin_density + offset), dtype=int)
    return np.where(where, bins % bin_density, bins)


def _prune_zeros(x: Array, atol: float = 1e-08) -> Array:
    # prunes nearly zero entries
    x[np.isclose(x, 0.0, rtol=0.0, atol=atol)] = 0.0
    return x


def prune_zeros(x: Array, atol: float = 1e-08) -> Array:
    """Prunes nearly zero real and imaginary parts"""
    if np.iscomplexobj(x):
        # Check if complex part is nonzero at all
        if np.allclose(x.imag, 0.0, rtol=0.0, atol=atol):
            return _prune_zeros(x.real)
        else:
            return _prune_zeros(x.real) + 1j * _prune_zeros(x.imag)
    else:
        return _prune_zeros(x)


def is_approx_int(x: Array, atol: float = 1e-08) -> Array:
    """
    Returns `True` for all elements of the array that are within
    `atol` to an integer.
    """
    return np.isclose(x, np.rint(x), rtol=0.0, atol=atol)
