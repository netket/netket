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
from netket.utils.types import Array, Union


def comparable(
    x: Array, *, bin_density: int = 3326400, offset: float = 5413 / 15629
) -> Array:
    """
    Casts a floating point input to integer-indexed bins that are safe to compare or hash.

    Arguments:
        x: the float array to be converted
        bin_density: the inverse width of each bin. When binning rational numbers,
            it's best to use a multiple of all expected denominators `bin_density`.
            The default is :math:`3326400 = 2^6\times 3^3\times 5^2\times 7\times 11`.
        offset: constant offset added to `bin_density * x` before rounding. To minimse
            the chances of "interesting" numbers appearing on bin boundaries, it's
            best to use a rational number with a large prime denominator.
            The default is 5413/15629, both are primes.

    Returns:
        `x * bin_density + offset` rounded to an integer
    """
    return np.asarray(np.rint(x * bin_density + offset), dtype=int)


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
            of the input is to be used. Must be broadcastable to `x.shape`
        bin_density: the inverse width of each bin. When binning rational numbers,
            it's best to use a multiple of all expected denominators `bin_density`.
            The default is :math:`3326400 = 2^6\times 3^3\times 5^2\times 7\times 11`.
        offset: constant offset added to `bin_density * x` before rounding. To minimse
            the chances of "interesting" numbers appearing on bin boundaries, it's
            best to use a rational number with a large prime denominator.
            The default is 5413/15629, both are primes.

    Returns:
        [`x` or frac(`x`)]` * bin_density + offset` rounded to an integer
    """
    frac = (
        np.asarray(x) % 1.0
    )  # not strictly needed, but may be good for numerical stability
    bin_frac = np.asarray(np.rint(frac * bin_density + offset), dtype=int)
    bin_frac %= bin_density
    if where is not True:  # i.e. we might keep some full values
        bin_x = np.asarray(np.rint(x * bin_density + offset), dtype=int)
        bin_frac = np.where(where, bin_frac, bin_x)
    return bin_frac
