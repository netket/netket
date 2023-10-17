"""
This code has been taken and modified from emcee (https://github.com/dfm/emcee/),
version 3.1.1.

The original copyright notice is reproduced below:

    Copyright (c) 2010-2021 Daniel Foreman-Mackey & contributors.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import jax.numpy as jnp
from jax import lax


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series
    Args:
        x: The series as a 1-D numpy array.
    Returns:
        array: The autocorrelation function of the time series.
    """
    x = jnp.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = jnp.fft.fft(x - jnp.mean(x), n=2 * n)
    acf = jnp.fft.ifft(f * jnp.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf


def auto_window(taus, c):
    m = jnp.arange(len(taus)) < c * taus
    return lax.cond(jnp.any(m), lambda: jnp.argmin(m), lambda: len(taus) - 1)


def integrated_time(x, c=5):
    """Estimate the integrated autocorrelation time of a time series.

    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <https://www.semanticscholar.org/paper/Monte-Carlo-Methods-in-Statistical-Mechanics%3A-and-Sokal/0bfe9e3db30605fe2d4d26e1a288a5e2997e7225>`_
    to determine a reasonable window size.

    Args:
        x: The time series.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)

    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x``.
    """
    if x.ndim != 1:
        raise ValueError("invalid shape")

    f = autocorr_1d(x)
    taus = 2.0 * jnp.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]
