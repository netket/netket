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

from functools import partial

import jax
import jax.numpy as jnp

from netket.utils import mpi, warn_deprecation


def sanitize_diag_shift(diag_shift, diag_scale, rescale_shift):
    """Sanitises different inputs for diag_shift etc.


    Also raises a deprecation warnings for `rescale_shift`.

    Returns:
        the tuple `(diag_shift, diag_scale)`.
    """

    if diag_shift is None:
        diag_shift = 0.0

    if rescale_shift is False:
        warn_deprecation(
            "`rescale_shift` is deprecated, please do not specify `rescale_shift=False`."
        )
        if diag_scale is not None:
            raise ValueError(
                "`rescale_shift` and `diag_scale` must not be specified together."
            )

        return diag_shift, 0.0
    elif rescale_shift is True:
        warn_deprecation(
            f"`rescale_shift` is deprecated, use `diag_scale={diag_shift}, diag_shift=0` instead."
        )
        if diag_scale is not None:
            raise ValueError(
                "`rescale_shift` and `diag_scale` must not be specified together."
            )

        return 0.0, diag_shift
    elif rescale_shift is None:
        if diag_scale is None:
            diag_scale = 0.0
        return diag_shift, diag_scale
    else:
        raise ValueError("`rescale_shift` must be boolean or None.")


def to_shift_offset(diag_shift, diag_scale):
    if diag_scale == 0.0:
        return diag_shift, None
    else:
        return diag_scale, diag_shift / diag_scale


@partial(jax.jit, static_argnames="ndims")
def rescale(centered_oks, offset, *, ndims: int = 1):
    """
    compute ΔOₖ/√Sₖₖ and √Sₖₖ
    to do scale-invariant regularization (Becca & Sorella 2017, pp. 143)
    Sₖₗ/(√Sₖₖ√Sₗₗ) = ΔOₖᴴΔOₗ/(√Sₖₖ√Sₗₗ) = (ΔOₖ/√Sₖₖ)ᴴ(ΔOₗ/√Sₗₗ)

    Args:
        centered_oks: A mean-zero Jacobian.
        ndims: A number of leading dimensions to use to compute the
            rescale factor. Those should be all the non-parameters
            axes in the jacobian (so it should be 1 normally, 2 for
            non holomorphic stacked jacobians).
    """
    # should be (0,) for standard, (0,1) when we have 2 jacobians in complex mode
    axis = tuple(range(ndims))

    scale = jax.tree_map(
        lambda x: (
            mpi.mpi_sum_jax(jnp.sum((x * x.conj()).real, axis=axis, keepdims=True))[0]
            + offset
        )
        ** 0.5,
        centered_oks,
    )
    centered_oks = jax.tree_map(jnp.divide, centered_oks, scale)
    scale = jax.tree_map(partial(jnp.squeeze, axis=axis), scale)
    return centered_oks, scale
