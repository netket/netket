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
import warnings
from textwrap import dedent

import jax
import jax.numpy as jnp

import netket.jax as nkjax
from netket.utils import mpi, struct, warn_deprecation


@struct.dataclass
class JacobianMode:
    """
    Jax-compatible string type, used to return static information from a jax-jitted
    function.
    """

    name: str = struct.field(pytree_node=False)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"JacobianMode({self.name})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, o):
        if isinstance(o, JacobianMode):
            o = o.name
        return self.name == o


RealMode = JacobianMode("real")
ComplexMode = JacobianMode("complex")
HolomorphicMode = JacobianMode("holomorphic")


@partial(jax.jit, static_argnames=("apply_fun", "holomorphic"))
def choose_jacobian_mode(
    apply_fun, pars, model_state, samples, *, holomorphic
) -> JacobianMode:
    """
    Select an implementation of Jacobian. Returns a Jax-compatible
    (static) string type between

    "real", "complex", "holomorphic"
    """
    homogeneous_vars = nkjax.tree_ishomogeneous(pars)
    leaf_iscomplex = nkjax.tree_leaf_iscomplex(pars)

    if holomorphic is True:
        if homogeneous_vars and leaf_iscomplex:
            ## all complex parameters
            mode = HolomorphicMode
        elif homogeneous_vars and not leaf_iscomplex:
            # all real parameters
            raise ValueError(
                dedent(
                    """
                A function with real parameters cannot be holomorphic.

                Please remove the kw-arg `holomorphic=True`.
                """
                )
            )
        else:
            # mixed complex and real parameters
            warnings.warn(
                dedent(
                    """The ansatz has non homogeneous variables, which might not behave well with the
                       holomorhic implementation.

                       Use `holomorphic=False` or mode='complex' for more accurate results but
                       lower performance.
                    """
                )
            )
            mode = HolomorphicMode
    else:
        complex_output = jax.numpy.iscomplexobj(
            jax.eval_shape(
                apply_fun,
                {"params": pars, **model_state},
                samples.reshape(-1, samples.shape[-1]),
            )
        )

        if complex_output:
            if leaf_iscomplex:
                if holomorphic is None:
                    warnings.warn(
                        dedent(
                            """
                                Complex-to-Complex model detected. Defaulting to `holomorphic=False` for
                                the implementation of QGTJacobianDense.
                                If your model is holomorphic, specify `holomorphic=True` to use a more
                                performant implementation.
                                To suppress this warning specify `holomorphic`.
                                """
                        ),
                        UserWarning,
                    )
                mode = ComplexMode
            else:
                mode = ComplexMode
        else:
            mode = RealMode
    return mode


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
