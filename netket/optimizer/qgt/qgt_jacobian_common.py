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
from netket.utils import mpi


@partial(jax.jit, static_argnums=(0, 4, 5))
def _choose_jacobian_mode(apply_fun, pars, model_state, samples, mode, holomorphic):
    homogeneous_vars = nkjax.tree_ishomogeneous(pars)
    leaf_iscomplex = nkjax.tree_leaf_iscomplex(pars)

    if holomorphic is True:
        if homogeneous_vars and leaf_iscomplex:
            ## all complex parameters
            mode = "holomorphic"
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
            mode = "holomorphic"
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
                mode = "complex"
            else:
                mode = "complex"
        else:
            mode = "real"

    if mode == "real":
        return 0
    elif mode == "complex":
        return 1
    elif mode == "holomorphic":
        return 2
    else:
        raise ValueError(f"unknown mode {mode}")


def choose_jacobian_mode(afun, pars, state, samples, *, mode, holomorphic):
    """
    Select an implementation of Jacobian
    """
    i = int(_choose_jacobian_mode(afun, pars, state, samples, mode, holomorphic))
    if i == 0:
        return "real"
    elif i == 1:
        return "complex"
    elif i == 2:
        return "holomorphic"
    else:
        raise ValueError(f"unknown mode {i}")


def rescale(centered_oks, *, ndims: int = 1):
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
        lambda x: mpi.mpi_sum_jax(
            jnp.sum((x * x.conj()).real, axis=axis, keepdims=True)
        )[0]
        ** 0.5,
        centered_oks,
    )
    centered_oks = jax.tree_map(jnp.divide, centered_oks, scale)
    scale = jax.tree_map(partial(jnp.squeeze, axis=axis), scale)
    return centered_oks, scale
