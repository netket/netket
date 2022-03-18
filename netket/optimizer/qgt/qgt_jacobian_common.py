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

import netket.jax as nkjax
from netket.utils.struct import dataclass


@dataclass
class RealT:
    pass


@dataclass
class ComplexT:
    pass


@dataclass
class HoloT:
    pass


@partial(jax.jit, static_argnums=(0, 4, 5))
def _choose_jacobian_mode(apply_fun, pars, model_state, samples, mode, holomorphic):
    homogeneous_vars = nkjax.tree_ishomogeneous(pars)

    if holomorphic is True:
        if not homogeneous_vars:
            warnings.warn(
                dedent(
                    """The ansatz has non homogeneous variables, which might not behave well with the
                       holomorhic implemnetation.
                       Use `holomorphic=False` or mode='complex' for more accurate results but
                       lower performance.
                    """
                )
            )
        mode = "holomorphic"
    else:
        leaf_iscomplex = nkjax.tree_leaf_iscomplex(pars)
        complex_output = nkjax.is_complex(
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
        return RealT()
    elif mode == "complex":
        return ComplexT()
    elif mode == "holomorphic":
        return HoloT()
    else:
        raise ValueError(f"unknown mode {mode}")


def choose_jacobian_mode(afun, pars, state, samples, *, mode, holomorphic):
    """
    Select an implementation of Jacobian
    """
    i = _choose_jacobian_mode(afun, pars, state, samples, mode, holomorphic)

    if isinstance(i, RealT):
        return "real"
    elif isinstance(i, ComplexT):
        return "complex"
    elif isinstance(i, HoloT):
        return "holomorphic"
    else:
        raise ValueError(f"unknown mode {i}")
