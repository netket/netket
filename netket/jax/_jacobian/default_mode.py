# Copyright 2023 The NetKet Authors - All rights reserved.
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

from collections.abc import Callable
from functools import partial
import warnings

import jax

import netket.jax as nkjax
from netket.utils import struct
from netket.utils.types import PyTree, Array
from netket.errors import (
    HolomorphicUndeclaredWarning,
    IllegalHolomorphicDeclarationForRealParametersError,
)


class JacobianMode(struct.Pytree):
    """
    Jax-compatible string type, used to return static information from a jax-jitted
    function.
    """

    name: str = struct.field(pytree_node=False)

    def __init__(self, name: str) -> None:
        self.name = name

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


@partial(jax.jit, static_argnames=("apply_fun", "holomorphic", "warn"))
def jacobian_default_mode(
    apply_fun: Callable[[PyTree, Array], Array],
    pars: PyTree,
    model_state: PyTree | None,
    samples: Array,
    *,
    holomorphic: bool | None = None,
    warn: bool = True,
) -> JacobianMode:
    """
    Returns the default `mode` for {func}`netket.jax.jacobian` given a certain
    wave-function ansatz.

    This function uses an abstract evaluation of the ansatz to determine if
    the ansatz has real or complex output, and uses that to determine the
    default mode to be used to compute the Jacobian.

    In particular:
     - for functions with a real output, it will return `RealMode`.
     - for functions with a complex output, it will return:
       - If `holomorphic==False` or it not been specified, it will return
       `ComplexMode`, which will force the calculation of both the jacobian
       and adjoint jacobian. See the documentation of{func}`nk.jax.jacobian`
       for more details.
       - If `holomorphic==True`, it will compute only the complex-valued
       jacobian, and assumes the adjoint-jacobian to be zero.

    This function will also raise an error if `holomorphic` is not specified
    but the output is complex.

    Args:
        apply_fun: A callable taking as input a pytree of parameters and the samples,
            and returning the output.
        pars: The Pytree of parameters.
        model_state: The optional `model_state`, according to the flax model definition.
        samples: An array of samples.
        holomorphic: A boolean specifying whether `apply_fun` is
            holomorphic or not (`None` by default).
        warn: A boolean specifying whether to raise a warning
            when holomorphic is not specified. For internal use
            only.

    """
    nkjax.tree_ishomogeneous(pars)
    nkjax.tree_leaf_iscomplex(pars)
    leaf_isreal = nkjax.tree_leaf_isreal(pars)

    if holomorphic is True:
        if leaf_isreal:
            # all real or mixed real/complex parameters. It's not holomorphic
            raise IllegalHolomorphicDeclarationForRealParametersError()
        else:
            ## all complex parameters
            mode = HolomorphicMode
    else:
        if model_state is None:
            model_state = {}

        complex_output = jax.numpy.iscomplexobj(
            jax.eval_shape(
                apply_fun,
                {"params": pars, **model_state},
                samples.reshape(-1, samples.shape[-1]),
            )
        )

        if complex_output:
            if not leaf_isreal:
                if holomorphic is None and warn:
                    warnings.warn(
                        HolomorphicUndeclaredWarning(),
                        UserWarning,
                        stacklevel=2,
                    )
                mode = ComplexMode
            else:
                mode = ComplexMode
        else:
            mode = RealMode
    return mode
