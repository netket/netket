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

from typing import Any, Tuple
import warnings

import jax
from jax import numpy as jnp
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.utils.types import PyTree
from netket.utils.dispatch import TrueT, Bool

from netket.vqs import expect_and_grad, expect_and_forces

from .state import MCState


# If batch_size is None, ignore it and remove it from signature
@expect_and_grad.dispatch
def expect_and_grad_nochunking(  # noqa: F811
    vstate: MCState,
    operator: AbstractOperator,
    use_covariance: Bool,
    chunk_size: None,
    *args,
    **kwargs,
):
    return expect_and_grad(vstate, operator, use_covariance, *args, **kwargs)


# if no implementation exists for batched, run the code unbatched
@expect_and_grad.dispatch
def expect_and_grad_fallback(  # noqa: F811
    vstate: MCState,
    operator: AbstractOperator,
    use_covariance: Bool,
    chunk_size: Any,
    *args,
    **kwargs,
):
    warnings.warn(
        f"Ignoring chunk_size={chunk_size} for expect_and_grad method with signature "
        f"({type(vstate)}, {type(operator)}) because no implementation supporting "
        f"chunking for this signature exists."
    )

    return expect_and_grad(vstate, operator, use_covariance, *args, **kwargs)


# dispatch for given chunk_size and use_covariance == True
@expect_and_grad.dispatch
def expect_and_grad_covariance_chunked(  # noqa: F811
    vstate: MCState,
    Ô: AbstractOperator,
    use_covariance: TrueT,
    chunk_size: int,
    *,
    mutable: CollectionFilter,
) -> Tuple[Stats, PyTree]:

    Ō, Ō_grad = expect_and_forces(vstate, Ô, chunk_size, mutable=mutable)
    Ō_grad = _force_to_grad(Ō_grad, vstate.parameters)
    return Ō, Ō_grad


@jax.jit
def _force_to_grad(Ō_grad, parameters):
    Ō_grad = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )
    return Ō_grad
