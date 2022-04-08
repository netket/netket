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

from functools import partial, lru_cache
from typing import Callable, Any, Tuple

import jax
from jax import numpy as jnp
from netket import jax as nkjax
from netket.stats import Stats
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch, TrueT

from netket.operator import DiscreteOperator


from .state import ExactState


def _check_hilbert(A, B):
    if A.hilbert != B.hilbert:
        raise NotImplementedError(  # pragma: no cover
            f"Non matching hilbert spaces {A.hilbert} and {B.hilbert}"
        )


# TODO: This cache is here so that we don't re-compute the sparse representation of the operators at every VMC step
# but instead we cache the last 5 used. Should investigate a better way to implement this caching.
@lru_cache(5)
def sparsify(Ô):
    """
    Converts to sparse but also cache the sparsificated result to speed up.
    """
    return Ô.to_sparse()


@dispatch
def expect(vstate: ExactState, Ô: DiscreteOperator) -> Stats:  # noqa: F811
    _check_hilbert(vstate, Ô)

    O = sparsify(Ô)
    Ψ = vstate.to_array()

    # TODO: This performs the full computation on all MPI ranks.
    # It would be great if we could split the computation among ranks.

    OΨ = O @ Ψ
    expval_O = (Ψ.conj() * OΨ).sum()

    variance = jnp.sum(jnp.abs(OΨ - expval_O * Ψ) ** 2)
    return Stats(mean=expval_O, error_of_mean=0.0, variance=variance)


@dispatch
def expect_and_grad(
    vstate: ExactState,
    Ô: DiscreteOperator,
    use_covariance: TrueT,
    *,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    _check_hilbert(vstate, Ô)

    O = sparsify(Ô)
    Ψ = vstate.to_array()
    OΨ = O @ Ψ

    _, Ō_grad, expval_O, new_model_state = _exp_grad(
        vstate._apply_fun,
        mutable,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,
        OΨ,
        Ψ,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return expval_O, Ō_grad


@partial(jax.jit, static_argnums=(0, 1))
def _exp_grad(
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    OΨ: jnp.ndarray,
    Ψ: jnp.ndarray,
) -> Tuple[PyTree, PyTree]:
    is_mutable = mutable is not False

    expval_O = (Ψ.conj() * OΨ).sum()
    ΔOΨ = (OΨ - expval_O * Ψ).conj() * Ψ

    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )

    Ō_grad = vjp_fun(ΔOΨ)[0]

    Ō_grad = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    new_model_state = new_model_state[0] if is_mutable else None

    return (
        None,
        Ō_grad,
        expval_O,
        new_model_state,
    )
