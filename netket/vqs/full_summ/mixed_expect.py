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

from typing import Callable, Tuple
from functools import partial, lru_cache

import jax
from jax import numpy as jnp
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket.operator import Squared
from netket.stats import Stats
from netket.utils.dispatch import dispatch, FalseT
from netket.utils.types import PyTree

from netket.operator import AbstractSuperOperator, DiscreteOperator


from .mixed_state import FullSumMixedState


def _check_hilbert(A, B):
    if A != B:
        raise NotImplementedError(  # pragma: no cover
            f"Non matching hilbert spaces {A} and {B}"
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
def expect(vstate: FullSumMixedState, Ô: DiscreteOperator) -> Stats:  # noqa: F811
    _check_hilbert(vstate.diagonal.hilbert, Ô.hilbert)

    O = sparsify(Ô)
    rho = vstate.to_matrix()

    # TODO: This performs the full computation on all MPI ranks.
    # It would be great if we could split the computation among ranks.

    Orho = O @ rho
    expval_O = jnp.trace(Orho)

    O2rho = O @ Orho
    expval_O2 = jnp.trace(O2rho)

    variance = expval_O2 - expval_O**2
    return Stats(mean=expval_O, error_of_mean=0.0, variance=variance)


@dispatch
def expect(vstate: FullSumMixedState, Ô: AbstractSuperOperator) -> Stats:  # noqa: F811
    _check_hilbert(vstate.hilbert, Ô.hilbert)
    print("density")

    O = sparsify(Ô)
    Ψ = vstate.to_array()

    # TODO: This performs the full computation on all MPI ranks.
    # It would be great if we could split the computation among ranks.

    OΨ = O @ Ψ
    expval_O = (Ψ.conj() * OΨ).sum()

    variance = jnp.sum(jnp.abs(OΨ - expval_O * Ψ) ** 2)
    return Stats(mean=expval_O, error_of_mean=0.0, variance=variance)


@dispatch
def expect(  # noqa: F811
    vstate: FullSumMixedState, Ô_sq: Squared[AbstractSuperOperator]
) -> Stats:
    _check_hilbert(vstate.hilbert, Ô_sq.hilbert)

    Ô = Ô_sq.parent
    O = sparsify(Ô)
    Ψ = vstate.to_array()

    # TODO: This performs the full computation on all MPI ranks.
    # It would be great if we could split the computation among ranks.

    OΨ = O @ Ψ
    expval_Osq = jnp.linalg.norm(OΨ) ** 2

    OsqΨ = O.conj().T @ OΨ
    expval_Osq2 = jnp.linalg.norm(OsqΨ) ** 2

    variance = expval_Osq2 - expval_Osq**2
    return Stats(mean=expval_Osq, error_of_mean=0.0, variance=variance)


#### gradients


@dispatch
def expect_and_forces(
    vstate: FullSumMixedState,
    Ô: AbstractSuperOperator,
    *,
    mutable: CollectionFilter,
) -> Tuple[Stats, PyTree]:
    if isinstance(Ô, Squared):
        raise NotImplementedError("expect_and_forces not yet implemented for `Squared`")

    _check_hilbert(vstate.hilbert, Ô.hilbert)

    O = sparsify(Ô)
    Ψ = vstate.to_array()
    OΨ = O @ Ψ

    expval_O, Ō_grad, new_model_state = _exp_forces(
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


@dispatch
def expect_and_forces(  # noqa: F811
    vstate: FullSumMixedState,
    Ô_sq: Squared[AbstractSuperOperator],
    *,
    mutable: CollectionFilter,
) -> Tuple[Stats, PyTree]:
    _check_hilbert(vstate.hilbert, Ô_sq.hilbert)

    Ô = Ô_sq.parent
    O = sparsify(Ô)
    Ψ = vstate.to_array()

    OΨ = O @ Ψ
    OOΨ = O.conj().T @ OΨ

    expval_O, Ō_grad, new_model_state = _exp_forces(
        vstate._apply_fun,
        mutable,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,
        OOΨ,
        Ψ,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return expval_O, Ō_grad


@partial(jax.jit, static_argnums=(0, 1))
def _exp_forces(
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    OΨ: jnp.ndarray,
    Ψ: jnp.ndarray,
) -> Tuple[PyTree, PyTree]:
    is_mutable = mutable is not False

    expval_O = (Ψ.conj() * OΨ).sum()
    variance = jnp.sum(jnp.abs(OΨ - expval_O * Ψ) ** 2)

    ΔOΨ = (OΨ - expval_O * Ψ).conj() * Ψ

    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )

    Ō_grad = vjp_fun(ΔOΨ)[0]

    new_model_state = new_model_state[0] if is_mutable else None

    return (
        Stats(mean=expval_O, error_of_mean=0.0, variance=variance),
        Ō_grad,
        new_model_state,
    )


# @dispatch
# def expect_and_grad(
#    vstate: FullSumMixedState,
#    Ô: AbstractSuperOperator,
#    use_covariance: TrueT,
#    *,
#    mutable: CollectionFilter,
# ) -> Tuple[Stats, PyTree]:
#    Ō, Ō_grad = expect_and_forces(vstate, Ô, mutable=mutable)
#    Ō_grad = _force_to_grad(Ō_grad, vstate.parameters)
#    return Ō, Ō_grad


@dispatch
def expect_and_grad(  # noqa: F811
    vstate: FullSumMixedState,
    Ô: Squared[AbstractSuperOperator],
    use_covariance: FalseT,
    *,
    mutable: CollectionFilter,
) -> Tuple[Stats, PyTree]:
    Ō, Ō_grad = expect_and_forces(vstate, Ô, mutable=mutable)
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
