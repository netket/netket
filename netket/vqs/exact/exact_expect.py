from functools import partial, lru_cache
from typing import Callable

import numpy as np

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket.stats import statistics, Stats
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    DiscreteOperator,
    AbstractSuperOperator,
    Squared,
)

from .exact_state import ExactState


def _check_hilbert(A, B):
    if A.hilbert != B.hilbert:
        raise NotImplementedError(  # pragma: no cover
            f"Non matching hilbert spaces {A.hilbert} and {B.hilbert}"
        )


@lru_cache
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

    OΨ = O @ Ψ
    expval_O = Ψ.conj().T @ OΨ

    expval_O2 = Ψ.conj().T @ O @ (OΨ)

    variance = expval_O2 - expval_O ** 2
    return Stats(mean=expval_O, error_of_mean=0.0, variance=variance)


# @dispatch
# def expect_and_grad(
#    vstate: ExactState,
#    Ô: DiscreteOperator,
#    use_covariance: TrueT,
#    mutable: Any,
# ) -> Tuple[Stats, PyTree]:
#    _check_hilbert(vstate, Ô)
#
#    O = sparsify(Ô)
#    Ψ = vstate.to_array()
#    σ = vstate._all_states
#
#    OΨ = O@Ψ
#    expval_O = Ψ.conj().T@OΨ
#
#    is_mutable = mutable is not False
#    model_apply_fun = vstate._apply_fun
#    model_state = vstate.model_state
#
#    _, vjp_fun, *new_model_state = nkjax.vjp(
#        lambda w: jnp.exp(model_apply_fun({"params": w, **model_state}, σ, mutable=mutable)),
#        parameters,
#        conjugate=True,
#        has_aux=is_mutable,
#    )
#
#    ΔOΨ = OΨ - expval_O
#    vjp_fun(OΨ)
#
#    return _exp_grad(vstate._apply_fun, )
