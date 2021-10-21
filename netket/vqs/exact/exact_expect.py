from functools import partial, lru_cache
from typing import Callable, Any, Tuple

import numpy as np

import jax
from jax import numpy as jnp
from jax import tree_map
from netket import jax as nkjax
from netket.stats import statistics, Stats
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch, TrueT, FalseT
from netket.utils import mpi

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
    expval_O = (Ψ.conj() * OΨ).sum()#Ψ.conj().T @ OΨ

    expval_O2 = Ψ.conj().T @ O @ (OΨ)

    variance = expval_O2 - expval_O ** 2
    return Stats(mean=expval_O, error_of_mean=0.0, variance=variance)


@dispatch
def expect_and_grad(
  vstate: ExactState,
  Ô: DiscreteOperator,
  use_covariance: TrueT,
  mutable: Any,
) -> Tuple[Stats, PyTree]:
  _check_hilbert(vstate, Ô)
  
  O = sparsify(Ô)

  Ψ = vstate.to_array()

  OΨ = O@Ψ
  expval_O = (Ψ.conj() * OΨ).sum()
  ΔOΨ = (OΨ - expval_O * Ψ.conj()) * Ψ
  _, Ō_grad, new_model_state = _exp_grad(
      vstate._apply_fun,
      mutable,
      vstate.parameters,
      vstate.model_state,
      vstate._all_states,
      ΔOΨ
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
    inp: jnp.ndarray,
) -> Tuple[PyTree, PyTree]:

  _, vjp_fun, *new_model_state = nkjax.vjp(
      lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
      parameters,
      conjugate=True,
      has_aux=mutable,
  )

  Ō_grad = vjp_fun(inp)[0]

  Ō_grad = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

  new_model_state = new_model_state[0] if mutable else None

  return None, tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad), new_model_state
