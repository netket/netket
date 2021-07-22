from functools import partial
from typing import Any, Callable, Tuple, List, Union

import numpy as np
from numpy import ndarray

import jax
from jax import numpy as jnp
from jax import tree_map

from netket import jax as nkjax
from netket import config
from netket.stats import Stats, statistics, mean
from netket.utils import mpi
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch, TrueT, FalseT

from netket.operator import (
    AbstractOperator,
    AbstractSuperOperator,
    local_cost_function,
    local_value_cost,
    Squared,
    _der_local_values_jax,
)

from .mc_state import MCState
from .mc_mixed_state import MCMixedState

from .mc_expect import local_value_kernel, local_value_squared_kernel
from .utils import vjp_with_aux
from .base import expect_and_grad


def _check_hilbert(A, B):
    if A.hilbert != B.hilbert:
        raise NotImplementedError(  # pragma: no cover
            f"Non matching hilbert spaces {A.hilbert} and {B.hilbert}"
        )


# mixed state, hermitian operator
@expect_and_grad.dispatch
def expect_and_grad_minibatch(  # noqa: F811
    vstate: MCState,
    Ô: AbstractOperator,
    use_covariance: TrueT,
    mutable: Any,
    n_minibatches: int,
) -> Tuple[Stats, PyTree]:

    σ = vstate.samples
    σ_batches = σ.reshape(n_minibatches, -1, σ.shape[1], σ.shape[2])

    σp, mels = Ô.get_conn_padded(σ_batches)

    Ō, Ō_grad, new_model_state = grad_expect_hermitian_minibatch_iterate(
        n_minibatches,
        vstate._apply_fun,
        mutable,
        vstate.parameters,
        vstate.model_state,
        σ_batches,
        σp,
        mels,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad


def grad_expect_hermitian_minibatch_iterate(n_minibatches, *args):
    if n_minibatches == 1:
        return _grad_expect_hermitian_minibatch_iterate_jit(n_minibatches, *args)
    else:
        return _grad_expect_hermitian_minibatch_iterate(n_minibatches, *args)


def _grad_expect_hermitian_minibatch_iterate(
    n_minibatches: int,
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ_batches: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Tuple[PyTree, PyTree]:

    n_minibatches = σ_batches.shape[0]
    n_samples_tot = σ_batches.size // σ_batches.shape[-1] * mpi.n_nodes

    O_locs = []
    for i in range(n_minibatches):
        O_locs.append(
            grad_expect_hermitian_minibatch_forward(
                model_apply_fun,
                parameters,
                model_state,
                σ_batches[i],
                σp[i],
                mels[i],
            )
        )

    Ō = grad_expect_hermitian_minibatch_average(O_locs)

    new_model_state = model_state
    Ō_grad = None
    for i in range(n_minibatches):
        σ_batch = σ_batches[i]
        Ō_grad, new_model_state = grad_expect_hermitian_minibatch_backward(
            model_apply_fun,
            mutable,
            parameters,
            new_model_state,
            σ_batch,
            O_locs[i],
            Ō,
            Ō_grad,
            n_samples_tot,
        )

    return Ō, grad_avg(Ō_grad), new_model_state


_grad_expect_hermitian_minibatch_iterate_jit = jax.jit(
    _grad_expect_hermitian_minibatch_iterate, static_argnums=(0, 1, 2)
)


# part 1: compute only oloc
@partial(jax.jit, static_argnums=0)
def grad_expect_hermitian_minibatch_forward(
    model_apply_fun: Callable,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Tuple[PyTree, PyTree]:

    σ_shape = σ.shape
    n_visible = σ_shape[-1]

    O_loc = local_cost_function(
        local_value_cost,
        model_apply_fun,
        {"params": parameters, **model_state},
        σp.reshape(-1, *σp.shape[-2:]),
        mels.reshape(-1, mels.shape[-1]),
        σ.reshape(-1, σ.shape[-1]),
    )

    return O_loc.reshape(σ_shape[:-1])


# part 2: average them
@jax.jit
def grad_expect_hermitian_minibatch_average(O_locs: List[jnp.ndarray]):
    stats = [statistics(O_loc.T, precompute=False) for O_loc in O_locs]
    stat_acc = None
    for stat in stats:
        if stat_acc is None:
            stat_acc = stat
        else:
            stat_acc = stat_acc.merge(stat)

    stat_acc._precompute_cached_properties()
    return stat_acc


# part 3
@partial(jax.jit, static_argnums=(0, 1))
def grad_expect_hermitian_minibatch_backward(
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    O_loc: ndarray,
    Ō: Stats,
    Ō_grad_accum: PyTree,
    n_samples_tot: int,
) -> Tuple[PyTree, PyTree]:

    σ_shape = σ.shape
    n_samples = σ.shape[0]

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    _, vjp_fun, new_model_state = vjp_with_aux(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        mutable=mutable,
    )
    new_model_state = model_state if mutable is False else new_model_state
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples_tot)[0]

    Ō_grad = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    if Ō_grad_accum is not None:
        Ō_grad = jax.tree_multimap(lambda x, y: x + y, Ō_grad_accum, Ō_grad)

    return Ō_grad, new_model_state


@partial(jax.jit)
def grad_avg(Ō_grad):
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad)
