from functools import partial
from typing import Any, Callable, Tuple

import numpy as np

import jax
from jax import numpy as jnp
from jax import tree_map

from netket import jax as nkjax
from netket import config
from netket.operator import AbstractOperator, DiscreteOperator
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch, TrueT

from netket.vqs import expect_and_grad
from netket.vqs.mc import kernels, check_hilbert, get_configs, get_fun

from .state import MCState


# If batch_size is None, ignore it and remove it from signature
@expect_and_grad.dispatch(precedence=20)
def expect_and_grad_nominibatch(
    vstate: MCState,
    operator: AbstractOperator,
    cov: Any,
    batch_size: None,
    *args,
    **kwargs,
):
    return expect_and_grad(vstate, operator, cov, *args, **kwargs)


# if no implementation exists for batched, run the code unbatched
@expect_and_grad.dispatch
def expect_and_grad_fallback(
    vstate: MCState,
    operator: AbstractOperator,
    cov: Any,
    batch_size: Any,
    *args,
    **kwargs,
):
    if config.FLAGS["NETKET_DEBUG"]:
        print(
            "Ignoring `batch_size={batch_size}` because no implementation supporting:"
            "batching exists."
        )

    return expect_and_grad(vstate, operator, cov, *args, **kwargs)


@dispatch
def expect_and_grad(  # noqa: F811
    vstate: MCState,
    Ô: DiscreteOperator,
    use_covariance: TrueT,
    batch_size: int,
    *,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    σ, σp, mels = get_configs(vstate, Ô)

    local_estimator_fun = get_fun(vstate, Ô, batch_size)

    Ō, Ō_grad, new_model_state = grad_expect_hermitian(
        local_estimator_fun,
        vstate._apply_fun,
        mutable,
        batch_size,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def grad_expect_hermitian(
    local_value_kernel_batched: Callable,
    model_apply_fun: Callable,
    mutable: bool,
    batch_size: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Tuple[PyTree, PyTree]:

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ_shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    n_samples = σ.shape[0] * mpi.n_nodes

    O_loc = local_value_kernel_batched(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        σp,
        mels,
        batch_size=batch_size,
    )

    Ō = statistics(O_loc.reshape(σ_shape[:-1]).T)

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    if mutable is False:
        vjp_fun_batched = nkjax.vjp_batched(
            lambda w, σ: model_apply_fun({"params": w, **model_state}, σ),
            parameters,
            σ,
            conjugate=True,
            batch_size=batch_size,
            batch_argnums=1,
            nondiff_argnums=1,
        )
        new_model_state = None
    else:
        raise NotImplementedError

    Ō_grad = vjp_fun_batched(
        (jnp.conjugate(O_loc) / n_samples),
    )[0]

    Ō_grad = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    return Ō, tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad), new_model_state
