from functools import partial
from typing import Any, Callable, Tuple

import numpy as np

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


def _check_hilbert(A, B):
    if A.hilbert != B.hilbert:
        raise NotImplementedError(  # pragma: no cover
            f"Non matching hilbert spaces {A.hilbert} and {B.hilbert}"
        )


# pure state, squared operator
@dispatch
def expect_and_grad(
    vstate: MCState,
    Ô: Squared[AbstractOperator],
    use_covariance: TrueT,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    _check_hilbert(vstate, Ô)

    Ô = Ô.parent

    σ = vstate.samples
    σp, mels = Ô.get_conn_padded(np.asarray(σ.reshape((-1, σ.shape[-1]))))

    Ō, Ō_grad, new_model_state = grad_expect_operator_kernel(
        vstate.sampler.machine_pow,
        vstate._apply_fun,
        local_value_squared_kernel,
        mutable,
        vstate.parameters,
        vstate.model_state,
        vstate.samples,
        σp,
        mels,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad


# mixed state, squared super-operator
@dispatch
def expect_and_grad(  # noqa: F811
    vstate: MCMixedState,
    Ô: Squared[AbstractSuperOperator],
    use_covariance: TrueT,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    _check_hilbert(vstate, Ô)

    Ô = Ô.parent

    σ = vstate.samples
    σp, mels = Ô.get_conn_padded(np.asarray(σ.reshape((-1, σ.shape[-1]))))

    Ō, Ō_grad, new_model_state = grad_expect_operator_Lrho2(
        vstate._apply_fun,
        mutable,
        vstate.parameters,
        vstate.model_state,
        vstate.samples,
        σp,
        mels,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad


# mixed state, hermitian operator
@dispatch.multi(
    (MCState, AbstractOperator, TrueT, Any),
    (MCMixedState, AbstractSuperOperator, TrueT, Any),
)
def expect_and_grad(  # noqa: F811
    vstate: MCState,
    Ô: AbstractOperator,
    use_covariance: TrueT,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    _check_hilbert(vstate, Ô)

    σ = vstate.samples
    σp, mels = Ô.get_conn_padded(np.asarray(σ.reshape((-1, σ.shape[-1]))))

    Ō, Ō_grad, new_model_state = grad_expect_hermitian(
        vstate._apply_fun,
        mutable,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad


# mixed state, non-hermitian operator
@dispatch
def expect_and_grad(  # noqa: F811
    vstate: MCState,
    Ô: AbstractOperator,
    use_covariance: FalseT,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    _check_hilbert(vstate, Ô)

    σ = vstate.samples
    σp, mels = Ô.get_conn_padded(np.asarray(σ.reshape((-1, σ.shape[-1]))))

    Ō, Ō_grad, new_model_state = grad_expect_operator_kernel(
        vstate.sampler.machine_pow,
        vstate._apply_fun,
        local_value_kernel,
        mutable,
        vstate.parameters,
        vstate.model_state,
        vstate.samples,
        σp,
        mels,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad


@partial(jax.jit, static_argnums=(0, 1))
def grad_expect_hermitian(
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Tuple[PyTree, PyTree]:

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples = σ.shape[0] * mpi.n_nodes

    O_loc = local_cost_function(
        local_value_cost,
        model_apply_fun,
        {"params": parameters, **model_state},
        σp,
        mels,
        σ,
    )

    Ō = statistics(O_loc.reshape(σ_shape[:-1]).T)

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    if mutable is False:
        _, vjp_fun = nkjax.vjp(
            lambda w: model_apply_fun({"params": w, **model_state}, σ),
            parameters,
            conjugate=True,
        )
        new_model_state = None
    else:
        _, vjp_fun, new_model_state = nkjax.vjp(
            lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
            parameters,
            conjugate=True,
            has_aux=True,
        )
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]

    Ō_grad = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    return Ō, tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad), new_model_state


@partial(jax.jit, static_argnums=(1, 2, 3))
def grad_expect_operator_kernel(
    machine_pow: int,
    model_apply_fun: Callable,
    local_kernel: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Tuple[PyTree, PyTree, Stats]:

    if not config.FLAGS["NETKET_EXPERIMENTAL"]:
        raise RuntimeError(
            """
                           Computing the gradient of a squared or non hermitian
                           operator is an experimental feature under development
                           and is known not to return wrong values sometimes.

                           If you want to debug it, set the environment variable
                           NETKET_EXPERIMENTAL=1
                           """
        )

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    has_aux = mutable is not False
    # if not has_aux:
    #    out_axes = (0, 0)
    # else:
    #    out_axes = (0, 0, 0)

    if not has_aux:
        logpsi = lambda w, σ: model_apply_fun({"params": w, **model_state}, σ)
    else:
        # TODO: output the mutable state
        logpsi = lambda w, σ: model_apply_fun(
            {"params": w, **model_state}, σ, mutable=mutable
        )[0]

    log_pdf = (
        lambda w, σ: machine_pow * model_apply_fun({"params": w, **model_state}, σ).real
    )

    def expect_closure(*args):
        local_kernel_vmap = jax.vmap(
            partial(local_kernel, logpsi), in_axes=(None, 0, 0, 0), out_axes=0
        )

        return nkjax.expect(log_pdf, local_kernel_vmap, *args, n_chains=σ_shape[0])

    def expect_closure_pars(pars):
        return expect_closure(pars, σ, σp, mels)

    Ō, Ō_pb, Ō_stats = nkjax.vjp(expect_closure_pars, parameters, has_aux=True)
    Ō_pars_grad = Ō_pb(jnp.ones_like(Ō))

    return (
        Ō_stats,
        tree_map(lambda x: mpi.mpi_mean_jax(x)[0], Ō_pars_grad),
        model_state,
    )


@partial(jax.jit, static_argnums=(0, 1))
def grad_expect_operator_Lrho2(
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Tuple[PyTree, PyTree, Stats]:
    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples_node = σ.shape[0]

    has_aux = mutable is not False
    # if not has_aux:
    #    out_axes = (0, 0)
    # else:
    #    out_axes = (0, 0, 0)

    if not has_aux:
        logpsi = lambda w, σ: model_apply_fun({"params": w, **model_state}, σ)
    else:
        # TODO: output the mutable state
        logpsi = lambda w, σ: model_apply_fun(
            {"params": w, **model_state}, σ, mutable=mutable
        )[0]

    # local_kernel_vmap = jax.vmap(
    #    partial(local_value_kernel, logpsi), in_axes=(None, 0, 0, 0), out_axes=0
    # )

    # _Lρ = local_kernel_vmap(parameters, σ, σp, mels).reshape((σ_shape[0], -1))
    (
        Lρ,
        der_loc_vals,
    ) = _der_local_values_jax._local_values_and_grads_notcentered_kernel(
        logpsi, parameters, σp, mels, σ
    )
    # _der_local_values_jax._local_values_and_grads_notcentered_kernel returns a loc_val that is conjugated
    Lρ = jnp.conjugate(Lρ)

    LdagL_stats = statistics((jnp.abs(Lρ) ** 2).T)
    LdagL_mean = LdagL_stats.mean

    # old implementation
    # this is faster, even though i think the one below should be faster
    # (this works, but... yeah. let's keep it here and delete in a while.)
    grad_fun = jax.vmap(nkjax.grad(logpsi, argnums=0), in_axes=(None, 0), out_axes=0)
    der_logs = grad_fun(parameters, σ)
    der_logs_ave = tree_map(lambda x: mean(x, axis=0), der_logs)

    # TODO
    # NEW IMPLEMENTATION
    # This should be faster, but should benchmark as it seems slower
    # to compute der_logs_ave i can just do a jvp with a ones vector
    # _logpsi_ave, d_logpsi = nkjax.vjp(lambda w: logpsi(w, σ), parameters)
    # TODO: this ones_like might produce a complexXX type but we only need floatXX
    # and we cut in 1/2 the # of operations to do.
    # der_logs_ave = d_logpsi(
    #    jnp.ones_like(_logpsi_ave).real / (n_samples_node * utils.n_nodes)
    # )[0]
    der_logs_ave = tree_map(lambda x: mpi.mpi_sum_jax(x)[0], der_logs_ave)

    def gradfun(der_loc_vals, der_logs_ave):
        par_dims = der_loc_vals.ndim - 1

        _lloc_r = Lρ.reshape((n_samples_node,) + tuple(1 for i in range(par_dims)))

        grad = mean(der_loc_vals.conjugate() * _lloc_r, axis=0) - (
            der_logs_ave.conjugate() * LdagL_mean
        )
        return grad

    LdagL_grad = jax.tree_util.tree_multimap(gradfun, der_loc_vals, der_logs_ave)

    # ⟨L†L⟩ ∈ R, so if the parameters are real we should cast away
    # the imaginary part of the gradient.
    # we do this also for standard gradient of energy.
    # this avoid errors in #867, #789, #850
    LdagL_grad = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        LdagL_grad,
        parameters,
    )

    return (
        LdagL_stats,
        LdagL_grad,
        model_state,
    )
