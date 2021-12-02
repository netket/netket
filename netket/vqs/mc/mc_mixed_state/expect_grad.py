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
from typing import Any, Callable, Tuple

import numpy as np

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket.stats import Stats, statistics, mean
from netket.utils import mpi
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch, FalseT

from netket.operator import (
    AbstractSuperOperator,
    # local_value_cost,
    Squared,
    _der_local_values_jax,
)

from .state import MCMixedState


# mixed state, squared super-operator
@dispatch
def expect_and_grad(  # noqa: F811
    vstate: MCMixedState,
    Ô: Squared[AbstractSuperOperator],
    use_covariance: FalseT,
    *,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
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
    der_logs_ave = jax.tree_map(lambda x: mean(x, axis=0), der_logs)

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
    der_logs_ave = jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], der_logs_ave)

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
