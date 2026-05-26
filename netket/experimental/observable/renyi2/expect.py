# Copyright 2022 The NetKet Authors - All rights reserved.
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

import jax.numpy as jnp
import jax

from netket.vqs import MCState, expect
from netket.vqs.mc.common import local_estimators
from netket._src.stats.local_estimators import LocalEstimatorsBatch
from netket import jax as nkjax

from .S2_operator import Renyi2EntanglementEntropy


@partial(jax.jit, static_argnames=("afun", "chunk_size"))
def _renyi2_kernel_values(
    afun, params, model_state, σ_η, σp_ηp, partition, *, chunk_size
):
    N = σ_η.shape[-1]
    σ_η = σ_η.reshape(-1, N)
    σp_ηp = σp_ηp.reshape(-1, N)

    σ = σ_η[:, partition]
    σp = σp_ηp[:, partition]

    σ_ηp = jnp.copy(σp_ηp).at[:, partition].set(σ)
    σp_η = jnp.copy(σ_η).at[:, partition].set(σp)

    @partial(
        nkjax.apply_chunked, in_axes=(None, None, 0, 0, 0, 0), chunk_size=chunk_size
    )
    def kernel_fun(params, model_state, σ_ηp, σp_η, σ_η, σp_ηp):
        W = {"params": params, **model_state}
        return jnp.exp(afun(W, σ_ηp) + afun(W, σp_η) - afun(W, σ_η) - afun(W, σp_ηp))

    return kernel_fun(params, model_state, σ_ηp, σp_η, σ_η, σp_ηp)


def _renyi2_combinator(mu):
    return -jnp.log2(mu[0]).real


@local_estimators.dispatch
def _(
    vstate: MCState, op: Renyi2EntanglementEntropy, chunk_size: int | None
) -> LocalEstimatorsBatch:  # noqa: F811
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    samples = vstate.samples
    n_chains = samples.shape[0]
    n_samples = samples.shape[0] * samples.shape[1]

    if n_chains % 2 != 0 and not vstate.sampler.is_exact:
        raise ValueError("Use an even number of chains.")

    if n_chains == 1:
        if n_samples % 2 != 0:
            samples = samples[:, :-1]
        σ_η = samples[:, : (n_samples // 2)]
        σp_ηp = samples[:, (n_samples // 2) :]
    else:
        σ_η = samples[: (n_chains // 2)]
        σp_ηp = samples[(n_chains // 2) :]

    n_chains_eff = σ_η.shape[0]

    kernel_values = _renyi2_kernel_values(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        σ_η,
        σp_ηp,
        op.partition,
        chunk_size=chunk_size,
    )

    data = kernel_values.reshape(n_chains_eff, -1, 1)
    return LocalEstimatorsBatch(data=data, combinator=_renyi2_combinator)


@expect.dispatch
def Renyi2(
    vstate: MCState, op: Renyi2EntanglementEntropy, chunk_size: int | None
):  # noqa: F811
    return local_estimators(vstate, op, chunk_size).to_stats()
