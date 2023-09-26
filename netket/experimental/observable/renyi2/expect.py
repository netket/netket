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
import netket as nk

from netket.vqs import MCState, expect
from netket.stats import statistics as mpi_statistics

from .S2_operator import Renyi2EntanglementEntropy


@expect.dispatch
def Renyi2(vstate: MCState, op: Renyi2EntanglementEntropy):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    return Renyi2_sampling_MCState(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        vstate.samples,
        op.subsystem,
    )


@partial(jax.jit, static_argnames=("afun"))
def Renyi2_sampling_MCState(
    afun,
    params,
    model_state,
    samples,
    subsystem,
):
    N = samples.shape[-1]
    n_chains = int(samples.shape[0] / 2)

    σ_η = samples[:n_chains]
    σp_ηp = samples[n_chains:]

    σ_η = σ_η.reshape(-1, N)
    σp_ηp = σp_ηp.reshape(-1, N)

    n_samples = int(σ_η.shape[0] / 2)

    σ = σ_η[:, subsystem]
    σp = σp_ηp[:, subsystem]

    σ_ηp = jnp.copy(σp_ηp)
    σp_η = jnp.copy(σ_η)

    σ_ηp = σ_ηp.at[:, subsystem].set(σ)
    σp_η = σp_η.at[:, subsystem].set(σp)

    def kernel_fun(params, model_state, σ_ηp, σp_η, σ_η, σp_ηp):
        W = {"params": params, **model_state}

        return jnp.exp(afun(W, σ_ηp) + afun(W, σp_η) - afun(W, σ_η) - afun(W, σp_ηp))

    kernel_values = kernel_fun(params, model_state, σ_ηp, σp_η, σ_η, σp_ηp)

    Renyi2_stats = mpi_statistics(kernel_values.reshape((n_chains, -1)).T)

    Renyi2_stats = Renyi2_stats.replace(
        variance=Renyi2_stats.variance / (Renyi2_stats.mean.real * jnp.log(2)) ** 2
    )

    Renyi2_stats = Renyi2_stats.replace(
        error_of_mean=jnp.sqrt(
            Renyi2_stats.variance / (n_samples * nk.utils.mpi.n_nodes)
        )
    )

    Renyi2_stats = Renyi2_stats.replace(mean=-jnp.log2(Renyi2_stats.mean).real)

    return Renyi2_stats
