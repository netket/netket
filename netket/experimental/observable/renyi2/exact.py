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

import jax.numpy as jnp
import qutip
import numpy as np

from netket.vqs import FullSumState, expect
from netket.stats import Stats

from .S2_operator import Renyi2EntanglementEntropy


@expect.dispatch
def Renyi2(vstate: FullSumState, op: Renyi2EntanglementEntropy):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    return Renyi2_sampling_FullSumState(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,
        op.subsystem,
    )


def Renyi2_sampling_FullSumState(
    afun,
    params,
    model_state,
    sigma,
    subsystem,
):
    N = sigma.shape[-1]

    state = jnp.exp(afun({"params": params, **model_state}, sigma))
    state = state / jnp.sqrt(jnp.sum(jnp.abs(state) ** 2))

    state_qutip = qutip.Qobj(np.array(state))

    state_qutip.dims = [[2] * N, [1] * N]

    mask = np.zeros(N, dtype=bool)

    if len(subsystem) == mask.size or len(subsystem) == 0:
        out = 0

    else:
        mask[subsystem] = True

        rdm = state_qutip.ptrace(np.arange(N)[mask])

        n = 2
        out = np.log2(np.trace(np.linalg.matrix_power(rdm, n))) / (1 - n)

        out = np.absolute(out.real)

    S2_stats = Stats(mean=out, error_of_mean=0.0, variance=0.0)

    return S2_stats
