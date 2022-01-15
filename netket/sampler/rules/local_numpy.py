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

from numba import jit

import numpy as np
from flax import struct

from ..metropolis import MetropolisRule


@struct.dataclass
class LocalRuleState:
    local_states: np.ndarray
    """Preallocated array for sections"""


@struct.dataclass
class LocalRuleNumpy(MetropolisRule):
    def init_state(rule, sampler, machine, params, key):
        return LocalRuleState(local_states=np.array(sampler.hilbert.local_states))

    def transition(rule, sampler, machine, parameters, state, rng, σ):
        σ = state.σ
        σ1 = state.σ1

        si = rng.integers(0, sampler.hilbert.size, size=(σ.shape[0]))
        rs = rng.integers(0, sampler.hilbert.local_size - 1, size=(σ.shape[0]))

        _kernel(σ, σ1, si, rs, state.rule_state.local_states)

    def __repr__(self):
        return "LocalRuleNumpy()"


@jit(nopython=True)
def _kernel(σ, σ1, si, rs, local_states):
    σ1[:] = σ

    for i in range(σ1.shape[0]):
        _si = si[i]
        _rs = rs[i]
        _rs_offset = int(local_states[_rs] >= σ1[i, _si])

        σ1[i, _si] = local_states[_rs + _rs_offset]
