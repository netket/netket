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

from typing import Any
import math

from numba import jit

import numpy as np
from flax import struct

from netket.legacy import random as _random
from netket.operator import AbstractOperator


from ..metropolis import MetropolisRule


@struct.dataclass
class CustomRuleState:
    sections: np.ndarray
    rand_op_n: np.ndarray
    weight_cumsum: np.ndarray


@struct.dataclass
class CustomRuleNumpy(MetropolisRule):
    operator: Any = struct.field(pytree_node=False)
    weight_list: Any = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        # Raise errors if hilbert is not an Hilbert
        if not isinstance(self.operator, AbstractOperator):
            raise TypeError(
                "Argument to CustomRuleNumpy must be a valid operator.".format(
                    type(operator)
                )
            )
        _check_operators(self.operator.operators)

        if self.weight_list is not None:
            if self.weight_list.shape != (self.operator.n_operators,):
                raise ValueError("move_weights have the wrong shape")
            if self.weight_list.min() < 0:
                raise ValueError("move_weights must be positive")
        else:
            object.__setattr__(
                self,
                "weight_list",
                np.ones(self.operator.n_operators, dtype=np.float32),
            )

        object.__setattr__(
            self, "weight_list", self.weight_list / self.weight_list.sum()
        )

    def init_state(rule, sampler, machine, params, key):
        return CustomRuleState(
            sections=np.empty(sampler.n_batches, dtype=np.int32),
            rand_op_n=np.empty(sampler.n_batches, dtype=np.int32),
            weight_cumsum=rule.weight_list.cumsum(),
        )

    def transition(rule, sampler, machine, parameters, state, rng, σ):
        rule_state = state.rule_state

        _pick_random_and_init(
            σ.shape[0],
            rule_state.weight_cumsum,
            out=rule_state.rand_op_n,
        )

        σ_conns, mels = rule.operator.get_conn_filtered(
            state.σ, rule_state.sections, rule_state.rand_op_n
        )

        _choose_and_return(
            state.σ1, σ_conns, mels, rule_state.sections, state.log_prob_corr
        )


@jit(nopython=True)
def _pick_random_and_init(batch_size, move_cumulative, out):
    for i in range(batch_size):
        p = _random.uniform()
        out[i] = np.searchsorted(move_cumulative, p)

    # return out


@jit(nopython=True)
def _choose_and_return(σp, x_prime, mels, sections, log_prob_corr):
    low = 0
    for i in range(σp.shape[0]):
        p = _random.uniform()
        exit_state = 0
        cumulative_prob = mels[low].real
        while p > cumulative_prob:
            exit_state += 1
            cumulative_prob += mels[low + exit_state].real
        σp[i] = x_prime[low + exit_state]
        low = sections[i]

    log_prob_corr.fill(0.0)


def _check_operators(operators):
    for op in operators:
        assert op.imag.max() < 1.0e-10
        assert op.min() >= 0
        assert np.allclose(op.sum(axis=0), 1.0)
        assert np.allclose(op.sum(axis=1), 1.0)
        assert np.allclose(op, op.T)
