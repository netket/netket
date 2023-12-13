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

from numba import jit

import numpy as np
from flax import struct

from netket.operator import DiscreteOperator
from netket.utils.types import Array

from .base import MetropolisRule


@struct.dataclass
class CustomRuleState:
    sections: np.ndarray
    rand_op_n: np.ndarray
    weight_cumsum: np.ndarray


class CustomRuleNumpy(MetropolisRule):
    operator: Any = struct.field(pytree_node=False)
    weight_list: Any = struct.field(pytree_node=False)

    def __init__(self, operator: DiscreteOperator, weight_list: Array = None):
        """
        Construct a Custom Rule.

        Args:
            operator: a LocalOperator describing the possible moves.
            weight_list: an optional list of probability for every move.
        """
        if not isinstance(operator, DiscreteOperator):
            raise TypeError(
                "Argument to CustomRuleNumpy must be a valid operator, "
                f"but operator is a {type(operator)}."
            )

        # Raise errors if hilbert is not valid
        _check_operators(operator.operators)

        if weight_list is None:
            weight_list = np.ones(operator.n_operators, dtype=np.float32)

        if weight_list is not None:
            if weight_list.shape != (operator.n_operators,):
                raise ValueError("move_weights have the wrong shape")
            if weight_list.min() < 0:
                raise ValueError("move_weights must be positive")

        self.operator = operator
        # normalise
        self.weight_list = weight_list / weight_list.sum()

    def init_state(rule, sampler, machine, params, key):
        return CustomRuleState(
            sections=np.empty(sampler.n_batches, dtype=np.int32),
            rand_op_n=np.empty(sampler.n_batches, dtype=np.int32),
            weight_cumsum=rule.weight_list.cumsum(),
        )

    def transition(rule, sampler, machine, parameters, state, rng, σ):
        rule_state = state.rule_state

        # numba does not support jitting np.random number generators
        # so we have to generate the random numbers outside the jit
        # block
        rnd_uniform = rng.uniform(0.0, 1.0, size=σ.shape[0])

        _pick_random_and_init(
            σ.shape[0],
            rule_state.weight_cumsum,
            rnd_uniform=rnd_uniform,
            out=rule_state.rand_op_n,
        )

        σ_conns, mels = rule.operator.get_conn_filtered(
            state.σ, rule_state.sections, rule_state.rand_op_n
        )

        # numba does not support jitting np.random number generators
        # so we have to generate the random numbers outside the jit
        # block
        rnd_uniform = rng.uniform(0.0, 1.0, size=state.σ1.shape[0])

        _choose_and_return(
            state.σ1,
            σ_conns,
            mels,
            rule_state.sections,
            state.log_prob_corr,
            rnd_uniform,
        )


@jit(nopython=True)
def _pick_random_and_init(batch_size, move_cumulative, rnd_uniform, out):
    for i in range(batch_size):
        p = rnd_uniform[i]
        out[i] = np.searchsorted(move_cumulative, p)

    # return out


@jit(nopython=True)
def _choose_and_return(σp, x_prime, mels, sections, log_prob_corr, rnd_uniform):
    low = 0
    for i in range(σp.shape[0]):
        p = rnd_uniform[i]
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
        op = op.todense()
        assert op.imag.max() < 1.0e-10
        assert op.min() >= 0
        assert np.allclose(op.sum(axis=0), 1.0)
        assert np.allclose(op.sum(axis=1), 1.0)
        assert np.allclose(op, op.T)
