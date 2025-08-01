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

import math

from numba import jit

import numpy as np

from netket.operator import DiscreteOperator
from netket.operator._discrete_operator_jax import DiscreteJaxOperator
from netket.utils import struct

from .base import MetropolisRule


@struct.dataclass
class HamiltonianRuleState:
    sections: np.ndarray
    """Preallocated array for sections"""


class HamiltonianRuleNumpy(MetropolisRule):
    r"""
    Rule for Numpy sampler backend proposing moves according to the terms in an operator.

    In this case, the transition matrix is taken to be:

    .. math::

       T( \mathbf{s} \rightarrow \mathbf{s}^\prime) =
        \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),

    """

    operator: DiscreteOperator = struct.field(pytree_node=False)
    """The (hermitian) operator giving the transition amplitudes."""

    def __init__(self, operator: DiscreteOperator):
        """
        Constructs the Hamiltonian sampling rule for the
        :class:`netket.sampler.MetropolisNumpy` sampler.

        If you are using the standard jax sampler, look for
        :class:`netket.sampler.rules.HamiltonianRule` .

        Args:
            operator: The hermitian operator to be used to generate
                configurations.
        """
        if not isinstance(operator, DiscreteOperator):
            raise TypeError(
                "Argument to HamiltonianRule must be a valid operator, "
                f"but operator is a {type(operator)}."
            )
        # It might be faster to use Numba versions...
        # if isinstance(operator, DiscreteJaxOperator):
        #    operator = operator.to_numba_operator()
        self.operator = operator

    def init_state(rule, sampler, machine, params, key):
        if sampler.hilbert != rule.operator.hilbert:
            raise ValueError(
                f"""
            The hilbert space of the sampler ({sampler.hilbert}) and the hilbert space
            of the operator ({rule.operator.hilbert}) for HamiltonianRule must be the same.
            """
            )

        return HamiltonianRuleState(
            sections=np.empty(sampler.n_batches, dtype=np.int32)
        )

    def transition(rule, sampler, machine, parameters, state, rng, σ):
        σ = state.σ
        σ1 = state.σ1
        log_prob_corr = state.log_prob_corr

        sections = state.rule_state.sections
        σp = rule.operator.get_conn_flattened(σ, sections)[0]

        rand_vec = rng.uniform(0, 1, size=σ.shape[0])

        _choose(σp, sections, σ1, log_prob_corr, rand_vec)
        if isinstance(rule.operator, DiscreteJaxOperator):
            sections = rule.operator.n_conn(σ1)
        else:
            rule.operator.n_conn(σ1, sections)
        log_prob_corr -= np.log(sections)

    def __repr__(self):
        return f"HamiltonianRuleNumpy({self.operator})"


@jit(nopython=True)
def _choose(states, sections, out, w, rand_vec):
    low_range = 0
    for i, s in enumerate(sections):
        n_rand = low_range + int(np.floor(rand_vec[i] * (s - low_range)))
        out[i] = states[n_rand]
        w[i] = math.log(s - low_range)
        low_range = s
