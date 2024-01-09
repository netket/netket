# Copyright 2023 The NetKet Authors - All rights reserved.
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

from typing import Any, Optional

import jax
import jax.numpy as jnp

from flax import struct
from flax import linen as nn

from netket import config
from netket.hilbert import TensorHilbert
from netket.utils.types import PyTree, PRNGKeyT

# Necessary for the type annotation to work
if config.netket_sphinx_build:
    from netket import sampler

from .base import MetropolisRule


class TensorRule(MetropolisRule):
    r"""A Metropolis sampling rule that can be used to combine different rules acting
    on different subspaces of the same tensor-hilbert space.
    """

    hilbert: TensorHilbert = struct.field(pytree_node=False)
    """The hilbert space upon which this rule is defined. This
    must be a :class:`nk.hilbert.TensorHilbert` with the same
    size as the expected input samples.
    """

    rules: tuple[MetropolisRule, ...]
    """Tuple of rules to be used on every partition of the hilbert
    space.
    """

    def __init__(
        self, hilbert: TensorHilbert, rules: tuple[MetropolisRule, ...]
    ) -> "TensorRule":
        r"""Construct the composition of rules.

        It should be constructed by passing a `TensorHilbert` space as a first argument
        and a list of rules as a second argument. Each `rule[i]` will be used to generate
        a transition for the `i`-th subspace of the tensor hilbert space.

        Args:
            hilbert: The tensor hilbert space on which the rule acts.
            rules: A list of rules, one for each subspace of the tensor hilbert space.
        """
        if not isinstance(hilbert, TensorHilbert):
            raise TypeError(
                "The Hilbert space of a `CombinedRule` must be a TensorHilbert,"
                "which is constructed as a product of different Hilbert spaces."
            )

        if not isinstance(rules, (tuple, list)) or not all(
            isinstance(r, MetropolisRule) for r in rules
        ):
            raise TypeError(
                "The second argument (rules) must be a tuple of `MetropolisRule` "
                f"rules, but you have passed {type(rules)}."
            )

        if len(hilbert.subspaces) != len(rules):
            raise ValueError(
                "Length mismatch between the rules and the hilbert space: Hilbert "
                f"has {len(hilbert.subspaces)} subpsaces, but you specified {len(rules)}."
            )

        self.hilbert = hilbert
        self.rules = tuple(rules)

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Optional[Any]:
        N = self.hilbert._n_hilbert_spaces
        keys = jax.random.split(key, N)
        return tuple(
            self.rules[i].init_state(
                sampler.replace(hilbert=self.hilbert.subspaces[i]),
                machine,
                params,
                keys[i],
            )
            for i in range(N)
        )

    def reset(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
    ) -> Optional[Any]:
        rule_states = []
        for i in range(self.hilbert._n_hilbert_spaces):
            # construct temporary sampler and rule state with correct sub-hilbert and
            # sampler-state objects.
            _sampler = sampler.replace(hilbert=self.hilbert.subspaces[i])
            _state = sampler_state.replace(rule_state=sampler_state.rule_state[i])

            rule_states.append(self.rules[i].reset(_sampler, machine, params, _state))
        return tuple(rule_states)

    def transition(self, sampler, machine, parameters, state, key, σ):
        keys = jax.random.split(key, self.hilbert._n_hilbert_spaces)

        σps = []
        log_prob_corr = []
        for i in range(self.hilbert._n_hilbert_spaces):
            σ_i = σ[..., self.hilbert._cum_indices[i] : self.hilbert._cum_sizes[i]]

            # construct temporary sampler and rule state with correct sub-hilbert and
            # sampler-state objects.
            _sampler = sampler.replace(hilbert=self.hilbert.subspaces[i])
            _state = state.replace(rule_state=state.rule_state[i])

            σps_i, log_prob_corr_i = self.rules[i].transition(
                _sampler, machine, parameters, _state, keys[i], σ_i
            )

            σps.append(σps_i)
            if log_prob_corr_i is not None:
                log_prob_corr.append(log_prob_corr_i)

        σp = jnp.concatenate(σps, axis=-1)
        log_prob_corr = sum(log_prob_corr) if len(log_prob_corr) > 0 else None
        return σp, log_prob_corr

    def __repr__(self):
        return "TensorRule(hilbert={self.hilbert}, rules={self.rules})"
