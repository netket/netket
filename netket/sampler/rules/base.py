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

from typing import Any, Optional, Tuple
import abc

from flax import linen as nn
from jax import numpy as jnp

from netket.utils.types import PyTree, PRNGKeyT

from netket.utils import struct


@struct.dataclass
class MetropolisRule(abc.ABC):
    """
    Base class for transition rules of Metropolis, such as Local, Exchange, Hamiltonian
    and several others.
    """

    def init_state(
        self,
        sampler: "MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Optional[Any]:
        """
        Initialises the optional internal state of the Metropolis sampler transition
        rule.

        The provided key is unique and does not need to be splitted.

        It should return an immutable data structure.

        Arguments:
            sampler: The Metropolis sampler.
            machine: A Flax module with the forward pass of the log-pdf.
            params: The PyTree of parameters of the model.
            key: A Jax PRNGKey.

        Returns:
            An optional state.
        """
        return None

    def reset(
        self,
        sampler: "MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "SamplerState",  # noqa: F821
    ) -> Optional[Any]:
        """
        Resets the internal state of the Metropolis Sampler Transition Rule.

        The default implementation returns the current rule_state without modofying it.

        Arguments:
            sampler: The Metropolis sampler.
            machine: A Flax module with the forward pass of the log-pdf.
            params: The PyTree of parameters of the model.
            sampler_state: The current state of the sampler. Should not modify it.

        Returns:
           A resetted, state of the rule. This returns the same type of
           :py:meth:`~nk.sampler.rule.MetropolisRule.rule_state` and might be `None`.
        """
        return sampler_state.rule_state

    @abc.abstractmethod
    def transition(
        self,
        sampler: "MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "SamplerState",  # noqa: F821
        key: PRNGKeyT,
        σ: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        r"""
        Proposes a new configuration set of configurations $\sigma'$ starting from the current
        chain configurations :math:`\sigma`.

        The new configurations :math:`\sigma'` should be a matrix with the same dimension as
        :math:`\sigma`.

        This function should return a tuple. where the first element are the new configurations
        $\sigma'$ and the second element is either `None` or an array of length `σ.shape[0]`
        containing an optional log-correction factor. The correction factor should be non-zero
        when the transition rule is non-symmetrical.

        Arguments:
            sampler: The Metropolis sampler.
            machine: A Flax module with the forward pass of the log-pdf.
            params: The PyTree of parameters of the model.
            sampler_state: The current state of the sampler. Should not modify it.
            key: A Jax PRNGKey to use to generate new random configurations.
            σ: The current configurations stored in a 2D matrix.

        Returns:
           A tuple containing the new configurations :math:`\sigma'` and the optional vector of
           log corrections to the transition probability.
        """
        pass

    def random_state(
        self,
        sampler: "MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "SamplerState",  # noqa: F821
        key: PRNGKeyT,
    ):
        """
        Generates a random state compatible with this rule.

        By default this calls :func:`netket.hilbert.random.random_state`.

        Arguments:
            sampler: The Metropolis sampler.
            machine: A Flax module with the forward pass of the log-pdf.
            params: The PyTree of parameters of the model.
            sampler_state: The current state of the sampler. Should not modify it.
            key: The PRNGKey to use to generate the random state.
        """
        return sampler.hilbert.random_state(
            key, size=sampler.n_batches, dtype=sampler.dtype
        )
