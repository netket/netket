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
import abc

from flax import linen as nn
from jax import numpy as jnp

from netket.utils.types import PyTree, PRNGKeyT

from netket.utils import struct
from netket import config

# Necessary for the type annotation to work
if config.netket_sphinx_build:
    from netket import sampler


class MetropolisRule(struct.Pytree):
    """
    Base class for transition rules of Metropolis, such as Local, Exchange, Hamiltonian
    and several others.
    """

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Any | None:
        """
        Initialises the optional internal state of the Metropolis sampler transition
        rule.

        The provided key is unique and does not need to be split.

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
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
    ) -> Any | None:
        """
        Resets the internal state of the Metropolis Sampler Transition Rule.

        The default implementation returns the current rule_state without modifying it.

        Arguments:
            sampler: The Metropolis sampler.
            machine: A Flax module with the forward pass of the log-pdf.
            params: The PyTree of parameters of the model.
            sampler_state: The current state of the sampler. Should not modify it.

        Returns:
           A reset state of the rule. This returns the same type of
           :py:meth:`~nk.sampler.rule.MetropolisRule.rule_state` and might be `None`.
        """
        return sampler_state.rule_state

    @abc.abstractmethod
    def transition(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
        key: PRNGKeyT,
        σ: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
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

    def update_rule_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
        accepted: jnp.ndarray,
    ) -> Any | None:
        """
        Updates the internal ``rule_state`` given the accept/reject outcome of the
        Metropolis sub-step that just completed.

        This is the post-acceptance dual of :meth:`transition`: ``transition``
        proposes a new configuration *before* the accept/reject decision, while
        this hook is invoked *after* it, with the per-chain acceptance mask. It
        lets a rule tune itself toward a target acceptance (e.g. adapting a
        proposal width or flip count) without subclassing the sampler.

        The default implementation is a no-op that returns the current
        ``rule_state`` unchanged. Rules that do not adapt need not override it
        and pay no runtime cost: the sampler skips the call entirely unless the
        rule overrides this method.

        This is called once per Metropolis sub-step, inside the sampler's
        ``fori_loop``, so the returned value must keep the same PyTree structure
        (and leaf dtypes/shapes) as ``sampler_state.rule_state``.

        Arguments:
            sampler: The Metropolis sampler.
            machine: A Flax module with the forward pass of the log-pdf.
            params: The PyTree of parameters of the model.
            sampler_state: The state of the sampler *before* this update. Read
                its ``rule_state`` to compute the new one; do not mutate it.
            accepted: Boolean array of shape ``(n_chains,)`` flagging which
                chains accepted their proposal in the current sub-step.

        Returns:
            The new ``rule_state``, with the same PyTree structure as
            ``sampler_state.rule_state``.
        """
        return sampler_state.rule_state

    def random_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
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
