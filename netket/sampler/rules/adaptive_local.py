# Copyright 2024 The NetKet Authors - All rights reserved.
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

import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.hilbert import HomogeneousHilbert

from .base import MetropolisRule


class AdaptiveLocalRule(MetropolisRule):
    r"""A multi-site local rule whose flip count adapts to a target acceptance.

    This is the discrete-lattice analogue of
    :class:`~netket.sampler.rules.GaussianRule` driven toward a target
    acceptance: where the Gaussian rule adapts a continuous proposal *width*,
    this rule adapts how *many* local degrees of freedom it disturbs per step.

    Like :class:`~netket.sampler.rules.LocalRule` it proposes a new
    configuration by resampling local degrees of freedom, but instead of
    touching exactly one site it visits each of the :math:`N` sites
    independently with probability :math:`p = \lambda / N` and resamples every
    visited site to a different local value. The expected number of resampled
    sites :math:`\lambda` lives in the sampler ``rule_state`` and is rescaled
    after every Metropolis sub-step using the *oneqmc* scheme,

    .. math::

       \lambda \leftarrow \mathrm{clip}\!\left(
       \lambda \cdot \frac{\max(a,\, a_\mathrm{floor})}{a^\star},\; 1,\; N\right),

    where :math:`a` is the batch acceptance of the sub-step, :math:`a^\star` is
    :attr:`target_acceptance` and :math:`a_\mathrm{floor}` is
    :attr:`acceptance_floor`. Above-target acceptance grows :math:`\lambda`
    (bolder, more-site moves); below-target shrinks it back toward single-site
    flips. The :math:`[1, N]` clamp is the discrete counterpart of the Gaussian
    rule keeping its width positive.

    Because each visited site is resampled uniformly among the other
    :math:`m - 1` local values and the per-site visit probability depends only
    on the number of changed sites, the proposal is symmetric and carries no
    log-probability correction. A sub-step may visit zero sites (probability
    :math:`(1 - p)^N`), proposing the identity move; this is accepted trivially
    and leaves detailed balance intact.

    This rule benefits from per-step acceptance feedback through
    :meth:`update_rule_state`, which the standard
    :class:`~netket.sampler.MetropolisSampler` provides automatically. Driven by
    a sampler that does not feed acceptance back it degrades gracefully to a
    fixed multi-flip rule at the initial ``n_flips``.

    Only unconstrained :class:`~netket.hilbert.HomogeneousHilbert` spaces (e.g.
    :class:`~netket.hilbert.Spin`, :class:`~netket.hilbert.Qubit`,
    :class:`~netket.hilbert.Fock`) are supported, since independent per-site
    resampling does not preserve constraints such as fixed magnetisation.
    """

    n_flips0: float = struct.field(pytree_node=False)
    """Initial expected number of resampled sites per Metropolis sub-step."""
    target_acceptance: float = struct.field(pytree_node=False)
    """Acceptance rate the flip count is driven towards."""
    acceptance_floor: float = struct.field(pytree_node=False)
    """Lower clamp on the measured acceptance used in the rescaling factor."""

    def __init__(
        self,
        n_flips: float = 1.0,
        *,
        target_acceptance: float = 0.5,
        acceptance_floor: float = 0.05,
    ) -> None:
        """Construct the rule.

        Args:
            n_flips: Initial expected number of resampled sites per sub-step.
                Must be ``>= 1``. ``1.0`` recovers (in expectation) a single-site
                update like :class:`~netket.sampler.rules.LocalRule`.
            target_acceptance: Acceptance rate the flip count is driven towards.
                The default ``0.5`` is the usual rule-of-thumb for discrete
                Metropolis updates (unlike the ``0.57`` Roberts--Rosenthal value
                appropriate to continuous random-walk proposals).
            acceptance_floor: Lower clamp on the measured acceptance entering the
                rescaling factor; must lie in ``(0, target_acceptance]``.
        """
        if n_flips < 1.0:
            raise ValueError("n_flips must be >= 1")
        if not 0.0 < target_acceptance < 1.0:
            raise ValueError("target_acceptance must lie in (0, 1)")
        if not 0.0 < acceptance_floor <= target_acceptance:
            raise ValueError("acceptance_floor must lie in (0, target_acceptance]")
        self.n_flips0 = float(n_flips)
        self.target_acceptance = float(target_acceptance)
        self.acceptance_floor = float(acceptance_floor)

    def init_state(self, sampler, machine, parameters, key):
        hilb = sampler.hilbert
        if not isinstance(hilb, HomogeneousHilbert):
            raise TypeError(
                "AdaptiveLocalRule only supports HomogeneousHilbert spaces, "
                f"got {type(hilb).__name__}."
            )
        if hilb.constrained:
            raise ValueError(
                "AdaptiveLocalRule does not support constrained Hilbert spaces, "
                "as independent per-site resampling breaks the constraint. "
                "Use an ExchangeRule-based sampler instead."
            )
        return {"n_flips": jnp.asarray(self.n_flips0, dtype=float)}

    def transition(self, sampler, machine, parameters, state, key, σ):
        hilb = sampler.hilbert
        m = hilb.local_size
        N = σ.shape[-1]

        key_mask, key_vals = jax.random.split(key)

        # Per-site visit probability, clamped to [1/N, 1].
        n_flips = jnp.asarray(state.rule_state["n_flips"], dtype=float)
        p = jnp.clip(n_flips / N, 1.0 / N, 1.0)
        mask = jax.random.bernoulli(key_mask, p=p, shape=σ.shape)

        σ_resampled = _resample_all(hilb, key_vals, σ, m)
        σp = jnp.where(mask, σ_resampled, σ)

        return σp.astype(σ.dtype), None

    def update_rule_state(self, sampler, machine, parameters, state, accepted):
        n_flips = state.rule_state["n_flips"]
        N = sampler.hilbert.size
        acc = jnp.mean(accepted.astype(n_flips.dtype))
        floor = jnp.asarray(self.acceptance_floor, dtype=n_flips.dtype)
        factor = jnp.maximum(acc, floor) / self.target_acceptance
        return {"n_flips": jnp.clip(n_flips * factor, 1.0, float(N))}

    def __repr__(self):
        return (
            f"AdaptiveLocalRule(n_flips={self.n_flips0}, "
            f"target_acceptance={self.target_acceptance}, "
            f"acceptance_floor={self.acceptance_floor})"
        )


def _resample_all(hilb, key, σ, m):
    """Resample every site of ``σ`` to a uniformly-chosen different local value.

    Returns a configuration ``σ'`` with ``σ'[i] != σ[i]`` at every site. The
    caller selects which of these resampled sites to keep via a mask, so the
    overall proposal only changes the masked subset.
    """
    x_old = hilb.states_to_local_indices(σ)
    if m == 2:
        x_new = 1 - x_old
    else:
        # offset in {0, ..., m-2}; (x_old + 1 + offset) % m is uniform over the
        # m-1 indices different from x_old.
        offset = jax.random.randint(key, shape=σ.shape, minval=0, maxval=m - 1)
        x_new = (x_old + 1 + offset) % m
    return hilb.local_indices_to_states(x_new.astype(x_old.dtype), dtype=σ.dtype)
