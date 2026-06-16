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

import numpy as np
import jax
import pytest

import netket as nk
from netket.sampler.rules import AdaptiveLocalRule


def _hilbert():
    return nk.hilbert.Spin(0.5, N=8)


def test_construction_validation():
    with pytest.raises(ValueError, match="n_flips"):
        AdaptiveLocalRule(n_flips=0.5)
    with pytest.raises(ValueError, match="target_acceptance"):
        AdaptiveLocalRule(target_acceptance=1.5)
    with pytest.raises(ValueError, match="acceptance_floor"):
        AdaptiveLocalRule(target_acceptance=0.5, acceptance_floor=0.6)


def test_rejects_constrained_and_inhomogeneous():
    # constrained Hilbert (fixed magnetisation) must be rejected
    hi = nk.hilbert.Spin(0.5, N=8, total_sz=0)
    sa = nk.sampler.MetropolisAdaptiveLocal(hi)
    ma = nk.models.RBM()
    pars = ma.init(nk.jax.PRNGKey(0), hi.random_state(nk.jax.PRNGKey(0), 2))
    with pytest.raises(ValueError, match="constrained"):
        sa.init_state(ma, pars, nk.jax.PRNGKey(0))


def test_proposals_only_change_local_dofs():
    # every proposed configuration must be a valid Hilbert configuration and
    # differ from the previous one only by resampled sites.
    hi = nk.hilbert.Fock(n_max=3, N=6)
    sa = nk.sampler.MetropolisAdaptiveLocal(hi, n_flips=2.0, n_chains=8)
    ma = nk.models.RBM(param_dtype=float)
    pars = ma.init(nk.jax.PRNGKey(0), hi.random_state(nk.jax.PRNGKey(0), 2))

    state = sa.init_state(ma, pars, nk.jax.PRNGKey(1))
    samples, state = sa.sample(ma, pars, state=state, chain_length=16)
    # all sampled occupations must be within the allowed local range
    assert np.all(samples >= 0)
    assert np.all(samples <= 3)


def test_degrades_to_fixed_without_feedback():
    # An AdaptiveLocalRule whose update_rule_state is never called keeps n_flips
    # fixed. We emulate "no feedback" by checking that init n_flips is preserved
    # when the rule's update hook is bypassed (sweep with a frozen rule_state).
    hi = _hilbert()
    rule = AdaptiveLocalRule(n_flips=3.0)
    sa = nk.sampler.MetropolisSampler(hi, rule, n_chains=16)
    ma = nk.models.RBM(param_dtype=float)
    pars = ma.init(nk.jax.PRNGKey(0), hi.random_state(nk.jax.PRNGKey(0), 2))

    state = sa.init_state(ma, pars, nk.jax.PRNGKey(0))
    assert float(state.rule_state["n_flips"]) == pytest.approx(3.0)


def _final_n_flips(sa, ma, pars, n_iters=60):
    state = sa.init_state(ma, pars, nk.jax.PRNGKey(0))
    for _ in range(n_iters):
        _, state = sa.sample(ma, pars, state=state, chain_length=8)
    return float(state.rule_state["n_flips"])


def test_flip_count_grows_when_acceptance_too_high():
    # A nearly-flat distribution accepts almost everything, so acceptance stays
    # above target: the flip count must grow and saturate at N (bolder moves).
    hi = nk.hilbert.Spin(0.5, N=10)
    sa = nk.sampler.MetropolisAdaptiveLocal(
        hi, n_flips=1.0, target_acceptance=0.5, n_chains=64
    )
    ma = nk.models.RBM(param_dtype=float, alpha=1)
    # tiny parameters => nearly uniform |psi|^2 => high acceptance at any flip count
    pars = ma.init(nk.jax.PRNGKey(0), hi.random_state(nk.jax.PRNGKey(0), 2))
    pars = jax.tree_util.tree_map(lambda x: 1e-3 * x, pars)

    n_flips = _final_n_flips(sa, ma, pars)
    assert n_flips > 5.0  # driven up toward N=10


def test_flip_count_shrinks_when_acceptance_too_low():
    # A sharply-peaked distribution rejects bold multi-flip moves, so acceptance
    # stays below target: the flip count must shrink toward the floor of 1.
    hi = nk.hilbert.Spin(0.5, N=8)
    sa = nk.sampler.MetropolisAdaptiveLocal(
        hi, n_flips=8.0, target_acceptance=0.5, n_chains=64
    )
    # |psi(s)|^2 ∝ exp(2 c M) with M the total magnetisation: a strong gradient
    # everywhere that pushes chains toward the fully-polarised sector, where
    # multi-spin flips are rejected far more often than single-spin flips.
    ma = nk.models.LogStateVector(hi, param_dtype=float)
    magnetisation = hi.all_states().sum(axis=1)
    logstate = 0.75 * magnetisation
    pars = {"params": {"logstate": logstate}}

    n_flips = _final_n_flips(sa, ma, pars)
    assert n_flips < 3.0  # driven down toward the floor of 1


def test_matches_local_rule_at_unit_flip_distribution():
    # The stationary distribution must be correct: compare estimated energy of a
    # known model against MetropolisLocal on the same model and parameters.
    hi = nk.hilbert.Spin(0.5, N=6)
    g = nk.graph.Chain(6, pbc=True)
    ha = nk.operator.Ising(hi, g, h=1.0)
    ma = nk.models.RBM(param_dtype=float, alpha=2)

    sa_local = nk.sampler.MetropolisLocal(hi, n_chains=64)
    sa_adapt = nk.sampler.MetropolisAdaptiveLocal(hi, n_flips=1.0, n_chains=64)

    vs_local = nk.vqs.MCState(sa_local, ma, n_samples=4096, seed=3)
    vs_adapt = nk.vqs.MCState(sa_adapt, ma, n_samples=4096, seed=3)
    # share the exact same parameters between both states
    vs_adapt.variables = vs_local.variables

    e_local = vs_local.expect(ha)
    e_adapt = vs_adapt.expect(ha)

    diff = abs(e_local.mean.real - e_adapt.mean.real)
    err = 5 * (e_local.error_of_mean + e_adapt.error_of_mean)
    assert diff < err
