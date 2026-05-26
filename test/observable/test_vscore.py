# Copyright 2026 The NetKet Authors - All rights reserved.
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

import netket as nk
import numpy as np

import pytest


seed = 123


def _setup(use_exact_sampler=True):
    N = 3
    hi = nk.hilbert.Spin(0.5, N)

    ma = nk.models.RBM(alpha=1)
    n_samples = 8192

    if use_exact_sampler:
        sa = nk.sampler.ExactSampler(hilbert=hi)
        vs = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            seed=seed,
        )
    else:
        sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
        vs = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            n_discard_per_chain=1e3,
            seed=seed,
        )

    vs_exact = nk.vqs.FullSumState(
        hilbert=hi,
        model=ma,
        seed=seed,
    )

    H = nk.operator.IsingJax(hi, graph=nk.graph.Chain(N), h=1, J=-1)
    H2 = H @ H

    return vs, vs_exact, H, H2


def vscore_exact_fun(params, vs, H, H2, *, trace_diagonal):
    params_old = vs.parameters
    vs.parameters = params
    state = vs.to_array()
    vs.parameters = params_old

    e_mean = (state.conj() @ (H @ state)).real
    e2_mean = (state.conj() @ (H2 @ state)).real
    var = e2_mean - e_mean**2
    return var / (e_mean - trace_diagonal) ** 2


@pytest.mark.parametrize(
    "use_exact_sampler",
    [
        pytest.param(True, id="ExactSampler"),
        pytest.param(False, id="MetropolisSampler"),
    ],
)
@pytest.mark.parametrize(
    "trace_diagonal",
    [
        pytest.param(0.0, id="trace=0"),
        pytest.param(-1.5, id="trace=-1.5"),
    ],
)
def test_MCState_expect(use_exact_sampler, trace_diagonal):
    vs, vs_exact, H, H2 = _setup(use_exact_sampler)
    vscore_op = nk.observable.VScore(H, trace_diagonal=trace_diagonal)

    vscore_stats = vs.expect(vscore_op)
    vscore_exact = vscore_exact_fun(
        vs.parameters,
        vs_exact,
        H,
        H2,
        trace_diagonal=trace_diagonal,
    )

    err = float(vscore_stats.error_of_mean)
    atol = 5 * err if err > 0 else 0.05

    np.testing.assert_allclose(
        vscore_exact.real,
        float(vscore_stats.mean.real),
        atol=atol,
    )


@pytest.mark.parametrize(
    "trace_diagonal",
    [
        pytest.param(0.0, id="trace=0"),
        pytest.param(-1.5, id="trace=-1.5"),
    ],
)
def test_FullSumState(trace_diagonal):
    _, vs_exact, H, H2 = _setup()
    vscore_op = nk.observable.VScore(H, trace_diagonal=trace_diagonal)

    vscore_stats = vs_exact.expect(vscore_op)
    vscore_exact = vscore_exact_fun(
        vs_exact.parameters,
        vs_exact,
        H,
        H2,
        trace_diagonal=trace_diagonal,
    )

    np.testing.assert_allclose(
        vscore_exact.real,
        float(vscore_stats.mean.real),
        atol=1e-6,
    )
    np.testing.assert_equal(float(vscore_stats.error_of_mean), 0.0)


def test_expect_to_precision():
    vs, _, H, _ = _setup(use_exact_sampler=False)
    vscore_op = nk.observable.VScore(H, trace_diagonal=-1.5)

    stats = vs.expect_to_precision(vscore_op, atol=1.0e6, max_iter=2, verbose=False)
    s = stats.get_stats()

    assert np.isfinite(float(s.mean))
    assert np.isfinite(float(s.error_of_mean))


def test_trace_diagonal_validation():
    _, _, H, _ = _setup()

    # The constructor requires an explicit real-valued diagonal shift.
    with pytest.raises(TypeError, match="trace_diagonal"):
        nk.observable.VScore(H)

    with pytest.raises(TypeError, match="trace_diagonal"):
        nk.observable.VScore(H, trace_diagonal=1j)


def test_hilbert_mismatch():
    hi1 = nk.hilbert.Spin(0.5, 3)
    hi2 = nk.hilbert.Spin(0.5, 4)
    H = nk.operator.IsingJax(hi1, graph=nk.graph.Chain(3), h=1)
    vs = nk.vqs.FullSumState(hilbert=hi2, model=nk.models.RBM(alpha=1), seed=seed)

    with pytest.raises(TypeError, match="Hilbert"):
        vs.expect(nk.observable.VScore(H, trace_diagonal=0.0))


def test_repr():
    _, _, H, _ = _setup()
    vscore_op = nk.observable.VScore(H, trace_diagonal=-1.0)

    assert "VScore" in repr(vscore_op)
    assert "-1.0" in repr(vscore_op)
