import pytest

import netket as nk
import netket.experimental as nkx
import numpy as np


from .renyi2_exact import _renyi2_exact

N = 3
hi = nk.hilbert.Spin(0.5, N)


def _setup():
    n_samples = 1e4
    n_discard_per_chain = 1e3

    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
    ma = nk.models.RBM(alpha=1)

    vs = nk.vqs.MCState(
        sampler=sa,
        model=ma,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
    )

    vs_exact = nk.vqs.FullSumState(
        hilbert=hi,
        model=ma,
    )

    subsys = [0, 1]
    S2 = nkx.observable.Renyi2EntanglementEntropy(hi, subsys)

    return vs, vs_exact, S2, subsys


def test_MCState():
    vs, vs_exact, S2, subsys = _setup()
    S2_stats = vs.expect(S2)
    S2_exact = _renyi2_exact(vs, subsys)

    S2_mean = S2_stats.mean
    err = 3 * S2_stats.error_of_mean

    np.testing.assert_allclose(S2_exact, S2_mean.real, atol=err)


def test_FullSumState():
    vs, vs_exact, S2, subsys = _setup()
    S2_stats = vs_exact.expect(S2)
    S2_exact = _renyi2_exact(vs_exact, subsys)

    S2_mean = S2_stats.mean
    err = 1e-12

    np.testing.assert_allclose(S2_exact, S2_mean.real, atol=err)
