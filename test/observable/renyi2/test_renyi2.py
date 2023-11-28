import netket as nk
import netket.experimental as nkx
import numpy as np

import pytest


from .renyi2_exact import _renyi2_exact


def _setup(useExactSampler=True):
    N = 3
    hi = nk.hilbert.Spin(0.5, N)

    ma = nk.models.RBM(alpha=1)
    n_samples = 1e4

    if useExactSampler:
        sa = nk.sampler.ExactSampler(hilbert=hi)
        vs = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
        )

    else:
        n_discard_per_chain = 1e3

        sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
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


@pytest.mark.parametrize(
    "useExactSampler",
    [
        pytest.param(True, id="ExactSampler"),
        pytest.param(False, id="MetropolisSampler"),
    ],
)
def test_MCState(useExactSampler):
    pytest.importorskip("qutip")

    vs, vs_exact, S2, subsys = _setup(useExactSampler)
    S2_stats = vs.expect(S2)
    S2_exact = _renyi2_exact(vs, subsys)

    S2_mean = S2_stats.mean
    err = 3 * S2_stats.error_of_mean

    np.testing.assert_allclose(S2_exact, S2_mean.real, atol=err)


def test_FullSumState():
    pytest.importorskip("qutip")

    vs, vs_exact, S2, subsys = _setup()
    S2_stats = vs_exact.expect(S2)
    S2_exact = _renyi2_exact(vs_exact, subsys)

    S2_mean = S2_stats.mean
    err = 1e-12

    np.testing.assert_allclose(S2_exact, S2_mean.real, atol=err)


def test_continuous():
    pytest.importorskip("qutip")

    N = 3
    hi = nk.hilbert.Particle(N, L=0, pbc=True)
    subsys = [0, 1]

    with pytest.raises(TypeError):
        nkx.observable.Renyi2EntanglementEntropy(hi, subsys)

    hi = nk.hilbert.Fock(N=5, n_particles=3)

    with pytest.raises(ValueError):
        nkx.observable.Renyi2EntanglementEntropy(hi, subsys)


def test_invalid_partition():
    pytest.importorskip("qutip")

    N = 3
    hi = nk.hilbert.Spin(0.5, N)
    subsys = [-1, 0]

    with pytest.raises(ValueError):
        nkx.observable.Renyi2EntanglementEntropy(hi, subsys)

    subsys = [0, 1, 2, 3]

    with pytest.raises(ValueError):
        nkx.observable.Renyi2EntanglementEntropy(hi, subsys)


@pytest.mark.skipif(
    nk.config.netket_experimental_sharding, reason="Only run without sharding"
)
def test_oddchains():
    pytest.importorskip("qutip")

    vs, vs_exact, S2, subsys = _setup()

    N = 3
    hi = nk.hilbert.Spin(0.5, N)
    subsys = [0, 1]

    vs.sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=3)
    S2 = nkx.observable.Renyi2EntanglementEntropy(hi, subsys)

    with pytest.raises(ValueError):
        vs.expect(S2)
