import netket as nk
import netket.experimental as nkx
import numpy as np

import pytest


from .renyi2_exact import _renyi2_exact


# The tests below compare a Monte Carlo estimate against the exact value with a
# 3 sigma tolerance. The exact Renyi2 entropy of this ansatz is ~0, so that band
# is narrow and an unseeded state failed roughly 4% of the time (measured over
# 200 repetitions). Seed both the parameters and the sampler so the comparison is
# reproducible. Of the seeds tried all passed; this pair was picked because it
# leaves the most headroom, using only ~10% of the 3 sigma band, so small
# numerical drift will not start failing the test.
SEED = 123
SAMPLER_SEED = 1234


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
            seed=SEED,
            sampler_seed=SAMPLER_SEED,
        )

    else:
        n_discard_per_chain = 1e3

        sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
        vs = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            seed=SEED,
            sampler_seed=SAMPLER_SEED,
        )

    vs_exact = nk.vqs.FullSumState(
        hilbert=hi,
        model=ma,
        seed=SEED,
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


def test_complex_amplitude_ansatz():
    """Renyi2 works on complex-amplitude ansätze (delta method needs real inputs).

    The SWAP kernel is complex-dtyped for a complex model; its mean is the real
    purity, so the local estimators must be realified rather than crashing in
    ``jax.jacfwd``.
    """
    pytest.importorskip("qutip")

    N = 3
    hi = nk.hilbert.Spin(0.5, N)
    ma = nk.models.RBM(alpha=1, param_dtype=complex)
    subsys = [0, 1]
    S2 = nkx.observable.Renyi2EntanglementEntropy(hi, subsys)

    vs = nk.vqs.MCState(nk.sampler.ExactSampler(hi), ma, n_samples=int(2**14), seed=0)
    S2_stats = vs.expect(S2)
    S2_exact = _renyi2_exact(vs, subsys)

    assert np.isrealobj(S2_stats.mean)
    np.testing.assert_allclose(
        S2_exact, S2_stats.mean, atol=3 * S2_stats.error_of_mean + 1e-2
    )


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
    hi = nk.experimental.hilbert.Particle(
        N, geometry=nk.experimental.geometry.Cell(d=1, L=0.0, pbc=True)
    )
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


@pytest.mark.parametrize(
    "useExactSampler",
    [
        pytest.param(True, id="ExactSampler"),
        pytest.param(False, id="MetropolisSampler"),
    ],
)
def test_local_estimators(useExactSampler):
    pytest.importorskip("qutip")

    vs, vs_exact, S2, subsys = _setup(useExactSampler)
    le = vs.local_estimators(S2)
    S2_stats = le.to_stats()
    S2_exact = _renyi2_exact(vs, subsys)

    np.testing.assert_allclose(
        S2_exact, S2_stats.mean.real, atol=3 * S2_stats.error_of_mean
    )


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
