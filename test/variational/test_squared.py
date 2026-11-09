import jax
import jax.numpy as jnp
import numpy as np
import pytest

import netket as nk


@pytest.fixture(params=[nk.operator.LocalOperatorNumba, nk.operator.LocalOperatorJax])
def operator_cls(request):
    if nk.config.netket_sharding and jax.device_count() > 1:
        if request.param is nk.operator.LocalOperatorNumba:
            pytest.skip("Numba operators do not support sharding")
    return request.param


@pytest.mark.parametrize("chunk_size", [None, 16])
@pytest.mark.parametrize("param_dtype", [float, complex])
@pytest.mark.parametrize("visible_bias", [0.0, np.log(2.0)])
def test_squared_changes_sector(operator_cls, chunk_size, param_dtype, visible_bias):
    hi = nk.hilbert.Fock(n_max=2, N=1, n_particles=1)
    vs = nk.vqs.MCState(
        nk.sampler.ExactSampler(hi),
        nk.models.RBM(alpha=1, param_dtype=param_dtype),
        n_samples=128,
        chunk_size=chunk_size,
        seed=0,
    )
    parameters = jax.tree_util.tree_map(jnp.zeros_like, vs.parameters)
    parameters["visible_bias"] = jnp.full_like(parameters["visible_bias"], visible_bias)
    vs.parameters = parameters
    a = nk.operator.boson.destroy(hi, 0, dtype=param_dtype, cls=operator_cls)
    squared = a.H @ a

    # Both parameter choices represent |1>, regardless of the amplitudes at |0>, |2>.
    np.testing.assert_allclose(vs.to_array(), [1.0])
    np.testing.assert_allclose(vs.expect(squared.collect()).mean, 1.0)
    np.testing.assert_allclose(vs.expect(squared).mean, 1.0)
    np.testing.assert_allclose(vs.local_estimators(squared).data, 1.0)

    stats, gradient = vs.expect_and_grad(squared)
    np.testing.assert_allclose(stats.mean, 1.0)
    for leaf in jax.tree_util.tree_leaves(gradient):
        np.testing.assert_allclose(leaf, 0.0, atol=1e-12)


@pytest.mark.parametrize("chunk_size", [None, 16])
def test_squared_preserves_sector(operator_cls, chunk_size):
    hi = nk.hilbert.Fock(n_max=1, N=2, n_particles=1)
    vs = nk.vqs.MCState(
        nk.sampler.ExactSampler(hi),
        nk.models.RBM(alpha=1),
        n_samples=128,
        chunk_size=chunk_size,
        seed=0,
    )
    a0 = nk.operator.boson.destroy(hi, 0, cls=operator_cls)
    a1 = nk.operator.boson.destroy(hi, 1, cls=operator_cls)
    parent = a0.H.collect() @ a1 + 0.7 * nk.operator.boson.number(
        hi, 0, cls=operator_cls
    )
    squared = parent.H @ parent
    collected = squared.collect()

    np.testing.assert_allclose(
        vs.local_estimators(squared).data, vs.local_estimators(collected).data
    )
    stats, gradient = vs.expect_and_grad(squared)
    reference, reference_gradient = vs.expect_and_grad(collected)
    np.testing.assert_allclose(stats.mean, reference.mean)
    jax.tree_util.tree_map(np.testing.assert_allclose, gradient, reference_gradient)


@pytest.mark.parametrize("chunk_size", [None, 16])
def test_squared_unconstrained(operator_cls, chunk_size):
    hi = nk.hilbert.Fock(n_max=2, N=1)
    vs = nk.vqs.MCState(
        nk.sampler.ExactSampler(hi),
        nk.models.RBM(alpha=1),
        n_samples=128,
        chunk_size=chunk_size,
        seed=0,
    )
    number = nk.operator.boson.number(hi, 0, cls=operator_cls)
    squared = nk.operator.Squared(number)
    expected = vs.samples[..., 0] ** 2
    np.testing.assert_allclose(vs.local_estimators(squared).data, expected)
    np.testing.assert_allclose(vs.expect(squared).mean, expected.mean())
