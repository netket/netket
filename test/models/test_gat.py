import pytest
import jax
import jax.numpy as jnp
import numpy as np

from netket.models.gat import GATLayer, GAT


def test_gat_layer():
    rng = jax.random.PRNGKey(0)
    N = 5
    F_in = 4
    F_out = 8
    heads = 2

    layer = GATLayer(features=F_out, heads=heads)

    x = jax.random.normal(rng, (N, F_in))

    # Adjacency matrix
    A = jnp.ones((N, N))

    variables = layer.init(rng, x, A)
    out = layer.apply(variables, x, A)

    assert out.shape == (N, F_out)


def test_gat_layer_equivariance():
    rng = jax.random.PRNGKey(0)
    N = 10
    F_in = 4
    F_out = 8

    layer = GATLayer(features=F_out, heads=2)

    x = jax.random.normal(rng, (N, F_in))
    A = jax.random.uniform(rng, (N, N))
    A = A + A.T
    A = jnp.where(A > 1.0, 1.0, 0.0)

    variables = layer.init(rng, x, A)
    out_orig = layer.apply(variables, x, A)

    P_idx = np.random.permutation(N)
    x_perm = x[P_idx]
    A_perm = A[P_idx][:, P_idx]

    out_perm = layer.apply(variables, x_perm, A_perm)

    np.testing.assert_allclose(out_perm, out_orig[P_idx], atol=1e-5)


@pytest.mark.parametrize("complex_output", [True, False])
def test_gat_model(complex_output):
    rng = jax.random.PRNGKey(1)
    N = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

    model = GAT(
        n_nodes=N, edges=edges, layers=(16, 8), heads=2, complex_output=complex_output
    )

    # Batch size 3
    x = jax.random.choice(rng, jnp.array([-1.0, 1.0]), shape=(3, N))

    variables = model.init(rng, x)
    out = model.apply(variables, x)

    assert out.shape == (3,)
    if complex_output:
        assert out.dtype == jnp.complex64 or out.dtype == jnp.complex128
    else:
        assert not jnp.iscomplexobj(out)
