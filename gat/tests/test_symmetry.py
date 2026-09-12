import jax
import jax.numpy as jnp
import numpy as np
from gat.model import GATModel
from gat.layer import GATLayer

def test_layer_equivariance():
    rng = jax.random.PRNGKey(0)
    N = 10
    F_in = 4
    F_out = 8
    
    layer = GATLayer(features=F_out)
    
    # Random node features
    x = jax.random.normal(rng, (N, F_in))
    
    # Random adjacency matrix (symmetric)
    A = jax.random.uniform(rng, (N, N))
    A = A + A.T
    A = jnp.where(A > 1.0, 1.0, 0.0)
    
    # Initialize variables
    variables = layer.init(rng, x, A)
    
    # Output for original graph
    out_orig = layer.apply(variables, x, A)
    
    # Random permutation
    P_idx = np.random.permutation(N)
    P = np.eye(N)[P_idx]
    
    # Permute input
    x_perm = x[P_idx]
    A_perm = A[P_idx][:, P_idx]
    
    # Output for permuted graph
    out_perm = layer.apply(variables, x_perm, A_perm)
    
    # The output of the permuted graph should equal the permuted output of the original graph
    np.testing.assert_allclose(out_perm, out_orig[P_idx], atol=1e-5)

def test_model_invariance():
    rng = jax.random.PRNGKey(1)
    N = 10
    
    model = GATModel(layers=[8, 8])
    
    # Spin config (e.g. +1 or -1)
    x = jax.random.choice(rng, jnp.array([-1.0, 1.0]), shape=(2, N))
    
    A = jax.random.uniform(rng, (N, N))
    A = A + A.T
    A = jnp.where(A > 1.0, 1.0, 0.0)
    
    variables = model.init(rng, x, A)
    
    # Output for original
    out_orig = model.apply(variables, x, A)
    
    # Permute input
    P_idx = np.random.permutation(N)
    x_perm = x[:, P_idx]
    A_perm = A[P_idx][:, P_idx]
    
    # Output for permuted
    out_perm = model.apply(variables, x_perm, A_perm)
    
    # Model should be invariant to node permutations since it pools over all nodes
    np.testing.assert_allclose(out_perm, out_orig, atol=1e-5)
