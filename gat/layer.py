import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable

class GATLayer(nn.Module):
    """
    A single Graph Attention Network (GAT) layer.
    """
    features: int
    dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x, A):
        """
        Args:
            x: Node feature matrix of shape (..., N, F_in)
            A: Adjacency matrix of shape (N, N)
        Returns:
            Output node features of shape (..., N, F_out)
        """
        # x is expected to be real-valued internally
        # W * x
        h = nn.Dense(
            self.features,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            use_bias=False
        )(x)
        
        # Attention mechanism
        # a(Wh_i, Wh_j)
        a_src = nn.Dense(1, dtype=self.dtype, kernel_init=self.kernel_init, use_bias=False)(h)
        a_dst = nn.Dense(1, dtype=self.dtype, kernel_init=self.kernel_init, use_bias=False)(h)
        
        # e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
        e = a_src + jnp.swapaxes(a_dst, -1, -2)
        e = nn.leaky_relu(e, negative_slope=0.2)
        
        # Mask out non-existent edges
        zero_vec = -9e15 * jnp.ones_like(e)
        attention = jnp.where(A > 0, e, zero_vec)
        
        # Softmax over neighbors
        attention_weights = nn.softmax(attention, axis=-1)
        
        # h' = \sum_j \alpha_{ij} W h_j
        h_prime = jnp.matmul(attention_weights, h)
        
        bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
        h_prime = h_prime + bias
        
        return h_prime
