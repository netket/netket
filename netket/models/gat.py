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
from flax import linen as nn
from typing import Sequence, Any, Callable
import numpy as np

class GATLayer(nn.Module):
    """
    A single Graph Attention Network (GAT) layer with Multi-Head Attention.
    """
    features: int
    heads: int = 1
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
            Output node features of shape (..., N, features)
        """
        assert self.features % self.heads == 0, "features must be perfectly divisible by heads"
        head_dim = self.features // self.heads

        # W * x -> (..., N, features)
        h = nn.Dense(
            self.features,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            use_bias=False
        )(x)
        
        # Reshape for multi-head: (..., N, heads, head_dim)
        h = h.reshape(*h.shape[:-1], self.heads, head_dim)
        
        # Attention mechanism
        a_src = nn.Dense(1, dtype=self.dtype, kernel_init=self.kernel_init, use_bias=False)(h)
        a_dst = nn.Dense(1, dtype=self.dtype, kernel_init=self.kernel_init, use_bias=False)(h)
        
        # Transpose a_src and a_dst to (..., heads, N, 1)
        a_src_t = jnp.swapaxes(a_src, -3, -2)
        a_dst_t = jnp.swapaxes(a_dst, -3, -2)
        
        # e = a_src + a_dst^T -> shape (..., heads, N, N)
        e = a_src_t + jnp.swapaxes(a_dst_t, -1, -2)
        e = nn.leaky_relu(e, negative_slope=0.2)
        
        # Mask out non-existent edges
        zero_vec = -9e15 * jnp.ones_like(e)
        
        # Expand A to broadcast over batch and heads: (1, ..., 1, N, N)
        # However, jnp.where broadcasts the trailing dimensions naturally if A is (N, N)
        attention = jnp.where(A > 0, e, zero_vec)
        
        # Softmax over neighbors
        attention_weights = nn.softmax(attention, axis=-1)
        
        # h' = \\sum_j \\alpha_{ij} W h_j
        # h is (..., N, heads, head_dim). Transpose to (..., heads, N, head_dim)
        h_t = jnp.swapaxes(h, -3, -2)
        
        # Matmul over last two dims: (..., heads, N, N) x (..., heads, N, head_dim) -> (..., heads, N, head_dim)
        h_prime = jnp.matmul(attention_weights, h_t)
        
        # Transpose back to (..., N, heads, head_dim) and flatten to (..., N, features)
        h_prime = jnp.swapaxes(h_prime, -3, -2).reshape(*x.shape[:-1], self.features)
        
        bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
        h_prime = h_prime + bias
        
        return h_prime


class GAT(nn.Module):
    """
    Graph Attention Network model for quantum states.
    Produces a complex-valued output (log-amplitude) for a given spin configuration.
    """
    n_nodes: int
    edges: tuple
    layers: tuple
    heads: int = 1
    dtype: Any = jnp.float32
    activation: Callable = nn.elu
    complex_output: bool = True

    def setup(self):
        # Build adjacency matrix from graph edges
        n = self.n_nodes
        A = np.zeros((n, n))
        for u, v in self.edges:
            A[u, v] = 1.0
            A[v, u] = 1.0
        # Add self-loops!
        A = A + np.eye(n)
        self.A = jnp.array(A)

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Spin configurations of shape (..., N)
        Returns:
            log-wavefunction amplitude of shape (...)
        """
        # Add feature dimension to x: (..., N, 1)
        h = jnp.expand_dims(x, axis=-1).astype(self.dtype)
        
        # Pass through GAT layers
        for out_features in self.layers:
            h = GATLayer(features=out_features, heads=self.heads, dtype=self.dtype)(h, self.A)
            h = self.activation(h)
        
        # Readout: sum over all nodes
        h_pooled = jnp.sum(h, axis=-2)
        
        if self.complex_output:
            out = nn.Dense(2, dtype=self.dtype)(h_pooled)
            log_psi = out[..., 0] + 1j * out[..., 1]
        else:
            out = nn.Dense(1, dtype=self.dtype)(h_pooled)
            log_psi = out[..., 0]
            
        return log_psi
