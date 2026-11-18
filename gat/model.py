import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Any, Callable
from .layer import GATLayer


class GATModel(nn.Module):
    """
    Graph Attention Network model for quantum states.
    Produces a complex-valued output (log-amplitude) for a given spin configuration.
    """

    layers: Sequence[int]
    dtype: Any = jnp.float32
    activation: Callable = nn.elu
    complex_output: bool = True

    @nn.compact
    def __call__(self, x, A):
        """
        Args:
            x: Spin configurations of shape (..., N)
            A: Adjacency matrix of shape (N, N)

        Returns:
            log-wavefunction amplitude of shape (...)
        """
        # Add feature dimension to x: (..., N, 1)
        # We start with the spin values as the initial node features.
        h = jnp.expand_dims(x, axis=-1).astype(self.dtype)

        # Pass through GAT layers
        for out_features in self.layers:
            h = GATLayer(features=out_features, dtype=self.dtype)(h, A)
            h = self.activation(h)

        # Readout: sum over all nodes
        # shape: (..., F)
        h_pooled = jnp.sum(h, axis=-2)

        if self.complex_output:
            # Produce complex scalar via two separate Dense layers for real and imag parts
            # or a single Dense with 2 output channels. Let's use a single dense with 2 outputs.
            out = nn.Dense(2, dtype=self.dtype)(h_pooled)
            # shape: (..., 2)
            log_psi = out[..., 0] + 1j * out[..., 1]
        else:
            out = nn.Dense(1, dtype=self.dtype)(h_pooled)
            log_psi = out[..., 0]

        return log_psi
