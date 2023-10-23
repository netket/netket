# Copyright 2022 The NetKet Authors - All rights reserved.
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

from typing import Callable, Optional

import numpy as np
from flax import linen as nn
from flax.linen.dtypes import promote_dtype

from jax import lax
from jax import numpy as jnp
from jax.nn.initializers import orthogonal, zeros

from netket.utils import HashableArray
from netket.utils.types import Array, DType, NNInitFunc

default_kernel_init = orthogonal()


def _check_reorder_idx(
    reorder_idx: Optional[HashableArray],
    inv_reorder_idx: Optional[HashableArray],
    prev_neighbors: Optional[HashableArray],
):
    if reorder_idx is None and inv_reorder_idx is None and prev_neighbors is None:
        # There is a faster code path for 1D RNN
        return

    if reorder_idx is None or inv_reorder_idx is None or prev_neighbors is None:
        raise ValueError(
            "`reorder_idx`, `inv_reorder_idx`, and `prev_neighbors` must be "
            "provided at the same time."
        )

    reorder_idx = np.asarray(reorder_idx)
    inv_reorder_idx = np.asarray(inv_reorder_idx)
    prev_neighbors = np.asarray(prev_neighbors)

    if reorder_idx.ndim != 1:
        raise ValueError("`reorder_idx` must be 1D.")
    if inv_reorder_idx.ndim != 1:
        raise ValueError("`inv_reorder_idx` must be 1D.")
    if prev_neighbors.ndim != 2:
        raise ValueError("`prev_neighbors` must be 2D.")

    V = reorder_idx.size
    if inv_reorder_idx.size != V:
        raise ValueError(
            "`reorder_idx` and `inv_reorder_idx` must have the same length."
        )
    if prev_neighbors.shape[0] != V:
        raise ValueError(
            "`reorder_idx` and `prev_neighbors` must have the same length."
        )

    idx = reorder_idx[inv_reorder_idx]
    if not np.array_equal(idx, np.arange(V)):
        raise ValueError("`inv_reorder_idx` is not the inverse of `reorder_idx`.")

    for i in range(V):
        n = prev_neighbors[i]
        for j in n:
            if j < -1 or j >= V:
                raise ValueError(f"Invaild neighbor {j} of site {i}")

        n = [j for j in n if j >= 0]
        if len(set(n)) != len(n):
            raise ValueError(f"Duplicate neighbor {n} of site {i}")

        for j in n:
            if reorder_idx[j] >= reorder_idx[i]:
                raise ValueError(f"Site {j} is not a previous neighbor of site {i}")


class RNNLayer(nn.Module):
    """Base class for recurrent neural network layers."""

    features: int
    """output feature density, should be the last dimension."""
    exclusive: bool
    """True if an output element does not depend on the input element at the same index."""
    reorder_idx: Optional[HashableArray] = None
    """indices to transform the inputs from unordered to ordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    inv_reorder_idx: Optional[HashableArray] = None
    """indices to transform the inputs from ordered to unordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    prev_neighbors: Optional[HashableArray] = None
    """previous neighbors of each site."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    def __post_init__(self):
        super().__post_init__()
        _check_reorder_idx(self.reorder_idx, self.inv_reorder_idx, self.prev_neighbors)

    def _dense_params(self, name: Optional[str], in_features: int, out_features: int):
        kernel = self.param(
            name + "_kernel" if name else "kernel",
            self.kernel_init,
            (in_features, out_features),
            self.param_dtype,
        )
        bias = self.param(
            name + "_bias" if name else "bias",
            self.bias_init,
            (out_features,),
            self.param_dtype,
        )
        return kernel, bias

    # We need to define the parameters outside lax.scan,
    # and use the returned function inside lax.scan
    def _get_recur_func(
        self, inputs: Array, hid_features: int
    ) -> Callable[[Array, Array, Array], tuple[Array, Array]]:
        raise NotImplementedError

    @nn.compact
    def __call__(self, inputs):
        """
        Applies the RNN cell to a batch of input sequences.

        Args:
          inputs: input data with dimensions (batch, length, features).

        Returns:
          The output sequences.
        """
        batch_size, V, _ = inputs.shape

        if self.reorder_idx is None:
            max_prev_neighbors = 1
        else:
            reorder_idx = jnp.asarray(self.reorder_idx)
            prev_neighbors = jnp.asarray(self.prev_neighbors)
            max_prev_neighbors = prev_neighbors.shape[1]

        recur_func = self._get_recur_func(inputs, max_prev_neighbors * self.features)
        inputs = promote_dtype(inputs, dtype=self.param_dtype)[0]

        def scan_func(carry, k):
            cell, outputs = carry
            if self.reorder_idx is None:
                index = k
            else:
                index = reorder_idx[k]

            if self.exclusive:
                # Get the inputs at the previous site in the autoregressive order,
                # or zeros for the first site
                if self.reorder_idx is None:
                    inputs_i = inputs[:, k - 1, :]
                else:
                    inputs_i = inputs[:, reorder_idx[k - 1], :]
                inputs_i = jnp.where(k == 0, 0, inputs_i)
            else:
                # Get the inputs at the current site
                inputs_i = inputs[:, index, :]

            if self.reorder_idx is None:
                # Get the hidden memory at the previous site,
                # or zeros for the first site
                hidden = outputs[:, k - 1, :]
                hidden = jnp.expand_dims(hidden, axis=-1)
            else:
                # Get the hidden memories at the previous neighbors,
                # or zeros for boundaries
                n = prev_neighbors[index]
                hidden = outputs[:, n, :]
                hidden = jnp.where(n[None, :, None] == -1, 0, hidden)

            cell, hidden = recur_func(inputs_i, cell, hidden)

            outputs = outputs.at[:, index, :].set(hidden)
            return (cell, outputs), outputs

        cell = jnp.zeros((batch_size, self.features), dtype=inputs.dtype)
        outputs = jnp.zeros((batch_size, V, self.features), dtype=inputs.dtype)
        (_, outputs), _ = lax.scan(scan_func, (cell, outputs), jnp.arange(V))
        return outputs


class LSTMLayer(RNNLayer):
    """Long short-term memory layer."""

    def _get_recur_func(self, _inputs, hid_features):
        batch_size = _inputs.shape[0]
        in_features = _inputs.shape[-1]
        kernel, bias = self._dense_params(
            None, in_features + hid_features, self.features * 4
        )
        _, kernel, bias = promote_dtype(_inputs, kernel, bias, dtype=None)

        def recur_func(inputs, cell, hidden):
            hidden = hidden.reshape((batch_size, -1))
            in_cat = jnp.concatenate([inputs, hidden], axis=-1)
            ifgo = nn.sigmoid(in_cat @ kernel + bias)
            i, f, g, o = jnp.split(ifgo, 4, axis=-1)

            # sigmoid -> tanh
            g = g * 2 - 1

            cell = f * cell + i * g
            outputs = o * nn.tanh(cell)
            return cell, outputs

        return recur_func


class GRULayer1D(RNNLayer):
    """Gated recurrent unit layer. Only supports one previous neighbor at each site."""

    def _get_recur_func(self, _inputs, hid_features):
        batch_size = _inputs.shape[0]
        in_features = _inputs.shape[-1]
        rz_kernel, rz_bias = self._dense_params(
            "rz", in_features + hid_features, self.features * 2
        )
        n_kernel, n_bias = self._dense_params(
            "n", in_features + hid_features, self.features
        )
        _, rz_kernel, rz_bias, n_kernel, n_bias = promote_dtype(
            _inputs, rz_kernel, rz_bias, n_kernel, n_bias, dtype=None
        )

        def recur_func(inputs, cell, hidden):
            hidden = hidden.reshape((batch_size, -1))
            in_cat = jnp.concatenate([inputs, hidden], axis=-1)
            rz = nn.sigmoid(in_cat @ rz_kernel + rz_bias)
            r, z = jnp.split(rz, 2, axis=-1)

            in_cat = jnp.concatenate([inputs, r * hidden], axis=-1)
            n = nn.tanh(in_cat @ n_kernel + n_bias)

            outputs = (1 - z) * n + z * hidden
            return cell, outputs

        return recur_func
