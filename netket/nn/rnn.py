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

from typing import Callable, Optional, Tuple

import numpy as np
from flax import linen as nn
from flax.linen.dtypes import promote_dtype

from jax import lax
from jax import numpy as jnp
from jax.nn.initializers import orthogonal, zeros

from netket.utils import deprecate_dtype
from netket.utils.types import Array, DType, NNInitFunc

default_kernel_init = orthogonal()


def check_reorder_idx(reorder_idx: Array, inv_reorder_idx: Array):
    if reorder_idx is None and inv_reorder_idx is None:
        return

    if reorder_idx is None or inv_reorder_idx is None:
        raise ValueError(
            "`reorder_idx` and `inv_reorder_idx` must be set at the same time."
        )

    if reorder_idx.ndim != 1:
        raise ValueError("`reorder_idx` must be 1D.")
    if inv_reorder_idx.ndim != 1:
        raise ValueError("`inv_reorder_idx` must be 1D.")

    if reorder_idx.size != inv_reorder_idx.size:
        raise ValueError(
            "`reorder_idx` and `inv_reorder_idx` must have the same length."
        )

    idx = reorder_idx[inv_reorder_idx]
    if not np.array_equal(idx, np.arange(reorder_idx.size)):
        raise ValueError("`inv_reorder_idx` is not the inverse of `reorder_idx`.")


class RNNLayer(nn.Module):
    """Base class for recurrent neural network layers."""

    features: int
    """output feature density, should be the last dimension."""
    exclusive: bool
    """True if an output element does not depend on the input element at the same index."""
    reorder_idx: Optional[jnp.ndarray] = None
    """see :class:`netket.models.AbstractARNN`."""
    inv_reorder_idx: Optional[jnp.ndarray] = None
    """see :class:`netket.models.AbstractARNN`."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    def __post_init__(self):
        super().__post_init__()
        check_reorder_idx(self.reorder_idx, self.inv_reorder_idx)

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


class RNNLayer1D(RNNLayer):
    """Base class for 1D recurrent neural network layers."""

    def _get_recur_func(
        self, inputs: Array
    ) -> Callable[[Array, Array, Array], Tuple[Array, Array]]:
        raise NotImplementedError

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Applies the RNN cell to a batch of input sequences.

        Args:
          inputs: input data with dimensions (batch, length, features).

        Returns:
          The output sequences.
        """
        batch_size = inputs.shape[0]
        recur_func = self._get_recur_func(inputs)

        # (batch, length, features) -> (length, batch, features)
        inputs = inputs.transpose((1, 0, 2))
        if self.reorder_idx is not None:
            inputs = inputs[self.reorder_idx]

        if self.exclusive:
            inputs = inputs[:-1]
            inputs = jnp.pad(inputs, ((1, 0), (0, 0), (0, 0)))

        def scan_func(carry, inputs):
            cell, hidden = carry
            cell, hidden = recur_func(inputs, cell, hidden)
            return (cell, hidden), hidden

        inputs = promote_dtype(inputs, dtype=self.param_dtype)[0]
        zeros = jnp.zeros((batch_size, self.features), dtype=inputs.dtype)
        _, outputs = lax.scan(scan_func, (zeros, zeros), inputs)

        if self.inv_reorder_idx is not None:
            outputs = outputs[self.inv_reorder_idx]
        # (length, batch, features) -> (batch, length, features)
        outputs = outputs.transpose((1, 0, 2))
        return outputs


@deprecate_dtype
class LSTMLayer1D(RNNLayer1D):
    """1D long short-term memory layer."""

    def _get_recur_func(self, inputs):
        in_features = inputs.shape[-1]
        kernel, bias = self._dense_params(
            None, in_features + self.features, self.features * 4
        )
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=None)

        def recur_func(inputs, cell, hidden):
            in_cat = jnp.concatenate([inputs, hidden], axis=-1)
            ifgo = nn.sigmoid(in_cat @ kernel + bias)
            i, f, g, o = ifgo.split(4, axis=-1)

            # sigmoid -> tanh
            g = g * 2 - 1

            cell = f * cell + i * g
            outputs = o * nn.tanh(cell)
            return cell, outputs

        return recur_func


@deprecate_dtype
class GRULayer1D(RNNLayer1D):
    """1D gated recurrent unit layer."""

    def _get_recur_func(self, inputs):
        in_features = inputs.shape[-1]
        rz_kernel, rz_bias = self._dense_params(
            "rz", in_features + self.features, self.features * 2
        )
        n_kernel, n_bias = self._dense_params(
            "n", in_features + self.features, self.features
        )
        inputs, rz_kernel, rz_bias, n_kernel, n_bias = promote_dtype(
            inputs, rz_kernel, rz_bias, n_kernel, n_bias, dtype=None
        )

        def recur_func(inputs, cell, hidden):
            in_cat = jnp.concatenate([inputs, hidden], axis=-1)
            rz = nn.sigmoid(in_cat @ rz_kernel + rz_bias)
            r, z = rz.split(2, axis=-1)

            in_cat = jnp.concatenate([inputs, r * hidden], axis=-1)
            n = nn.tanh(in_cat @ n_kernel + n_bias)

            outputs = (1 - z) * n + z * hidden
            return cell, outputs

        return recur_func
