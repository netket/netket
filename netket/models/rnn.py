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

from math import sqrt
from typing import Iterable, Optional, Union

import numpy as np
from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.models.autoreg import AbstractARNN, _call, _conditionals, _get_feature_list
from netket.nn.rnn import GRULayer1D, LSTMLayer1D, default_kernel_init
from netket.nn.rnn_2d import LSTMLayer2D
from netket.utils import deprecate_dtype
from netket.utils.types import Array, DType, NNInitFunc


class RNN(AbstractARNN):
    """Base class for recurrent neural networks."""

    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    reorder_idx: Optional[jnp.ndarray] = None
    """see :class:`netket.models.AbstractARNN`."""
    inv_reorder_idx: Optional[jnp.ndarray] = None
    """see :class:`netket.models.AbstractARNN`."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def setup(self):
        raise NotImplementedError

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)

    def reorder(self, inputs: Array) -> Array:
        if self.reorder_idx is None:
            return inputs
        else:
            if inputs.ndim == 1:
                return inputs[self.reorder_idx]
            else:
                return inputs[:, self.reorder_idx]

    def inverse_reorder(self, inputs: Array) -> Array:
        if self.inv_reorder_idx is None:
            return inputs
        else:
            if inputs.ndim == 1:
                return inputs[self.inv_reorder_idx]
            else:
                return inputs[:, self.inv_reorder_idx]


@deprecate_dtype
class LSTMNet1D(RNN):
    """1D long short-term memory network."""

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            LSTMLayer1D(
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


@deprecate_dtype
class GRUNet1D(RNN):
    """1D gated recurrent unit network."""

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            GRULayer1D(
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


def _get_snake_ordering(V):
    L = int(sqrt(V))
    if L**2 != V:
        raise ValueError(f"Number of sites {V} is not a square number")

    a = np.arange(V, dtype=np.intp).reshape((L, L))
    a[1::2, :] = a[1::2, ::-1]
    a = a.flatten()
    a = jnp.asarray(a)

    # Snake ordering has reorder_idx == inv_reorder_idx,
    # but for a general ordering it's not always true
    return a, a


@deprecate_dtype
class LSTMNet2D(RNN):
    """2D long short-term memory network with snake ordering for square lattice."""

    def setup(self):
        if self.reorder_idx is not None or self.inv_reorder_idx is not None:
            raise ValueError(
                "`LSTMNet2D` only supports snake ordering for square lattice"
            )
        self._reorder_idx, self._inv_reorder_idx = _get_snake_ordering(
            self.hilbert.size
        )

        features = _get_feature_list(self)
        self._layers = [
            LSTMLayer2D(
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self._reorder_idx,
                inv_reorder_idx=self._inv_reorder_idx,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    @property
    def reorder_idx(self) -> Optional[Array]:
        return self._reorder_idx

    @reorder_idx.setter
    def reorder_idx(self, value: Optional[Array]):
        self._reorder_idx = value

    @property
    def inv_reorder_idx(self) -> Array:
        return self._inv_reorder_idx

    @inv_reorder_idx.setter
    def inv_reorder_idx(self, value: Optional[Array]):
        self._inv_reorder_idx = value
