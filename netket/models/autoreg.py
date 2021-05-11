# Copyright 2021 The NetKet Authors - All rights reserved.
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

import abc
from typing import Iterable, Tuple

from flax import linen as nn
from jax import numpy as jnp
from netket.utils.types import Array, DType, PyTree


class ARNN(nn.Module):
    """Base class for autoregressive neural networks."""
    def init_sample(self, size: Iterable[int],
                    dtype: DType) -> Tuple[Array, PyTree]:
        """
        Initializes the model for sampling.

        Args:
          size: (batch, Hilbert.size).
          dtype: dtype of the spins.

        Returns:
          spins: the initial state that all spins are not sampled yet, by default an array of zeros.
          state: some auxiliary states, e.g., used to implement fast autoregressive sampling.
        """
        spins = jnp.zeros(size, dtype=dtype)
        state = None
        return spins, state

    @abc.abstractmethod
    def conditional(self, inputs: Array, index: int,
                    state: PyTree) -> Tuple[Array, PyTree]:
        """
        Computes the probabilities for a spin to take each value.

        Args:
          inputs: input data with dimensions (batch, Hilbert.size).
          index: index of the spin to sample.
          state: some auxiliary states, e.g., used to implement fast autoregressive sampling.

        Returns:
          p: the probabilities with dimensions (batch, Hilbert.local_size).
          state: the updated model state.
        """
        raise NotImplementedError
