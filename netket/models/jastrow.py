# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import normal

from flax.linen.dtypes import promote_dtype

from netket.utils import deprecate_dtype
from netket.utils.types import DType, Array, NNInitFunc


@deprecate_dtype
class Jastrow(nn.Module):
    r"""
    Jastrow wave function :math:`\Psi(s) = \exp(\sum_{ij} s_i W_{ij} s_j)`.

    The W matrix is stored as a non-symmetric matrix, and symmetrized
    during computation by doing :code:`W = W + W.T` in the computation.
    """

    param_dtype: DType = jnp.complex128
    """The dtype of the weights."""
    kernel_init: NNInitFunc = normal()
    """Initializer for the weights."""

    @nn.compact
    def __call__(self, x_in: Array):
        nv = x_in.shape[-1]

        kernel = self.param("kernel", self.kernel_init, (nv, nv), self.param_dtype)
        kernel = kernel + kernel.T

        kernel, x_in = promote_dtype(kernel, x_in, dtype=None)
        y = jnp.einsum("...i,ij,...j", x_in, kernel, x_in)

        return y
