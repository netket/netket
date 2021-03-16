# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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
from jax import random

from .abstract_density_matrix import AbstractDensityMatrix
from ..flax import Flax as FlaxPure
from functools import partial


class Flax(FlaxPure, AbstractDensityMatrix):
    def __init__(self, hilbert, module, dtype=complex):
        """
        Wraps a stax network (which is a tuple of `init_fn` and `predict_fn`)
        so that it can be used as a NetKet density matrix.

        Args:
            hilbert: Hilbert space on which the state is defined. Should be a
                subclass of `netket.hilbert.Hilbert`.
            module: A pair `(init_fn, predict_fn)`. See the documentation of
                `jax.experimental.stax` for more info.
            dtype: either complex or float, is the type used for the weights.
                In both cases the module must have a single output.
        """
        AbstractDensityMatrix.__init__(self, hilbert, dtype)
        FlaxPure.__init__(self, hilbert, module)

        assert self.input_size == self.hilbert.size * 2

    @staticmethod
    @jax.jit
    def _dminput(xr, xc):
        if xc is None:
            x = xr
        else:
            x = jnp.hstack((xr, xc))
        return x

    def log_val(self, xr, xc=None, out=None):
        x = self._dminput(xr, xc)

        return FlaxPure.log_val(self, x, out=out)

    def der_log(self, xr, xc=None, out=None):
        x = self._dminput(xr, xc)

        return FlaxPure.der_log(self, x, out=out)

    def diagonal(self):
        from .diagonal import Diagonal

        diag = Diagonal(self)

        def diag_jax_forward(params, x):
            return self.jax_forward(params, self._dminput(x, x))

        diag.jax_forward = diag_jax_forward

        return diag
