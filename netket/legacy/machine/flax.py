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
from jax import numpy as jnp
import flax
from flax import linen as nn
from functools import partial

from .jax import Jax, add_package_wrapper

from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from ._jax_utils import forward_apply, grad, vjp, tree_leaf_iscomplex


def flax_init(module, rng, shape):
    shape = (1,) + shape[1:]
    dummy_input = jax.numpy.zeros(shape, dtype=jax.numpy.float64)
    params = module.init(rng, dummy_input)
    return (-1, 1), params


# Declare Flax as a jax wrapper package.
add_package_wrapper(
    lambda m: isinstance(m, flax.linen.Module),
    lambda m: (partial(flax_init, m), m.apply),
)


class Flax(Jax):
    def __init__(self, hilbert, module):
        """
        Wraps a stax network (which is a tuple of `init_fn` and `predict_fn`)
        so that it can be used as a NetKet machine.

        Args:
            hilbert: Hilbert space on which the state is defined. Should be a
                subclass of `netket.hilbert.Hilbert`.
            module: A pair `(init_fn, predict_fn)`. See the documentation of
                jax.experimental.stax` for more info.
            dtype: either complex or float, is the type used for the weights.
                In both cases the network must have a single output.
        """
        super().__init__(hilbert=hilbert, module=module)
        self._dtype = complex if tree_leaf_iscomplex(self._params) else float

    def init_random_parameters(self, seed=None, sigma=None):
        self.jax_init_parameters(seed)

    def _cast(self, p):
        return p
