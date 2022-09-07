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

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform

from .test_nn import _setup_symm

import pytest


def test_logstatevec():
    hi = nk.hilbert.Fock(3, 4)
    x = hi.random_state(nk.jax.PRNGKey(3), (3, 4))

    ma = nk.models.LogStateVector(hi, param_dtype=complex)
    pars = ma.init(nk.jax.PRNGKey(), hi.random_state(nk.jax.PRNGKey(), 1))

    s1 = jax.jit(ma.apply)(pars, x)
    s2 = jax.jit(jax.vmap(ma.apply, in_axes=(None, 0)))(pars, x)

    np.testing.assert_allclose(s1, s2)
    np.testing.assert_allclose(pars["params"]["logstate"][hi.states_to_numbers(x)], s1)
