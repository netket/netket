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

import jax
import netket as nk
import numpy as np
import optax
from jax import numpy as jnp

from .. import common

pytestmark = common.skipif_mpi


def test_adam():
    def f(x):
        return (x.conj() * x).real.sum()

    optimizer = nk.optimizer.Adam(learning_rate=0.1)

    @jax.jit
    def update(params, opt_state):
        grads = jax.grad(f)(params)
        # Complex gradients need to be conjugated before being added to parameters
        grads = jax.tree_map(lambda x: x.conj(), grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    x = jnp.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    opt_state = optimizer.init(x)
    for _ in range(1000):
        x, opt_state = update(x, opt_state)

    np.testing.assert_allclose(x, 0, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(f(x), 0, rtol=1e-7, atol=1e-7)


def test_split_complex():
    def loss_fun_complex_to_real(z):
        return (z.conj() * z).real.sum()

    def loss_fun_real_to_real(params):
        x, y = params
        return loss_fun_complex_to_real(x + y * 1j)

    def update(loss_fun, optimizer, params, opt_state):
        loss, grads = jax.value_and_grad(loss_fun)(params)
        # Complex gradients need to be conjugated before being added to parameters
        grads = jax.tree_map(lambda x: x.conj(), grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, grads, params, opt_state

    # Random initial parameters
    x = jnp.asarray(np.random.randn(3))
    y = jnp.asarray(np.random.randn(3))
    z = x + y * 1j

    optimizer = nk.optimizer.Adam()
    optimizer_complex = nk.optimizer.split_complex(optimizer)
    opt_state = optimizer.init((x, y))
    opt_state_complex = optimizer_complex.init(z)

    # Check that the loss, the gradients, and the parameters are the same for
    # real-to-real and complex-to-real loss functions in each step
    for _ in range(3):
        loss, (gx, gy), (x, y), opt_state = update(
            loss_fun_real_to_real, optimizer, (x, y), opt_state
        )
        loss_complex, gz, z, opt_state_complex = update(
            loss_fun_complex_to_real, optimizer_complex, z, opt_state_complex
        )
        np.testing.assert_allclose(loss, loss_complex)
        np.testing.assert_allclose(gx + gy * 1j, gz)
        np.testing.assert_allclose(x + y * 1j, z)
