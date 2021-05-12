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


def test_ARNNDense():
    """Test if the model is autoregressive."""

    batch = 3
    size = 4

    model = nk.models.ARNNDense(layers=3, features=5)
    in_shape = (batch, size)
    key_spins, key_model = jax.random.split(jax.random.PRNGKey(0))
    spins = jax.random.bernoulli(key_spins, shape=in_shape).astype(
        model.dtype) * 2 - 1
    (p, _), params = model.init_with_output(key_model,
                                            spins,
                                            None,
                                            method=model.conditionals)

    for i in range(batch):
        for j in range(size):
            # Change one input element at a time
            spins_new = spins.at[i, j].set(-spins[i, j])
            p_new, _ = model.apply(params,
                                   spins_new,
                                   None,
                                   method=model.conditionals)
            p_diff = p_new - p

            # The former output elements should not change
            p_diff = p_diff.at[i, j + 1:].set(0)

            np.testing.assert_allclose(p_diff, 0, err_msg=f'{i=} {j=}')
