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
import pytest
from jax import numpy as jnp


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
@pytest.mark.parametrize(
    "hilbert",
    [
        pytest.param(
            nk.hilbert.Spin(s=1 / 2, N=4),
            id="spin_1/2",
        ),
        pytest.param(
            nk.hilbert.Spin(s=1, N=4),
            id="spin_1",
        ),
        pytest.param(
            nk.hilbert.Fock(n_max=3, N=4),
            id="fock",
        ),
    ],
)
@pytest.mark.parametrize(
    "partial_model",
    [
        pytest.param(
            lambda hilbert, dtype: nk.models.ARNNDense(
                hilbert=hilbert,
                layers=3,
                features=5,
                dtype=dtype,
            ),
            id="dense",
        ),
        pytest.param(
            lambda hilbert, dtype: nk.models.ARNNConv1D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=2,
                dtype=dtype,
            ),
            id="conv1d",
        ),
        pytest.param(
            lambda hilbert, dtype: nk.models.ARNNConv1D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=2,
                kernel_dilation=2,
                dtype=dtype,
            ),
            id="conv1d_dilation",
        ),
    ],
)
def test_ARNN(partial_model, hilbert, dtype):
    batch_size = 3

    model = partial_model(hilbert, dtype)

    key_spins, key_model = jax.random.split(jax.random.PRNGKey(0))
    spins = hilbert.random_state(key_spins, size=batch_size)
    (p, _), params = model.init_with_output(
        key_model, spins, None, method=model.conditionals
    )

    # Test if the model is normalized
    # The result may not be very accurate, because it is in exp space
    psi = nk.nn.to_array(hilbert, model.apply, params, normalize=False)
    assert psi.conj() @ psi == pytest.approx(1, rel=1e-5, abs=1e-5)

    # Test if the model is autoregressive
    for i in range(batch_size):
        for j in range(hilbert.size):
            # Change one input element at a time
            spins_new = spins.at[i, j].set(-spins[i, j])
            p_new, _ = model.apply(params, spins_new, None, method=model.conditionals)
            p_diff = p_new - p

            # The former output elements should not change
            p_diff = p_diff.at[i, j + 1 :].set(0)

            np.testing.assert_allclose(p_diff, 0, err_msg=f"i={i} j={j}")


def test_throwing():
    def build_model(hilbert):
        nk.models.ARNNConv1D(hilbert=hilbert, layers=3, features=5, kernel_size=2)

    # Only homogeneous Hilbert spaces are supported
    with pytest.raises(ValueError):
        hilbert = nk.hilbert.Spin(s=1 / 2, N=4)
        hilbert = nk.hilbert.DoubledHilbert(hilbert)
        build_model(None)

    # Only unconstrained Hilbert spaces are supported
    with pytest.raises(ValueError):
        hilbert = nk.hilbert.Fock(n_max=3, N=4, n_particles=3)
        build_model(hilbert)
