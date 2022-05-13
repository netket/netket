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

import netket as nk
import jax
import jax.numpy as jnp


def test_deepsets():
    sdim = 2
    x = jnp.hstack([jnp.ones(4), -jnp.ones(4)]).reshape(1, -1)
    xp = jnp.roll(x, sdim)
    ds = nk.models.DeepSet(
        L=1.0,
        sdim=sdim,
        layers_phi=2,
        layers_rho=2,
        features_phi=(10, 10),
        features_rho=(10, 1),
    )
    p = ds.init(jax.random.PRNGKey(42), x)

    assert jnp.allclose(ds.apply(p, x), ds.apply(p, xp))
