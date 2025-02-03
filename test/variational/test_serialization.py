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
import numpy as np
from flax import serialization

import netket as nk


def test_init_with_variables_deserialization():
    # This test used to fail when sharding was disabled
    hi = nk.hilbert.Spin(0.5, 2)
    sa = nk.sampler.MetropolisLocal(hi)
    ma = nk.models.RBM()
    variables = ma.init(jax.random.key(1), hi.all_states())

    variables_np = jax.tree.map(np.asarray, variables)
    vs = nk.vqs.MCState(sa, ma, n_samples=64, variables=variables_np)

    state_bytes = serialization.to_bytes(vs)

    _ = serialization.from_bytes(vs, state_bytes)

    # We simply check that this does not fail..
