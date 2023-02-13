# Copyright 2022 The NetKet Authors - All rights reserved.
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

import pytest
import numpy as np
import jax
import jax.numpy as jnp

import netket as nk
import netket.nn as nknn

bare_modules = {}
bare_modules["RBM(real)"] = nk.models.RBM(alpha=1, param_dtype=float)
bare_modules["RBM(complex)"] = nk.models.RBM(alpha=1, param_dtype=complex)

character_ids = {}
character_ids["Char=None"] = None
character_ids["Char=0"] = 0
character_ids["Char=1"] = 1


@pytest.mark.parametrize(
    "bare_module", [pytest.param(v, id=k) for k, v in bare_modules.items()]
)
@pytest.mark.parametrize(
    "character_id", [pytest.param(v, id=k) for k, v in character_ids.items()]
)
def test_symmexpsum(bare_module, character_id):
    graph = nk.graph.Square(3)
    g = graph.translation_group()

    ma = nknn.blocks.SymmExpSum(bare_module, symm_group=g, character_id=character_id)

    hi = nk.hilbert.Spin(0.5, graph.n_nodes)
    vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), ma)

    log_psi = vs.log_value(hi.all_states())
    assert np.all(np.isfinite(log_psi))

    if character_id != 1:
        for Tg in g:
            np.testing.assert_allclose(log_psi, vs.log_value(Tg @ hi.all_states()))
