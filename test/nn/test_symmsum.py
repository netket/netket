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
from netket.utils import HashableArray

bare_modules = {}
bare_modules["RBM(real)"] = nk.models.RBM(alpha=1, param_dtype=float)
bare_modules["RBM(complex)"] = nk.models.RBM(alpha=1, param_dtype=complex)

ones = HashableArray(np.ones(9))


@pytest.mark.parametrize(
    "bare_module", [pytest.param(v, id=k) for k, v in bare_modules.items()]
)
@pytest.mark.parametrize(
    "character_id,characters,trivial",
    [
        pytest.param(None, None, True, id="Char=None"),
        pytest.param(0, None, True, id="Id=0"),
        pytest.param(1, None, False, id="Id=1"),
        pytest.param(None, ones, True, id="Char=Triv"),
        pytest.param(
            0, ones, True, marks=pytest.mark.xfail(raises=AttributeError), id="Invalid"
        ),
    ],
)
def test_symmexpsum(bare_module, character_id, characters, trivial):
    graph = nk.graph.Square(3)
    g = graph.translation_group()

    ma = nknn.blocks.SymmExpSum(
        bare_module, symm_group=g, character_id=character_id, characters=characters
    )

    hi = nk.hilbert.Spin(0.5, graph.n_nodes)
    vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), ma)

    log_psi = vs.log_value(hi.all_states())
    assert np.all(np.isfinite(log_psi))

    if trivial:
        # check that it gives the same output for all symmetry-related inputs
        for Tg in g:
            np.testing.assert_allclose(log_psi, vs.log_value(Tg @ hi.all_states()))

        # check that for symmetric inputs it gives same output as the original
        s0 = jnp.full((hi.size,), 1.3)
        out_sym = ma.apply(vs.variables, s0)
        out_bare = bare_module.apply({"params": vs.parameters["module"]}, s0)
        np.testing.assert_allclose(out_sym, out_bare)

    # check that it works with different shapes
    s0 = hi.random_state(jax.random.PRNGKey(0))
    out0, pars = ma.init_with_output(jax.random.PRNGKey(0), s0)
    assert out0.shape == ()

    out1 = ma.apply(pars, s0.reshape((1, -1)))
    assert out1.shape == (1,)

    np.testing.assert_allclose(out0, out1.reshape(()))

    # 2D and 3D
    s1 = hi.random_state(jax.random.PRNGKey(0), (100,))
    out1 = ma.apply(pars, s1)
    assert out1.shape == (100,)

    s2 = s1.reshape((10, 10, hi.size))
    out2 = ma.apply(pars, s2)
    assert out2.shape == (10, 10)

    def _logspace_allclose(x, y, logrcond=-10):
        # find elements where the amplitude is not essentially zero
        mask = x.real - x.real.max() > logrcond
        # check amplitude and phase of nonzeros
        np.testing.assert_allclose(x[mask], y[mask])
        # check amplitude of zeros with looser bound
        np.testing.assert_allclose(x[~mask].real, y[~mask].real, atol=0.5)

    _logspace_allclose(out1, out2.reshape((-1,)))

    s3 = s1.reshape((2, 5, 10, hi.size))
    out3 = ma.apply(pars, s3)
    assert out3.shape == (2, 5, 10)
    _logspace_allclose(out1, out3.reshape((-1,)))
