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

import jax
import flax


from .. import common

pytestmark = common.skipif_distributed


def test_is_holomorphic():
    from netket.utils import is_probably_holomorphic

    hi = nk.hilbert.Spin(0.5, 5)
    s = hi.all_states()
    rng = jax.random.PRNGKey(1)

    # real pars -> non holo
    ma = nk.models.RBM(param_dtype=float)
    state, pars = flax.core.pop(ma.init(rng, s), "params")
    assert not is_probably_holomorphic(ma.apply, pars, s)

    # complex pars -> holo
    ma = nk.models.RBM(param_dtype=complex)
    state, pars = flax.core.pop(ma.init(rng, s), "params")
    assert is_probably_holomorphic(ma.apply, pars, s)

    # complex pars, non holo fun -> holo
    ma = nk.models.RBM(activation=nk.nn.activation.reim_selu, param_dtype=complex)
    state, pars = flax.core.pop(ma.init(rng, s), "params")
    assert not is_probably_holomorphic(ma.apply, pars, s)

    ma = nk.models.ARNNDense(
        hi, 2, 2, param_dtype=complex, activation=nk.nn.activation.log_cosh
    )
    state, pars = flax.core.pop(ma.init(rng, s), "params")
    assert not is_probably_holomorphic(ma.apply, pars, s)
