# Copyright 2023 The NetKet Authors - All rights reserved.
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
import netket.experimental as nkx
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

import pytest


def test_Slater2nd():
    k = jax.random.PRNGKey(1)

    n_dets = 4
    for slater_class in [
        nkx.models.Slater2nd,
        partial(nkx.models.MultiSlater2nd, n_determinants=n_dets),
    ]:
        hi = nk.hilbert.SpinOrbitalFermions(3, n_fermions=2)
        ma = slater_class(hi, restricted=True, param_dtype=jnp.float32)

        pars = ma.init(k, hi.all_states())
        out0 = ma.apply(pars, hi.numbers_to_states(0))
        assert out0.shape == ()

        out1 = ma.apply(pars, hi.numbers_to_states([0, 1, 2]))
        assert out1.shape == (3,)

        out1 = ma.apply(pars, hi.random_state(k, (2, 3), dtype=jnp.float32))
        assert out1.shape == (2, 3)
        assert out1.dtype == jnp.complex64

        hi = nk.hilbert.SpinOrbitalFermions(3, s=0.5, n_fermions_per_spin=(2, 2))
        ma = slater_class(hi, restricted=True, param_dtype=jnp.float32)

        pars = ma.init(k, hi.all_states())
        out0 = ma.apply(pars, hi.numbers_to_states(0))
        assert out0.shape == ()

        out1 = ma.apply(pars, hi.numbers_to_states([0, 1, 2]))
        assert out1.shape == (3,)

        out1 = ma.apply(pars, hi.random_state(k, (2, 3), dtype=jnp.float32))
        assert out1.shape == (2, 3)
        assert out1.dtype == jnp.complex64

        # check that restricted gives same outputs for both orbitals
        x1 = jnp.array([0, 1, 1, 1, 1, 0])  # (0,1,1) (1,1,0)
        x2 = jnp.array([1, 1, 0, 0, 1, 1])  # (0,1,1) (1,1,0)
        np.testing.assert_allclose(ma.apply(pars, x1), ma.apply(pars, x2))

        # higher spin check
        hi = nk.hilbert.SpinOrbitalFermions(2, s=1.5, n_fermions_per_spin=(1, 1, 1, 1))
        ma = slater_class(hi, restricted=True, param_dtype=jnp.float32)

        pars = ma.init(k, hi.all_states())
        out0 = ma.apply(pars, hi.numbers_to_states(0))
        assert out0.shape == ()

        out1 = ma.apply(pars, hi.numbers_to_states([0, 1, 2]))
        assert out1.shape == (3,)

        out1 = ma.apply(pars, hi.random_state(k, (2, 3), dtype=jnp.float32))
        assert out1.shape == (2, 3)
        assert out1.dtype == jnp.complex64

        hi = nk.hilbert.SpinOrbitalFermions(3, s=0.5, n_fermions_per_spin=(2, 2))
        ma = slater_class(hi, restricted=False, param_dtype=jnp.float32)

        pars = ma.init(k, hi.all_states())
        out0 = ma.apply(pars, hi.numbers_to_states(0))
        assert out0.shape == ()

        out1 = ma.apply(pars, hi.numbers_to_states([0, 1, 2]))
        assert out1.shape == (3,)

        out1 = ma.apply(pars, hi.random_state(k, (2, 3), dtype=jnp.float32))
        assert out1.shape == (2, 3)
        assert out1.dtype == jnp.complex64

        # check that restricted gives same outputs for both orbitals
        x1 = jnp.array([0, 1, 1, 1, 1, 0])  # (0,1,1) (1,1,0)
        x2 = jnp.array([1, 1, 0, 0, 1, 1])  # (0,1,1) (1,1,0)
        assert not np.allclose(ma.apply(pars, x1), ma.apply(pars, x2))

        hi = nk.hilbert.SpinOrbitalFermions(3, n_fermions=2)
        ma = slater_class(hi, restricted=False, param_dtype=jnp.float32)

        # check generalized
        hi = nk.hilbert.SpinOrbitalFermions(3, s=0.5, n_fermions_per_spin=(2, 2))
        ma = slater_class(hi, generalized=True, param_dtype=jnp.float32)

        pars = ma.init(k, hi.all_states())
        out0 = ma.apply(pars, hi.numbers_to_states(0))
        assert out0.shape == ()

        out1 = ma.apply(pars, hi.numbers_to_states([0, 1, 2]))
        assert out1.shape == (3,)

        out1 = ma.apply(pars, hi.random_state(k, (2, 3), dtype=jnp.float32))
        assert out1.shape == (2, 3)
        assert out1.dtype == jnp.complex64

        if isinstance(ma, nkx.models.MultiSlater2nd):
            M = pars["params"]["VmapSlater2nd_0"]["M"]
            assert M.shape == (n_dets, hi.size, hi.n_fermions)
        else:
            M = pars["params"]["M"]
            assert M.shape == (hi.size, hi.n_fermions)


def test_Slater2nd_error():
    # Requires number of fermions
    with pytest.raises(TypeError):
        hi = nk.hilbert.SpinOrbitalFermions(3)
        ma = nkx.models.Slater2nd(hi, restricted=True)

    # Requires equal number of fermions
    with pytest.raises(ValueError):
        hi = nk.hilbert.SpinOrbitalFermions(3, s=0.5, n_fermions_per_spin=(2, 3))
        ma = nkx.models.Slater2nd(hi, restricted=True)

    # Wrong sample shape
    with pytest.raises(ValueError):
        hi = nk.hilbert.SpinOrbitalFermions(3, s=0.5, n_fermions_per_spin=(2, 2))
        ma = nkx.models.Slater2nd(hi, restricted=True)
        ma.init(jax.random.PRNGKey(1), jnp.ones((4,)))
