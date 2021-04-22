# Copyright 2020 The Netket Authors. - All Rights Reserved.
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

import itertools
import netket as nk
import networkx as nx
import numpy as np
import pytest
from netket.hilbert import *

import jax
from jax import numpy as jnp

hilberts = {}

# Spin 1/2
hilberts["Spin 1/2"] = Spin(s=0.5, N=20)

# Spin 1/2 with total Sz
hilberts["Spin 1/2 with total Sz"] = Spin(s=0.5, total_sz=1.0, N=20)

# Spin 1/2 with total Sz
hilberts["Spin 1 with total Sz, even sites"] = Spin(s=1.0, total_sz=5.0, N=6)

# Spin 1/2 with total Sz
hilberts["Spin 1 with total Sz, odd sites"] = Spin(s=1.0, total_sz=2.0, N=7)

# Spin 3
hilberts["Spin 3"] = Spin(s=3, N=25)

# Boson
hilberts["Fock"] = Fock(n_max=5, N=41)

# Boson with total number
hilberts["Fock with total number"] = Fock(n_max=3, n_particles=110, N=120)

# Qubit
hilberts["Qubit"] = nk.hilbert.Qubit(100)

# Custom Hilbert
hilberts["Custom Hilbert"] = CustomHilbert(local_states=[-1232, 132, 0], N=70)

# Heisenberg 1d
hilberts["Heisenberg 1d"] = Spin(s=0.5, total_sz=0.0, N=20)

# Bose Hubbard
hilberts["Bose Hubbard"] = Fock(n_max=4, n_particles=20, N=20)

#
# Small hilbert space tests
#

# Spin 1/2
hilberts["Spin 1/2 Small"] = Spin(s=0.5, N=10)

# Spin 3
hilberts["Spin 1/2 with total Sz Small"] = Spin(s=3, total_sz=1.0, N=4)

# Boson
hilberts["Fock Small"] = Fock(n_max=3, N=5)

# Qubit
hilberts["Qubit Small"] = nk.hilbert.Qubit(N=1)

# Custom Hilbert
hilberts["Custom Hilbert Small"] = CustomHilbert(local_states=[-1232, 132, 0], N=5)

# Custom Hilbert
hilberts["Doubled Hilbert"] = nk.hilbert.DoubledHilbert(
    CustomHilbert(local_states=[-1232, 132, 0], N=5)
)

# hilberts["Tensor: Spin x Fock"] = Spin(s=0.5, N=4) * Fock(4, N=2)


#
# Tests
#
@pytest.mark.parametrize(
    "hi", [pytest.param(hi, id=name) for name, hi in hilberts.items()]
)
def test_consistent_size(hi):
    assert hi.size > 0
    assert hi.local_size > 0
    if hi.is_discrete:
        assert len(hi.local_states) == hi.local_size
        for state in hi.local_states:
            assert np.isfinite(state).all()


@pytest.mark.parametrize(
    "hi", [pytest.param(hi, id=name) for name, hi in hilberts.items()]
)
def test_random_states(hi):
    assert hi.size > 0
    assert hi.local_size > 0
    assert len(hi.local_states) == hi.local_size

    if hi.is_discrete:
        local_states = hi.local_states
        for i in range(100):
            rstate = hi.random_state(jax.random.PRNGKey(i * 14))
            for state in rstate:
                assert state in local_states

        assert hi.random_state(jax.random.PRNGKey(13)).shape == (hi.size,)
        assert (
            hi.random_state(jax.random.PRNGKey(13), dtype=np.float32).dtype
            == np.float32
        )
        assert (
            hi.random_state(jax.random.PRNGKey(13), dtype=np.complex64).dtype
            == np.complex64
        )
        assert hi.random_state(jax.random.PRNGKey(13), 10).shape == (10, hi.size)
        assert hi.random_state(jax.random.PRNGKey(13), size=10).shape == (10, hi.size)
        # assert hi.random_state(jax.random.PRNGKey(13), size=(10,)).shape == (10, hi.size)
        # assert hi.random_state(jax.random.PRNGKey(13), size=(10, 2)).shape == (10, 2, hi.size)


@pytest.mark.parametrize(
    "hi", [pytest.param(hi, id=name) for name, hi in hilberts.items()]
)
def test_random_states_legacy(hi):
    """"""
    nk.legacy.random.seed(12345)

    assert hi.size > 0
    assert hi.local_size > 0
    assert len(hi.local_states) == hi.local_size

    if hi.is_discrete:
        rstate = np.zeros(hi.size)
        local_states = hi.local_states
        for i in range(100):
            hi.random_state(out=rstate)
            for state in rstate:
                assert state in local_states

        assert hi.random_state().shape == (hi.size,)
        assert hi.random_state(10).shape == (10, hi.size)
        assert hi.random_state(size=10).shape == (10, hi.size)
        assert hi.random_state(size=(10,)).shape == (10, hi.size)
        assert hi.random_state(size=(10, 2)).shape == (10, 2, hi.size)


@pytest.mark.parametrize(
    "hi", [pytest.param(hi, id=name) for name, hi in hilberts.items()]
)
def test_hilbert_index(hi):
    """"""
    assert hi.size > 0
    assert hi.local_size > 0

    log_max_states = np.log(nk.hilbert._abstract_hilbert.max_states)

    if hi.is_indexable:
        assert hi.size * np.log(hi.local_size) < log_max_states
        assert np.allclose(hi.states_to_numbers(hi.all_states()), range(hi.n_states))

        # batched version of number to state
        n_few = min(hi.n_states, 100)
        few_states = np.zeros(shape=(n_few, hi.size))
        for k in range(n_few):
            few_states[k] = hi.numbers_to_states(k)

        assert np.allclose(hi.numbers_to_states(np.asarray(range(n_few))), few_states)

    else:
        assert not hi.is_indexable

        with pytest.raises(RuntimeError):
            hi.n_states

    # Check that a large hilbert space raises error when constructing matrices
    g = nk.graph.Hypercube(length=100, n_dim=1)
    op = nk.operator.Heisenberg(hilbert=Spin(s=0.5, N=g.n_nodes), graph=g)

    with pytest.raises(RuntimeError):
        m1 = op.to_dense()
    with pytest.raises(RuntimeError):
        m2 = op.to_sparse()


def test_state_iteration():
    hilbert = Spin(s=0.5, N=10)

    reference = [np.array(el) for el in itertools.product([-1.0, 1.0], repeat=10)]

    for state, ref in zip(hilbert.states(), reference):
        assert np.allclose(state, ref)


def test_deprecations():
    g = nk.graph.Edgeless(3)

    with pytest.warns(FutureWarning):
        hilbert = Spin(s=0.5, graph=g)

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):
            hilbert = Spin(s=0.5, graph=g, N=3)
