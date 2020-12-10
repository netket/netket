# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
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

hilberts = {}

# Spin 1/2
hilberts["Spin 1/2"] = Spin(s=0.5, N=20)

# Spin 1/2 with total Sz
hilberts["Spin 1/2 with total Sz"] = Spin(s=0.5, total_sz=1.0, N=20)

# Spin 3
hilberts["Spin 3"] = Spin(s=3, N=25)

# Boson
hilberts["Boson"] = Boson(n_max=5, N=41)

# Boson with total number
hilberts["Bosons with total number"] = Boson(n_max=3, n_bosons=110, N=120)

# Qubit
hilberts["Qubit"] = nk.hilbert.Qubit(100)

# Custom Hilbert
hilberts["Custom Hilbert"] = CustomHilbert(local_states=[-1232, 132, 0], N=70)

# Heisenberg 1d
hilberts["Heisenberg 1d"] = Spin(s=0.5, total_sz=0.0, N=20)

# Bose Hubbard
hilberts["Bose Hubbard"] = Boson(n_max=4, n_bosons=20, N=20)

#
# Small hilbert space tests
#

# Spin 1/2
hilberts["Spin 1/2 Small"] = Spin(s=0.5, N=10)

# Spin 3
hilberts["Spin 1/2 with total Sz Small"] = Spin(s=3, total_sz=1.0, N=4)

# Boson
hilberts["Boson Small"] = Boson(n_max=3, N=5)

# Qubit
hilberts["Qubit Small"] = nk.hilbert.Qubit(N=1)

# Custom Hilbert
hilberts["Custom Hilbert Small"] = CustomHilbert(local_states=[-1232, 132, 0], N=5)

# Custom Hilbert
hilberts["Doubled Hilbert"] = nk.hilbert.DoubledHilbert(
    CustomHilbert(local_states=[-1232, 132, 0], N=5)
)


#
# Tests
#


def test_consistent_size():
    """"""

    for name, hi in hilberts.items():
        # print("Hilbert test: %s" % name)
        assert hi.size > 0
        assert hi.local_size > 0
        if hi.is_discrete:
            assert len(hi.local_states) == hi.local_size
            for state in hi.local_states:
                assert np.isfinite(state).all()


def test_random_states():
    """"""
    nk.random.seed(12345)

    for name, hi in hilberts.items():
        assert hi.size > 0
        assert hi.local_size > 0
        assert len(hi.local_states) == hi.local_size
        print("name", name)
        if hi.is_discrete:
            rstate = np.zeros(hi.size)
            local_states = hi.local_states
            for i in range(100):
                hi.random_state(out=rstate)
                for state in rstate:
                    assert state in local_states

            with pytest.raises(TypeError):
                hi.random_state(rstate)  # out is keyword-only


def test_hilbert_index():
    """"""
    for name, hi in hilberts.items():
        assert hi.size > 0
        assert hi.local_size > 0

        log_max_states = np.log(nk.hilbert.max_states)

        if hi.is_indexable:
            assert hi.size * np.log(hi.local_size) < log_max_states
            print(name, hi.n_states)
            assert np.allclose(hi.state_to_number(hi.all_states()), range(hi.n_states))

            # batched version of number to state
            n_few = min(hi.n_states, 100)
            few_states = np.zeros(shape=(n_few, hi.size))
            for k in range(n_few):
                few_states[k] = hi.number_to_state(k)

            print(name)
            assert np.allclose(
                hi.numbers_to_states(np.asarray(range(n_few))), few_states
            )

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


def test_graph_deprecation():
    g = nk.graph.Edgeless(3)

    with pytest.warns(FutureWarning):
        hilbert = Spin(s=0.5, graph=g)

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):
            hilbert = Spin(s=0.5, graph=g, N=3)
