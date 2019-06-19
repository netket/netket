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

hilberts = {}

# Spin 1/2
hilberts["Spin 1/2"] = nk.hilbert.Spin(
    s=0.5, graph=nk.graph.Hypercube(length=20, n_dim=1)
)

# Spin 1/2 with total Sz
hilberts["Spin 1/2 with total Sz"] = nk.hilbert.Spin(
    s=0.5, total_sz=1.0, graph=nk.graph.Hypercube(length=20, n_dim=1)
)

# Spin 3
hilberts["Spin 3"] = nk.hilbert.Spin(s=3, graph=nk.graph.Hypercube(length=25, n_dim=1))

# Boson
hilberts["Boson"] = nk.hilbert.Boson(
    n_max=5, graph=nk.graph.Hypercube(length=21, n_dim=1)
)

# Boson with total number
hilberts["Bosons with total number"] = nk.hilbert.Boson(
    n_max=5, n_bosons=11, graph=nk.graph.Hypercube(length=21, n_dim=1)
)

# Qubit
hilberts["Qubit"] = nk.hilbert.Qubit(graph=nk.graph.Hypercube(length=32, n_dim=1))

# Custom Hilbert
hilberts["Custom Hilbert"] = nk.hilbert.CustomHilbert(
    local_states=[-1232, 132, 0], graph=nk.graph.Hypercube(length=34, n_dim=1)
)

# Heisenberg 1d
g1 = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
hilberts["Heisenberg 1d"] = nk.hilbert.Spin(s=0.5, total_sz=0.0, graph=g1)

# Bose Hubbard
g3 = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
hilberts["Bose Hubbard"] = nk.hilbert.Boson(n_max=4, n_bosons=20, graph=g3)

#
# Small hilbert space tests
#

# Spin 1/2
hilberts["Spin 1/2 Small"] = nk.hilbert.Spin(
    s=0.5, graph=nk.graph.Hypercube(length=10, n_dim=1)
)

# Spin 3
hilberts["Spin 1/2 with total Sz Small"] = nk.hilbert.Spin(
    s=3, total_sz=1.0, graph=nk.graph.Hypercube(length=4, n_dim=1)
)

# Boson
hilberts["Boson Small"] = nk.hilbert.Boson(
    n_max=3, graph=nk.graph.Hypercube(length=5, n_dim=1)
)

# Qubit
hilberts["Qubit Small"] = nk.hilbert.Qubit(
    graph=nk.graph.Hypercube(length=1, n_dim=1, pbc=False)
)

# Custom Hilbert
hilberts["Custom Hilbert Small"] = nk.hilbert.CustomHilbert(
    local_states=[-1232, 132, 0], graph=nk.graph.Hypercube(length=5, n_dim=1)
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

    for name, hi in hilberts.items():
        assert hi.size > 0
        assert hi.local_size > 0
        assert len(hi.local_states) == hi.local_size

        if hi.is_discrete:
            rstate = np.zeros(hi.size)
            rg = nk.utils.RandomEngine(seed=1234)
            local_states = hi.local_states

            for i in range(100):
                hi.random_vals(rstate, rg)
                for state in rstate:
                    assert state in local_states


def test_hilbert_index():
    """"""

    for name, hi in hilberts.items():
        assert hi.size > 0
        assert hi.local_size > 0

        log_max_states = np.log(nk.hilbert.max_states)
        if hi.size * np.log(hi.local_size) < log_max_states:
            assert hi.is_indexable

            for k, state in enumerate(hi.states()):
                assert hi.state_to_number(state) == k
        else:
            assert not hi.is_indexable

            with pytest.raises(RuntimeError):
                hi.n_states

        # Check that a large hilbert space raises error when constructing matrices
        op = nk.operator.Heisenberg(
            hilbert=nk.hilbert.Spin(
                s=0.5, graph=nk.graph.Hypercube(length=100, n_dim=1)
            )
        )

        with pytest.raises(RuntimeError):
            m1 = op.to_dense()
        with pytest.raises(RuntimeError):
            m2 = op.to_sparse()


def test_state_iteration():
    g = nk.graph.Hypercube(10, 1)
    hilbert = nk.hilbert.Spin(g, s=0.5)

    reference = [np.array(el) for el in itertools.product([-1.0, 1.0], repeat=10)]

    for state, ref in zip(hilbert.states(), reference):
        assert np.allclose(state, ref)
