# Copyright 2022 The Netket Authors. - All Rights Reserved.
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

import netket as nk


def test_ising_int_dtype():
    H = nk.operator.Ising(nk.hilbert.Spin(0.5, 4), nk.graph.Chain(4), h=1, J=-1)
    H.to_local_operator()
    (H @ H).collect()


def test_Heisenberg():
    g = nk.graph.Hypercube(8, 1)
    hi = nk.hilbert.Spin(0.5) ** 8

    def gs_energy(ham):
        return nk.exact.lanczos_ed(ham)

    ha1 = nk.operator.Heisenberg(hi, graph=g)
    ha2 = nk.operator.Heisenberg(hi, graph=g, J=2.0)

    assert 2 * gs_energy(ha1) == pytest.approx(gs_energy(ha2))

    ha1 = nk.operator.Heisenberg(hi, graph=g, sign_rule=True)
    ha2 = nk.operator.Heisenberg(hi, graph=g, sign_rule=False)

    assert gs_energy(ha1) == pytest.approx(gs_energy(ha2))

    with pytest.raises(
        ValueError, match=r"sign_rule=True specified for a non-bipartite lattice"
    ):
        g = nk.graph.Hypercube(7, 1)
        hi = nk.hilbert.Spin(0.5, N=g.n_nodes)

        assert not g.is_bipartite()

        nk.operator.Heisenberg(hi, graph=g, sign_rule=True)

    L = 8
    edges = [(i, (i + 1) % L, 0) for i in range(L)] + [
        (i, (i + 2) % L, 1) for i in range(L)
    ]
    hi = nk.hilbert.Spin(0.5) ** L
    g = nk.graph.Graph(edges=edges)
    ha1 = nk.operator.Heisenberg(hi, graph=g, J=[1, 0.5])
    ha2 = nk.operator.Heisenberg(hi, graph=g, J=[1, 0.5], sign_rule=[True, False])

    assert gs_energy(ha1) == pytest.approx(gs_energy(ha2))
