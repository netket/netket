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

import numpy as np
import pytest

import netket as nk

from .. import common


def test_ising_int_dtype():
    H = nk.operator.Ising(nk.hilbert.Spin(0.5, 4), nk.graph.Chain(4), h=1, J=-1)
    H.to_local_operator()
    (H @ H).collect()


def test_ising_error():
    g = nk.graph.Hypercube(8, 1)
    with pytest.raises(TypeError):
        hi = nk.hilbert.Qubit(8)
        _ = nk.operator.Ising(hi, graph=g, h=1.0)

    with pytest.raises(ValueError):
        hi = nk.hilbert.Spin(1.0, 8)
        _ = nk.operator.Ising(hi, graph=g, h=1.0)

    with pytest.raises(ValueError):
        hi = nk.hilbert.Spin(1.0, 8)
        _ = nk.operator.IsingJax(hi, graph=g, h=1.0)


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


def _colored_graph(graph):
    return nk.graph.Graph(edges=[(i, j, i % 2) for i, j in graph.edges()])


@common.skipif_distributed
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
    ],
)
@pytest.mark.parametrize(
    "partial_H_pair",
    [
        pytest.param(
            (
                lambda hi, g: nk.operator.Ising(hi, g, h=0),
                lambda hi, g: nk.operator.IsingJax(hi, g, h=0),
            ),
            id="ising_zero_h",
        ),
        pytest.param(
            (
                lambda hi, g: nk.operator.Ising(hi, g, h=1),
                lambda hi, g: nk.operator.IsingJax(hi, g, h=1),
            ),
            id="ising",
        ),
        pytest.param(
            (
                lambda hi, g: nk.operator.PauliStrings(
                    hi,
                    [s + "I" * (g.n_nodes - len(s)) for s in ["XXI", "YZX", "IZX"]],
                    [0.1, 0.2, -1.4],
                ),
                lambda hi, g: nk.operator.PauliStringsJax(
                    hi,
                    [s + "I" * (g.n_nodes - len(s)) for s in ["XXI", "YZX", "IZX"]],
                    [0.1, 0.2, -1.4],
                ),
            ),
            id="pauli",
        ),
        pytest.param(
            (
                lambda hi, g: nk.operator.PauliStrings(
                    hi,
                    [s + "I" * (g.n_nodes - len(s)) for s in ["XXI", "YZY", "IZX"]],
                    [0.1, 0.2, -1.4],
                ),
                lambda hi, g: nk.operator.PauliStringsJax(
                    hi,
                    [s + "I" * (g.n_nodes - len(s)) for s in ["XXI", "YZY", "IZX"]],
                    [0.1, 0.2, -1.4],
                ),
            ),
            id="pauli_real",
        ),
    ],
)
@pytest.mark.parametrize(
    "partial_hilbert",
    [
        pytest.param(lambda g: nk.hilbert.Spin(0.5, g.n_nodes), id="spin"),
        pytest.param(lambda g: nk.hilbert.Qubit(g.n_nodes), id="qubit"),
    ],
)
@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(nk.graph.Hypercube(4, 1, pbc=False), id="4"),
        pytest.param(nk.graph.Hypercube(5, 1), id="5"),
        pytest.param(nk.graph.Hypercube(3, 2), id="3x3"),
        pytest.param(nk.graph.Hypercube(4, 2), id="4x4"),
    ],
)
def test_jax_conn(graph, partial_hilbert, partial_H_pair, dtype):
    hilbert = partial_hilbert(graph)
    H2 = partial_H_pair[1](hilbert, graph)

    if isinstance(hilbert, nk.hilbert.Qubit) and isinstance(H2, nk.operator.IsingJax):
        pytest.skip("The original Ising only supports Spin")

    H1 = partial_H_pair[0](hilbert, graph)

    σ = hilbert.random_state(nk.jax.PRNGKey(0), size=(10,), dtype=dtype)
    σp1, mels1 = H1.get_conn_padded(σ)
    σp2, mels2 = H2.get_conn_padded(σ)
    n_conn1 = H1.n_conn(σ)
    n_conn2 = H2.n_conn(σ)

    if not nk.config.netket_experimental_sharding:
        assert isinstance(σp1, np.ndarray)
        assert isinstance(mels1, np.ndarray)
        assert isinstance(n_conn1, np.ndarray)
    σp2 = np.asarray(σp2)
    mels2 = np.asarray(mels2)
    n_conn2 = np.asarray(n_conn2)

    assert n_conn1.shape == n_conn2.shape
    assert n_conn1.dtype == n_conn2.dtype
    # LocalOperator.n_conn and Ising.n_conn do not consider that mels[0] can be 0,
    # so n_conn1 can be n_conn2 + 1
    delta = n_conn1 - n_conn2
    assert (
        (delta == 0) | ((delta == 1) & (mels2[:, 0] == 0))
    ).all(), f"n_conn1:\n{n_conn1}\nn_conn2:\n{n_conn2}"

    # Shapes of σp and mels can be different because of padding
    assert σp1.dtype == σp2.dtype
    assert mels1.dtype == mels2.dtype

    # Filter out zero mels and sort σp lexicographically before comparing
    def canonize(σp, mels):
        assert σp.shape[0] == mels.size

        idx = mels != 0
        σp = σp[idx]
        mels = mels[idx]

        idx = np.lexsort(σp.T)
        σp = σp[idx]
        mels = mels[idx]
        return σp, mels

    # Compare each sample in the batch
    σp1 = σp1.reshape((-1, *σp1.shape[-2:]))
    σp2 = σp2.reshape((-1, *σp2.shape[-2:]))
    mels1 = mels1.reshape((-1, mels1.shape[-1]))
    mels2 = mels2.reshape((-1, mels2.shape[-1]))
    for σp1_i, σp2_i, mels1_i, mels2_i in zip(σp1, σp2, mels1, mels2):
        σp1_i, mels1_i = canonize(σp1_i, mels1_i)
        σp2_i, mels2_i = canonize(σp2_i, mels2_i)
        np.testing.assert_equal(np.asarray(σp1_i), np.asarray(σp2_i))
        np.testing.assert_equal(mels1_i, mels2_i)
