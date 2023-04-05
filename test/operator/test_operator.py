import netket as nk
import numpy as np
import netket.experimental as nkx

import pytest
import jax

operators = {}

# Ising 1D
g = nk.graph.Hypercube(length=10, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
operators["Ising 1D"] = nk.operator.Ising(hi, g, h=1.321)

# Heisenberg 1D
g = nk.graph.Hypercube(length=10, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
operators["Heisenberg 1D"] = nk.operator.Heisenberg(hilbert=hi, graph=g)

# Bose Hubbard
g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
hi = nk.hilbert.Fock(n_max=3, n_particles=6, N=g.n_nodes)
operators["Bose Hubbard"] = nk.operator.BoseHubbard(U=4.0, hilbert=hi, graph=g)

g = nk.graph.Hypercube(length=3, n_dim=1, pbc=True)
hi = nk.hilbert.Fock(n_max=3, N=g.n_nodes)
operators["Bose Hubbard Complex"] = nk.operator.BoseHubbard(
    U=4.0, V=2.3, mu=-0.4, J=0.7, hilbert=hi, graph=g
)

# Graph Hamiltonian
N = 10
sigmax = np.asarray([[0, 1], [1, 0]])
mszsz = np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
edges = [[i, i + 1] for i in range(N - 1)] + [[N - 1, 0]]

g = nk.graph.Graph(edges=edges)
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)
operators["Graph Hamiltonian"] = nk.operator.GraphOperator(
    hi, g, site_ops=[sigmax], bond_ops=[mszsz]
)

g_sub = nk.graph.Graph(edges=edges[:3])  # edges of first four sites
operators["Graph Hamiltonian (on subspace)"] = nk.operator.GraphOperator(
    hi,
    g_sub,
    site_ops=[sigmax],
    bond_ops=[mszsz],
    acting_on_subspace=4,
)

# Graph Hamiltonian with colored edges
edges_c = [(i, j, i % 2) for i, j in edges]
g = nk.graph.Graph(edges=edges_c)
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)
operators["Graph Hamiltonian (colored edges)"] = nk.operator.GraphOperator(
    hi,
    g,
    site_ops=[sigmax],
    bond_ops=[1.0 * mszsz, 2.0 * mszsz],
    bond_ops_colors=[0, 1],
)

# Heisenberg with colored edges
operators["Heisenberg (colored edges)"] = nk.operator.Heisenberg(
    hi, g, J=[1, 2], sign_rule=[True, False]
)

# Custom Hamiltonian
sx = [[0, 1], [1, 0]]
sy = [[0, -1.0j], [1.0j, 0]]
sz = [[1, 0], [0, -1]]
g = nk.graph.Graph(edges=[[i, i + 1] for i in range(20)])
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)


def _loc(*args):
    return nk.operator.LocalOperator(hi, *args)


sx_hat = _loc([sx] * 3, [[0], [1], [5]])
sy_hat = _loc([sy] * 4, [[2], [3], [4], [9]])
szsz_hat = _loc(sz, [0]) @ _loc(sz, [1])
szsz_hat += _loc(sz, [4]) @ _loc(sz, [5])
szsz_hat += _loc(sz, [6]) @ _loc(sz, [8])
szsz_hat += _loc(sz, [7]) @ _loc(sz, [0])

operators["Custom Hamiltonian"] = sx_hat + sy_hat + szsz_hat
operators["Custom Hamiltonian Prod"] = sx_hat * 1.5 + (2.0 * sy_hat)

operators["Pauli Hamiltonian (XX)"] = nk.operator.PauliStrings(["XX"], [0.1])
operators["Pauli Hamiltonian (XX+YZ+IZ)"] = nk.operator.PauliStrings(
    ["XX", "YZ", "IZ"], [0.1, 0.2, -1.4]
)

hi = nkx.hilbert.SpinOrbitalFermions(5)
operators["FermionOperator2nd"] = nkx.operator.FermionOperator2nd(
    hi,
    terms=(((0, 1), (3, 0)), ((3, 1), (0, 0))),
    weights=(0.5 + 0.3j, 0.5 - 0.3j),  # must add h.c.
)

op_special = {}
for name, op in operators.items():
    if hasattr(op, "to_local_operator"):
        op_special[name] = op


# Operators that can be converted to a dense matrix without going OOM
op_finite_size = {}
for name, op in operators.items():
    hi = op.hilbert
    if hi.is_finite and hi.n_states < 2**14:
        op_finite_size[name] = op


@pytest.mark.parametrize("attr", ["get_conn", "get_conn_padded"])
@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_produce_elements_in_hilbert(op, attr):
    rng = nk.jax.PRNGSeq(0)
    hi = op.hilbert
    get_conn_fun = getattr(op, attr)

    assert len(hi.local_states) == hi.local_size
    assert hi.size > 0

    local_states = hi.local_states
    max_conn_size = op.max_conn_size
    rstates = hi.random_state(rng.next(), 1000)

    for i in range(len(rstates)):
        rstatet, mels = get_conn_fun(rstates[i])

        assert np.all(np.isin(rstatet, local_states))
        assert len(mels) <= max_conn_size


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_is_hermitean(op):
    rng = nk.jax.PRNGSeq(20)

    hi = op.hilbert
    assert len(hi.local_states) == hi.local_size

    rstates = hi.random_state(rng.next(), 100)
    for i in range(len(rstates)):
        rstate = rstates[i]
        rstatet, mels = op.get_conn(rstate)

        for k, state in enumerate(rstatet):

            invstates, mels1 = op.get_conn(state)

            found = False
            for kp, invstate in enumerate(invstates):
                if np.array_equal(rstate, invstate.flatten()):
                    found = True
                    np.testing.assert_allclose(mels1[kp], np.conj(mels[k]))
                    break

            assert found


@pytest.mark.parametrize(
    "op",
    [pytest.param(op, id=name) for name, op in operators.items()],
)
def test_lazy_hermitian(op):
    if op.is_hermitian:
        assert isinstance(op.H, type(op))
        assert op == op.H
    else:
        if np.issubdtype(op.dtype, np.complexfloating):
            assert isinstance(op.H, nk.operator.Adjoint)
        else:
            assert isinstance(op.H, nk.operator.Transpose)


# We don't return squared anymore
# @pytest.mark.parametrize(
#    "op",
#    [pytest.param(op, id=name) for name, op in op_finite_size.items()],
# )
# def test_lazy_squared(op):
#
#    op2 = op.H @ op
#    opd = op.to_dense()
#    op2d = opd.transpose().conjugate() @ opd
#    assert isinstance(op2, nk.operator.Squared)
#
#    op2_c = op2.collect()
#    assert isinstance(op2_c, nk.operator.AbstractOperator)
#    assert not isinstance(op2_c, nk.operator.Squared)
#    np.testing.assert_allclose(op2_c.to_dense(), op2d, atol=1e-13)


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_repr(op):
    assert type(op).__name__ in repr(op)


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_get_conn_numpy_closure(op):
    hi = op.hilbert
    closure = op._get_conn_flattened_closure()
    v = hi.random_state(jax.random.PRNGKey(0), 120)
    conn = np.empty(v.shape[0], dtype=np.intp)

    vp, mels = closure(np.asarray(v), conn)
    vp2, mels2 = op.get_conn_flattened(v, conn, pad=False)

    np.testing.assert_equal(vp, vp2)
    np.testing.assert_equal(mels, mels2)


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(s, id=f"shape={s}")
        for s in [
            (2,),
            (
                2,
                1,
            ),
            (2, 1, 1),
        ]
    ],
)
def test_get_conn_padded(op, shape, dtype):
    hi = op.hilbert

    v = hi.random_state(jax.random.PRNGKey(0), shape, dtype=dtype)

    vp, mels = op.get_conn_padded(v)

    assert vp.ndim == v.ndim + 1
    assert mels.ndim == v.ndim
    assert vp.dtype == v.dtype
    assert mels.dtype == op.dtype

    vp_f, mels_f = op.get_conn_padded(v.reshape(-1, hi.size))
    np.testing.assert_allclose(vp_f, vp.reshape(-1, *vp.shape[-2:]))
    np.testing.assert_allclose(mels_f, mels.reshape(-1, mels.shape[-1]))


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in op_special.items()]
)
def test_to_local_operator(op):
    op_l = op.to_local_operator()
    assert isinstance(op_l, nk.operator.LocalOperator)
    np.testing.assert_allclose(op.to_dense(), op_l.to_dense(), atol=1e-13)


def test_enforce_float_BoseHubbard():
    g = nk.graph.Hypercube(5, 1)
    hi = nk.hilbert.Fock(N=g.n_nodes, n_particles=3)
    op = nk.operator.BoseHubbard(hilbert=hi, graph=g, J=1, U=2, V=3, mu=4)
    assert np.issubdtype(op.dtype, np.floating)


def test_no_segfault():
    g = nk.graph.Hypercube(8, 1)
    hi = nk.hilbert.Spin(0.5, N=g.n_nodes)

    lo = nk.operator.LocalOperator(hi, [[1, 0], [0, 1]], [0])
    lo = lo.transpose()

    hi = None

    lo = lo @ lo

    assert True


def test_operator_on_subspace():
    hi = nk.hilbert.Spin(1 / 2, N=3) * nk.hilbert.Qubit(N=3)
    g = nk.graph.Chain(3, pbc=False)

    h1 = nk.operator.GraphOperator(hi, g, bond_ops=[mszsz], acting_on_subspace=0)
    assert h1.acting_on_subspace == list(range(3))
    assert nk.exact.lanczos_ed(h1)[0] == pytest.approx(-2.0)

    h2 = nk.operator.GraphOperator(hi, g, bond_ops=[mszsz], acting_on_subspace=3)
    assert h2.acting_on_subspace == list(range(3, 6))
    assert nk.exact.lanczos_ed(h2)[0] == pytest.approx(-2.0)

    h12 = h1 + h2
    assert sorted(h12.acting_on) == [(0, 1), (1, 2), (3, 4), (4, 5)]
    assert nk.exact.lanczos_ed(h12)[0] == pytest.approx(-4.0)

    h3 = nk.operator.GraphOperator(
        hi, g, bond_ops=[mszsz], acting_on_subspace=[0, 2, 4]
    )
    assert h3.acting_on_subspace == [0, 2, 4]
    assert nk.exact.lanczos_ed(h3)[0] == pytest.approx(-2.0)
    assert h3.acting_on == [(0, 2), (2, 4)]

    h4 = nk.operator.Heisenberg(hi, g, acting_on_subspace=0)
    assert h4.acting_on == h1.acting_on
