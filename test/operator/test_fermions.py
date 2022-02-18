import netket as nk
import numpy as np
import netket.experimental as nkx
from netket.experimental.operator._fermions_2nd import _convert_terms_to_spin_blocks
from netket.experimental.operator.fermion import destroy, create, number

import pytest

op_ferm = {}
hi = nkx.hilbert.SpinOrbitalFermions(3)
op_ferm["FermionOperator2nd_hermitian"] = (
    nkx.operator.FermionOperator2nd(
        hi, terms=(((0, 0), (1, 1)), ((1, 0), (0, 1))), weights=(1.0 + 1j, 1 - 1j)
    ),
    True,
)
op_ferm["FermionOperator2nd_not_hermitian"] = (
    nkx.operator.FermionOperator2nd(
        hi, terms=(((0, 0), (2, 1)), ((1, 0), (0, 1))), weights=(1.0 + 1j, 1 - 1j)
    ),
    False,
)

op_ferm["FermionOperator2nd_hermitian_3term"] = (
    nkx.operator.FermionOperator2nd(
        hi,
        (((0, 0), (1, 1), (2, 1)), ((2, 0), (1, 0), (0, 1))),
        weights=(1.0 - 1j, 1 + 1j),
    ),
    True,
)
op_ferm["FermionOperator2nd_not_hermitian_3term"] = (
    nkx.operator.FermionOperator2nd(
        hi,
        (((0, 0), (1, 1), (2, 1)), ((3, 0), (1, 0), (0, 1))),
        weights=(1.0 - 1j, 2 + 2j),
    ),
    False,
)

op_ferm["fermihubbard_int"] = (
    nkx.operator.FermionOperator2nd(
        hi,
        terms=(
            ((0, 1), (0, 0), (1, 1), (1, 0)),
            ((0, 1), (0, 0), (1, 1), (1, 0)),
            ((0, 1), (0, 0), (1, 1), (1, 0)),
            ((0, 1), (0, 0), (1, 1), (1, 0)),
        ),
        weights=(1.0, 1.0, 1.0, 1.0),
    ),
    True,
)

op_ferm["ordering"] = (
    nkx.operator.FermionOperator2nd(
        hi,
        terms=(((0, 1), (0, 0), (1, 1), (1, 0)), ((1, 1), (1, 0), (0, 1), (0, 0))),
        weights=(1.0 - 1j, 1 + 1j),
    ),
    True,
)


@pytest.mark.parametrize(
    "op_ferm, is_hermitian",
    [pytest.param(op, is_herm, id=name) for name, (op, is_herm) in op_ferm.items()],
)
def test_is_hermitian_fermion2nd(op_ferm, is_hermitian):
    print("OPERATOR", op_ferm._operators)
    assert op_ferm.is_hermitian == is_hermitian


def test_fermion_operator_with_strings():
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    terms = (((0, 1), (2, 0)),)
    op1 = nkx.operator.FermionOperator2nd(hi, terms)
    op2 = nkx.operator.FermionOperator2nd(hi, ("0^ 2",))
    assert np.allclose(op1.to_dense(), op2.to_dense())

    terms = (((0, 1), (1, 0)), ((2, 1), (1, 0)))
    weights = (0.5 - 0.5j, 0.5 + 0.5j)
    op1 = nkx.operator.FermionOperator2nd(hi, terms, weights)
    op2 = nkx.operator.FermionOperator2nd(hi, ("0^ 1", "2^ 1"), weights)
    assert np.allclose(op1.to_dense(), op2.to_dense())

    terms = (((0, 1), (1, 0), (2, 1)), ((2, 1), (1, 0), (0, 1)))
    weights = (0.5 - 0.5j, 0.5 + 0.5j)
    op1 = nkx.operator.FermionOperator2nd(hi, terms, weights)
    op2 = nkx.operator.FermionOperator2nd(hi, ("0^ 1 2^", "2^ 1 0^"), weights)
    assert np.allclose(op1.to_dense(), op2.to_dense())


def compare_openfermion_fermions():
    # skip test if openfermion not installed
    pytest.importorskip("openfermion")
    from openfermion import FermionOperator, get_sparse_operator

    # openfermion
    of = FermionOperator("0^ 1", 1.0) + FermionOperator("1^ 0", 1.0)
    of_dense = get_sparse_operator(of).todense()
    # from_openfermion
    fo = nkx.operator.FermionOperator2nd.from_openfermion(of)
    fo_dense = fo.to_dense()
    # FermionOperator2nd
    hi = nkx.hilbert.SpinOrbitalFermions(2)  # two sites
    fermop = nkx.operator.FermionOperator2nd(
        hi, terms=(((0, 1), (1, 0)), ((1, 1), (0, 0))), weights=(1.0, 1.0)
    )
    fermop_dense = fermop.to_dense()
    # compare openfermion vs from_openfermion
    assert np.array_equal(of_dense, fo_dense)
    # compare openfermion vs FermionOperator2nd
    assert np.array_equal(of_dense, fermop_dense)
    # compare from_openfermion vs FermionOperator 2nd
    assert np.array_equal(fo_dense, fermop_dense)


def test_add_fermions():
    hi = nkx.hilbert.SpinOrbitalFermions(5)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("1^ 2"), weights=(1,), constant=2)
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4"), weights=(1.3,), constant=5.7
    )
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(1.3, 1), constant=7.7
    )
    op4 = op3 * 2
    op5 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(2 * 1.3, 2 * 1), constant=2 * 7.7
    )
    assert np.allclose((op1 + op2).to_dense(), op3.to_dense())
    assert np.allclose(op4.to_dense(), op5.to_dense())


def test_create_annihil_number():
    hi = nkx.hilbert.SpinOrbitalFermions(5)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0", "1^ 2"), weights=(0.3, 2))

    # without spin
    def c(site):
        return destroy(hi, site)

    def cdag(site):
        return create(hi, site)

    def cn(site):
        return number(hi, site)

    op2 = 0.3 * cn(0) + 2 * cdag(1) * c(2)
    assert np.allclose(op1.to_dense(), op2.to_dense())
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "1^ 2"), weights=(1 + 1j, 2 - 2j), constant=2
    )
    op4 = (1 + 1j) * cdag(0) * c(1) + (2 - 2j) * cdag(1) * c(2) + 2
    assert np.allclose(op3.to_dense(), op4.to_dense())

    # same, but with spin
    hi = nkx.hilbert.SpinOrbitalFermions(4, s=1 / 2)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0", "1^ 6"), weights=(0.3, 2))

    op2 = 0.3 * number(hi, 0, -0.5) + 2 * create(hi, 1, -0.5) * destroy(hi, 2, +0.5)
    assert np.allclose(op1.to_dense(), op2.to_dense())
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("4^ 1", "1^ 2"), weights=(1 + 1j, 2 - 2j), constant=2
    )
    op4 = (
        (1 + 1j) * create(hi, 0, +0.5) * destroy(hi, 1, -0.5)
        + (2 - 2j) * create(hi, 1, -0.5) * destroy(hi, 2, -0.5)
        + 2
    )
    assert np.allclose(op3.to_dense(), op4.to_dense())


def test_operations_fermions():
    hi = nkx.hilbert.SpinOrbitalFermions(5)
    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 2",), weights=(1,), constant=2, dtype=complex
    )
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4"), weights=(1.3,), constant=5.7
    )
    op2copy = op2.copy()
    assert op2copy.hilbert == op2.hilbert
    assert np.allclose(list(op2._operators.keys()), list(op2copy._operators.keys()))
    assert np.allclose(list(op2._operators.values()), list(op2copy._operators.values()))
    assert op2.is_hermitian == op2copy.is_hermitian
    assert np.allclose(op2.to_dense(), op2copy.to_dense())

    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(1.3, 1), constant=7.7
    )
    op12 = op1.copy()
    op12 += op2
    assert np.allclose((op1 + op2).to_dense(), op3.to_dense())
    assert np.allclose(op12.to_dense(), op3.to_dense())

    op4 = op3 * 2
    op5 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(2 * 1.3, 2 * 1), constant=2 * 7.7
    )
    op4b = op3.copy()
    op4b *= 2
    assert np.allclose(op4.to_dense(), op5.to_dense())
    assert np.allclose(op4b.to_dense(), op5.to_dense())

    op6 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 2", "0^ 1"), weights=(1j, -1.0j), constant=7.7
    )
    op7 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 2", "0^ 1"), weights=(1, 1), constant=7.7
    )
    op8 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 2", "0^ 1"), weights=(1.0 + 1j, 1 - 1j), constant=2 * 7.7
    )
    op67 = op6.copy()
    op67 += op7
    assert np.allclose((op6 + op7).to_dense(), op8.to_dense())
    assert np.allclose(op67.to_dense(), op8.to_dense())

    op8 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "2^ 3"), weights=(1 + 1j, 2 - 0.5j), constant=1.0 + 3j
    )
    op8_trueconj = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 0", "3^ 2"), weights=(1 - 1j, 2 + 0.5j), constant=1.0 - 3j
    )
    assert np.allclose(op8.conjugate().to_dense(), op8_trueconj.to_dense())


def test_fermion_op_matmul():
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0", "1^ 2"), weights=(0.3, 2))

    # multiply with a real constant
    op_real = nkx.operator.FermionOperator2nd(hi, [], [], constant=2.0)
    assert np.allclose((op1 @ op_real).to_dense(), (op1 * 2).to_dense())
    assert np.allclose((op1 * op_real).to_dense(), (op1 * 2).to_dense())

    # multiply with a real+complex constant
    op_complex = nkx.operator.FermionOperator2nd(hi, [], [], constant=2.0 + 2j)
    assert np.allclose((op1 @ op_complex).to_dense(), (op1 * (2 + 2j)).to_dense())
    assert np.allclose((op1 * op_complex).to_dense(), (op1 * (2 + 2j)).to_dense())

    # multiply with another operator
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 1", "0^ 2"), weights=(1 + 1j, 0.5)
    )
    op2b = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 0 1^ 1", "0^ 0 0^ 2", "1^ 2 1^ 1", "1^ 2 0^ 2"),
        weights=(0.3 * (1 + 1j), 0.3 * 0.5, 2 * (1 + 1j), 2 * 0.5),
    )
    assert np.allclose(
        (op1 @ op2).to_dense(),
        op2b.to_dense(),
    )
    assert np.allclose(
        (op1 * op2).to_dense(),
        op2b.to_dense(),
    )

    # multiply with another operator + constant
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 1",), weights=(1 + 1j,), constant=5
    )
    op3b = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 0 1^ 1", "0^ 0", "1^ 2 1^ 1", "1^ 2"),
        weights=(0.3 * (1 + 1j), 5 * 0.3, 2 * (1 + 1j), 10),
        constant=0,
    )
    assert np.allclose(
        (op1 @ op3).to_dense(),
        op3b.to_dense(),
    )
    assert np.allclose(
        (op1 * op3).to_dense(),
        op3b.to_dense(),
    )


def test_fermion_add_sub_mul():
    # check addition
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2"), weights=(0.3, 2), constant=2
    )
    assert np.allclose((op1 + op1).to_dense(), 2 * op1.to_dense())

    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "0^ 1"), weights=(0.5, 4j), constant=1
    )
    op2b = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2", "0^ 1"), weights=(0.3 + 0.5, 2, 4j), constant=3
    )
    assert np.allclose((op1 + op2).to_dense(), op2b.to_dense())
    op2c = op2.copy()
    op2c += op1
    assert np.allclose(op2c.to_dense(), op2b.to_dense())

    # check substraction
    op2d = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2", "0^ 1"), weights=(0.3 - 0.5, 2, -4j), constant=1
    )
    assert np.allclose((op1 - op2).to_dense(), op2d.to_dense())
    op1c = op1.copy()
    op1c -= op2
    assert np.allclose(op1c.to_dense(), op2d.to_dense())

    # check multiplication with scalar
    op1f = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 0", "1^ 2"),
        weights=(
            3,
            20,
        ),
        constant=20,
    )
    op1c = op1.copy()
    op1c *= 10
    assert np.allclose((op1 * 10).to_dense(), op1f.to_dense())
    assert np.allclose(op1c.to_dense(), op1f.to_dense())


@pytest.mark.parametrize("dtype1", [np.float32, np.float64])
@pytest.mark.parametrize("dtype2", [np.float32, np.float64])
def test_dtype_promotion(dtype1, dtype2):
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2"), weights=(0.3, 2), constant=2, dtype=dtype1
    )
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1"), weights=(0.1,), constant=2, dtype=dtype2
    )

    assert op1.dtype == dtype1
    assert op2.dtype == dtype2
    assert op1.to_dense().dtype == dtype1
    assert op2.to_dense().dtype == dtype2

    assert (-op1).dtype == dtype1
    assert (-op2).dtype == dtype2

    assert (op1 + op2).dtype == np.promote_types(op1.dtype, op2.dtype)
    assert (op1 - op2).dtype == np.promote_types(op1.dtype, op2.dtype)
    assert (op1 @ op2).dtype == np.promote_types(op1.dtype, op2.dtype)

    a = np.array(0.5, dtype=dtype1)
    assert (op2 + a + op2).dtype == np.promote_types(a.dtype, op2.dtype)
    assert (op2 - a).dtype == np.promote_types(a.dtype, op2.dtype)
    assert (op2 * a).dtype == np.promote_types(a.dtype, op2.dtype)

    a = np.array(0.5, dtype=dtype2)
    assert (op1 + a).dtype == np.promote_types(op1.dtype, a.dtype)
    assert (op1 - a).dtype == np.promote_types(op1.dtype, a.dtype)
    assert (op1 * a).dtype == np.promote_types(op1.dtype, a.dtype)


def test_convert_to_spin_blocks():
    # skip test if openfermion not installed
    pytest.importorskip("openfermion")
    import openfermion

    hi = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2)
    term1 = (((0, 1), (1, 0)),)
    term1_conv = _convert_terms_to_spin_blocks(term1, 3, 2)
    assert term1_conv == (((0, 1), (3, 0)),)

    term2 = (((2, 1), (3, 0)), ((4, 1), (5, 0)))
    term2_conv = _convert_terms_to_spin_blocks(term2, 3, 2)
    assert term2_conv == (((1, 1), (4, 0)), ((2, 1), (5, 0)))

    term3 = (((0, 1), (0, 0), (1, 1), (1, 0)),)
    term3_conv = _convert_terms_to_spin_blocks(term3, 3, 2)
    assert term3_conv == (((0, 1), (0, 0), (3, 1), (3, 0)),)

    # check fermi-hubbard - netket
    L = 2
    D = 2
    t = 1  # tunneling/hopping
    U = 0.01  # coulomb
    # create the graph where fermions can hop on
    g = nk.graph.Hypercube(length=L, n_dim=D, pbc=True)
    Nsites = g.n_nodes
    hi = nkx.hilbert.SpinOrbitalFermions(Nsites, s=1 / 2)
    # create an operator representing fermi hubbard interactions
    up = +1 / 2
    down = -1 / 2
    terms = []
    weights = []
    for sz in (up, down):
        for u, v in g.edges():
            c_u = hi._get_index(u, sz)
            c_v = hi._get_index(v, sz)

            terms.append([(c_u, 1), (c_v, 0)])
            terms.append([(c_v, 1), (c_u, 0)])

            weights.append(-t)
            weights.append(-t)

    for u in g.nodes():
        nc_up = hi._get_index(u, up)
        nc_down = hi._get_index(u, down)

        terms.append([(nc_up, 1), (nc_up, 0), (nc_down, 1), (nc_down, 0)])
        weights.append(U)
    op = nkx.operator.FermionOperator2nd(hi, terms, weights)

    # eigenspectrum
    eig = np.linalg.eigvalsh(op.to_dense())

    # check fermi-hubbard - openfermion
    op_of = openfermion.fermi_hubbard(
        L, D, tunneling=t, coulomb=U, periodic=True, spinless=False
    )
    terms_conv = _convert_terms_to_spin_blocks(op_of.terms, Nsites, 2)
    op_conv = nkx.operator.FermionOperator2nd(
        hi, terms_conv, list(op_of.terms.values())
    )

    # eigenspectrum
    eig_conv = np.linalg.eigvalsh(op_conv.to_dense())

    assert np.allclose(eig_conv, eig)
    assert np.allclose(op.to_dense(), op_conv.to_dense())


def test_identity_zero():
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op0 = nkx.operator.fermion.zero(hi)
    op1 = nkx.operator.fermion.identity(hi)

    assert np.allclose(op0.to_dense(), np.zeros((2**3, 2**3)))
    assert np.allclose(op1.to_dense(), np.identity(2**3))
    hi = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2)
    op0 = nkx.operator.fermion.zero(hi)
    op1 = nkx.operator.fermion.identity(hi)

    assert np.allclose(op0.to_dense(), np.zeros(((8**2, 8**2))))
    assert np.allclose(op1.to_dense(), np.identity(8**2))
