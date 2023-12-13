import netket as nk
import numpy as np
import netket.experimental as nkx
from netket.experimental.operator._fermion_operator_2nd_utils import (
    _convert_terms_to_spin_blocks,
)
from netket.experimental.operator.fermion import destroy, create, number

import pytest

op_ferm = {}
hi = nkx.hilbert.SpinOrbitalFermions(4)
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

op_ferm["float_round"] = (
    nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 1", "1^ 0"),
        weights=(0.15438015086600768 + 0j, 0.15438015086600765 + 0j),
    ),
    True,
)


@pytest.mark.parametrize(
    "op_ferm, is_hermitian",
    [pytest.param(op, is_herm, id=name) for name, (op, is_herm) in op_ferm.items()],
)
def test_is_hermitian_fermion2nd(op_ferm, is_hermitian):
    assert op_ferm.is_hermitian == is_hermitian


def test_fermion_operator_with_strings():
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    terms = (((0, 1), (2, 0)),)
    op1 = nkx.operator.FermionOperator2nd(hi, terms)
    op2 = nkx.operator.FermionOperator2nd(hi, ("0^ 2",))
    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())

    terms = (((0, 1), (1, 0)), ((2, 1), (1, 0)))
    weights = (0.5 - 0.5j, 0.5 + 0.5j)
    op1 = nkx.operator.FermionOperator2nd(hi, terms, weights)
    op2 = nkx.operator.FermionOperator2nd(hi, ("0^ 1", "2^ 1"), weights)
    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())

    terms = (((0, 1), (1, 0), (2, 1)), ((2, 1), (1, 0), (0, 1)))
    weights = (0.5 - 0.5j, 0.5 + 0.5j)
    op1 = nkx.operator.FermionOperator2nd(hi, terms, weights)
    op2 = nkx.operator.FermionOperator2nd(hi, ("0^ 1 2^", "2^ 1 0^"), weights)
    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())

    # make sure we do not throw away similar terms in the constructor
    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 2", "1^ 2"), weights=(1, -1), constant=2
    )
    op2 = nkx.operator.FermionOperator2nd(hi, terms=(), weights=(), constant=2)
    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())


def test_openfermion_conversion():
    # skip test if openfermion not installed
    pytest.importorskip("openfermion")
    from openfermion.ops import FermionOperator

    # FermionOperator
    of_fermion_operator = (
        FermionOperator("")  # todo
        + FermionOperator("0^ 3", 0.5 + 0.3j)
        + FermionOperator("3^ 0", 0.5 - 0.3j)
    )

    # no extra info given
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(of_fermion_operator)
    assert fo2.hilbert.size == 4

    # number of orbitals given
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(
        of_fermion_operator, n_orbitals=4
    )
    assert isinstance(fo2, nkx.operator.FermionOperator2nd)
    assert isinstance(fo2.hilbert, nkx.hilbert.SpinOrbitalFermions)
    assert fo2.hilbert.size == 4

    # with hilbert
    hilbert = nkx.hilbert.SpinOrbitalFermions(6)
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(hilbert, of_fermion_operator)
    assert fo2.hilbert == hilbert
    assert fo2.hilbert.size == 6

    # to check that the constraints are met (convention wrt ordering of states with different spin)
    from openfermion.hamiltonians import fermi_hubbard

    hilbert = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2, n_fermions_per_spin=(2, 1))
    of_fermion_operator = fermi_hubbard(1, 3, tunneling=1, coulomb=0, spinless=False)
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(
        hilbert, of_fermion_operator, convert_spin_blocks=True
    )
    assert fo2.hilbert.size == 3 * 2
    # will fail of we go outside of the allowed states with openfermion operators
    fo2.to_dense()


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
    np.testing.assert_array_equal(of_dense, fo_dense)
    # compare openfermion vs FermionOperator2nd
    np.testing.assert_array_equal(of_dense, fermop_dense)
    # compare from_openfermion vs FermionOperator 2nd
    np.testing.assert_array_equal(fo_dense, fermop_dense)

    # add a test from a non-hermitian operator
    of_fermion_operator = FermionOperator("") + FermionOperator(  # todo
        "0^ 2", 0.5 + 0.3j
    )
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(of_fermion_operator)
    fo_nk = nkx.operator.FermionOperator2nd(
        terms=["0^ 2"], weights=[0.5 + 0.3j], constant=1
    )
    assert np.array_equal(fo2.to_dense(), fo_nk.to_dense())


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
    np.testing.assert_allclose((op1 + op2).to_dense(), op3.to_dense())
    np.testing.assert_allclose(op4.to_dense(), op5.to_dense())

    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0"), weights=(1.0,), constant=0.3
    )
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0"), weights=(1.1,), constant=0.4
    )
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0"), weights=(2.1,), constant=0.7
    )
    np.testing.assert_allclose((op1 + op2).to_dense(), op3.to_dense())

    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "0^ 0"), weights=(1.0, 2.0), constant=0.3
    )
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0"), weights=(3.0,), constant=0.3
    )
    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())


def test_create_annihil_number():
    hi = nkx.hilbert.SpinOrbitalFermions(2)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0", "1^ 0"), weights=(0.3, 2))

    # without spin
    def c(site):
        return destroy(hi, site)

    def cdag(site):
        return create(hi, site)

    def cn(site):
        return number(hi, site)

    op2 = 0.3 * cn(0) + 2 * cdag(1) * c(0)
    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())

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
    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "1^ 2"), weights=(1 + 1j, 2 - 2j), constant=2
    )
    op4 = (1 + 1j) * cdag(0) * c(1) + (2 - 2j) * cdag(1) * c(2) + 2
    np.testing.assert_allclose(op3.to_dense(), op4.to_dense())

    # same, but with spin
    hi = nkx.hilbert.SpinOrbitalFermions(4, s=1 / 2)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0", "1^ 6"), weights=(0.3, 2))

    op2 = 0.3 * number(hi, 0, -1) + 2 * create(hi, 1, -1) * destroy(hi, 2, +1)
    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("4^ 1", "1^ 2"), weights=(1 + 1j, 2 - 2j), constant=2
    )
    op4 = (
        (1 + 1j) * create(hi, 0, +1) * destroy(hi, 1, -1)
        + (2 - 2j) * create(hi, 1, -1) * destroy(hi, 2, -1)
        + 2
    )
    np.testing.assert_allclose(op3.to_dense(), op4.to_dense())


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
    np.testing.assert_allclose(
        list(op2._operators.keys()), list(op2copy._operators.keys())
    )
    np.testing.assert_allclose(
        list(op2._operators.values()), list(op2copy._operators.values())
    )
    assert op2.is_hermitian == op2copy.is_hermitian
    np.testing.assert_allclose(op2.to_dense(), op2copy.to_dense())

    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(1.3, 1), constant=7.7
    )
    op12 = op1.copy()
    op12 += op2
    np.testing.assert_allclose((op1 + op2).to_dense(), op3.to_dense())
    np.testing.assert_allclose(op12.to_dense(), op3.to_dense())

    op4 = op3 * 2
    op5 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(2 * 1.3, 2 * 1), constant=2 * 7.7
    )
    op4b = op3.copy()
    op4b *= 2
    np.testing.assert_allclose(op4.to_dense(), op5.to_dense())
    np.testing.assert_allclose(op4b.to_dense(), op5.to_dense())

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
    np.testing.assert_allclose((op6 + op7).to_dense(), op8.to_dense())
    np.testing.assert_allclose(op67.to_dense(), op8.to_dense())

    op8 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "2^ 3"), weights=(1 + 1j, 2 - 0.5j), constant=1.0 + 3j
    )
    op8_trueconj = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 0", "3^ 2"), weights=(1 - 1j, 2 + 0.5j), constant=1.0 - 3j
    )
    np.testing.assert_allclose(op8.conjugate().to_dense(), op8_trueconj.to_dense())

    op9 = nkx.operator.FermionOperator2nd(
        hi, terms=("",), weights=(1,), constant=2, dtype=complex
    )
    op10 = nkx.operator.FermionOperator2nd(hi, constant=3)
    np.testing.assert_allclose(op9.to_dense(), op10.to_dense())


def test_fermion_remove_zeros():
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op1_orig = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "0^ 0", "1^", "1^"), weights=(1.0, 2.0, 1.0, -1.0)
    )
    op1 = op1_orig._remove_zeros()
    op2 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0",), weights=(3.0,))
    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())

    op2 = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 0 1^ 1", "0^ 0 0^ 2", "1^ 2 1^ 1", "1^ 2 0^ 2"),
        weights=(0.3 * (1 + 1j), 0.3 * 0.5, 2 * (1 + 1j), 2 * 0.5),
    )

    np.testing.assert_allclose(op2._remove_zeros().to_dense(), op2.to_dense())

    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0",), weights=(1.0,), constant=2.0, dtype=complex
    )
    op4 = op3 * 0.0
    op5 = nkx.operator.FermionOperator2nd(
        hi, terms=[], weights=[], constant=0.0, dtype=complex
    )
    np.testing.assert_allclose(op4.to_dense(), op5.to_dense())


def test_fermion_op_matmul():
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0", "1^ 2"), weights=(0.3, 2))

    # multiply with a real constant
    op_real = nkx.operator.FermionOperator2nd(hi, [], [], constant=2.0)
    np.testing.assert_allclose((op1 @ op_real).to_dense(), (op1 * 2).to_dense())
    np.testing.assert_allclose((op1 * op_real).to_dense(), (op1 * 2).to_dense())

    # multiply with a real+complex constant
    op_complex = nkx.operator.FermionOperator2nd(hi, [], [], constant=2.0 + 2j)
    np.testing.assert_allclose(
        (op1 @ op_complex).to_dense(), (op1 * (2 + 2j)).to_dense()
    )
    np.testing.assert_allclose(
        (op1 * op_complex).to_dense(), (op1 * (2 + 2j)).to_dense()
    )

    # multiply with another operator
    op_cr = nkx.operator.FermionOperator2nd(hi, terms=("1^",), weights=(2.0,))
    op_an = nkx.operator.FermionOperator2nd(
        hi,
        terms=("1"),
        weights=(4.0,),
    )
    op_num = nkx.operator.FermionOperator2nd(hi, terms=("1^ 1",), weights=(8.0,))

    np.testing.assert_allclose(
        (op_cr @ op_an).to_dense(),
        op_num.to_dense(),
    )

    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 1", "0^ 2"), weights=(1 + 1j, 0.5)
    )
    op2b = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 0 1^ 1", "0^ 0 0^ 2", "1^ 2 1^ 1", "1^ 2 0^ 2"),
        weights=(0.3 * (1 + 1j), 0.3 * 0.5, 2 * (1 + 1j), 2 * 0.5),
    )
    np.testing.assert_allclose(
        (op1 @ op2).to_dense(),
        op2b.to_dense(),
    )
    np.testing.assert_allclose(
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
    np.testing.assert_allclose(
        (op1 @ op3).to_dense(),
        op3b.to_dense(),
    )
    np.testing.assert_allclose(
        (op1 * op3).to_dense(),
        op3b.to_dense(),
    )

    hi = nkx.hilbert.SpinOrbitalFermions(1)
    op4a = nkx.operator.FermionOperator2nd(hi, terms=("0",), weights=(1,))
    op4b = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0",), weights=(1,))
    op4 = nkx.operator.FermionOperator2nd(hi, terms=("0 0^ 0",), weights=(1,))
    np.testing.assert_allclose(
        (op4a * op4b).to_dense(),
        op4.to_dense(),
    )
    op4 = nkx.operator.FermionOperator2nd(hi, terms=("0 0^ 0",), weights=(1,))
    np.testing.assert_allclose(
        (op4a * op4b).to_dense(),
        op4.to_dense(),
    )

    hi = nkx.hilbert.SpinOrbitalFermions(2)
    op5a = nkx.operator.FermionOperator2nd(hi, terms=("1^ 0",), weights=(1,))
    op5b = nkx.operator.FermionOperator2nd(hi, terms=("0 1",), weights=(1,))
    op5 = nkx.operator.FermionOperator2nd(hi)
    np.testing.assert_allclose(
        (op5a * op5b).to_dense(),
        op5.to_dense(),
    )


def test_fermion_add_sub_mul():
    # check addition
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2"), weights=(0.3, 2), constant=2
    )
    np.testing.assert_allclose((op1 + op1).to_dense(), 2 * op1.to_dense())

    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "0^ 1"), weights=(0.5, 4j), constant=1
    )
    op2b = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2", "0^ 1"), weights=(0.3 + 0.5, 2, 4j), constant=3
    )
    np.testing.assert_allclose((op1 + op2).to_dense(), op2b.to_dense())
    op2c = op2.copy()
    op2c += op1
    np.testing.assert_allclose(op2c.to_dense(), op2b.to_dense())

    # check subtraction
    op2d = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2", "0^ 1"), weights=(0.3 - 0.5, 2, -4j), constant=1
    )
    np.testing.assert_allclose((op1 - op2).to_dense(), op2d.to_dense())
    op1c = op1.copy()
    op1c -= op2
    np.testing.assert_allclose(op1c.to_dense(), op2d.to_dense())

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
    np.testing.assert_allclose((op1 * 10).to_dense(), op1f.to_dense())
    np.testing.assert_allclose(op1c.to_dense(), op1f.to_dense())


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
    up = +1
    down = -1
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

    np.testing.assert_allclose(eig_conv, eig)
    np.testing.assert_allclose(op.to_dense(), op_conv.to_dense())


def test_identity_zero():
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op0 = nkx.operator.fermion.zero(hi)
    op1 = nkx.operator.fermion.identity(hi)

    np.testing.assert_allclose(op0.to_dense(), np.zeros((2**3, 2**3)))
    np.testing.assert_allclose(op1.to_dense(), np.identity(2**3))
    hi = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2)
    op0 = nkx.operator.fermion.zero(hi)
    op1 = nkx.operator.fermion.identity(hi)

    np.testing.assert_allclose(op0.to_dense(), np.zeros((8**2, 8**2)))
    np.testing.assert_allclose(op1.to_dense(), np.identity(8**2))


def test_fermion_max_conn_size():
    def _compute_max_conn_size(op):
        mat = op.to_dense()
        mat = ~np.isclose(mat, 0)
        conn = np.sum(mat, axis=-1)
        return np.max(conn)

    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 1"), weights=(0.3, 2), constant=2
    )
    assert op.max_conn_size <= 3
    assert _compute_max_conn_size(op) == 1

    op = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 1"), weights=(0.3, 2), constant=0
    )
    assert op.max_conn_size == 1
    assert _compute_max_conn_size(op) == 1

    op = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "1^ 0"), weights=(1, 1), constant=0
    )
    assert op.max_conn_size <= 2
    assert _compute_max_conn_size(op) == 1

    op = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "1^ 0"), weights=(1, 0.5), constant=0
    )
    assert op.max_conn_size <= 2
    assert _compute_max_conn_size(op) == 1

    op = nkx.operator.FermionOperator2nd(
        hi, terms=("0 1^",), weights=(0.3,), constant=0
    )
    assert op.max_conn_size <= 1
    assert _compute_max_conn_size(op) == 1

    op = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0 1^", "1 0^ 0"), weights=(0.3, 0.3), constant=0
    )
    assert op.max_conn_size <= 2
    assert _compute_max_conn_size(op) == 1


def test_openfermion_conversion_2():
    # skip test if openfermion not installed
    pytest.importorskip("openfermion")
    from openfermion.ops import QubitOperator, FermionOperator

    # first term is a constant
    of_qubit_operator = (
        QubitOperator("") + 0.5 * QubitOperator("X0 X3") + 0.3 * QubitOperator("Z0")
    )

    # no extra info given
    ps = nk.operator.PauliStrings.from_openfermion(of_qubit_operator)
    assert isinstance(ps, nk.operator.PauliStrings)
    assert isinstance(ps.hilbert, nk.hilbert.Qubit)
    assert ps.hilbert.size == 4

    # number of qubits given
    ps = nk.operator.PauliStrings.from_openfermion(of_qubit_operator, n_qubits=6)
    assert isinstance(ps, nk.operator.PauliStrings)
    # check default
    assert isinstance(ps.hilbert, nk.hilbert.Qubit)
    assert ps.hilbert.size == 6

    # with hilbert
    hilbert = nk.hilbert.Spin(1 / 2, 6)
    ps = nk.operator.PauliStrings.from_openfermion(hilbert, of_qubit_operator)
    assert ps.hilbert == hilbert
    assert ps.hilbert.size == 6

    # FermionOperator
    of_fermion_operator = (
        FermionOperator("")  # todo
        + FermionOperator("0^ 3", 0.5 + 0.3j)
        + FermionOperator("3^ 0", 0.5 - 0.3j)
    )

    # no extra info given
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(of_fermion_operator)
    assert fo2.hilbert.size == 4

    # number of orbitals given
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(
        of_fermion_operator, n_orbitals=4
    )
    assert isinstance(fo2, nkx.operator.FermionOperator2nd)
    assert isinstance(fo2.hilbert, nkx.hilbert.SpinOrbitalFermions)
    assert fo2.hilbert.size == 4

    # with hilbert
    hilbert = nkx.hilbert.SpinOrbitalFermions(6)
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(hilbert, of_fermion_operator)
    assert fo2.hilbert == hilbert
    assert fo2.hilbert.size == 6

    # to check that the constraints are met (convention wrt ordering of states with different spin)
    from openfermion.hamiltonians import fermi_hubbard

    hilbert = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2, n_fermions_per_spin=(2, 1))
    of_fermion_operator = fermi_hubbard(1, 3, tunneling=1, coulomb=0, spinless=False)
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(
        hilbert, of_fermion_operator, convert_spin_blocks=True
    )
    assert fo2.hilbert.size == 3 * 2
    # will fail of we go outside of the allowed states with openfermion operators
    fo2.to_dense()


def test_fermion_matrices():
    # hard-code some common operator matrices that must be obtained

    # without fermion constraints
    hi = nkx.hilbert.SpinOrbitalFermions(2)
    op = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0",), weights=(2,))
    mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    np.testing.assert_allclose(mat, op.to_dense())

    hi = nkx.hilbert.SpinOrbitalFermions(2)
    op = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0 1^ 1",), weights=(2,))
    mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    np.testing.assert_allclose(mat, op.to_dense())

    # test non hermitian (!!! convention is <x|O|x'> !!!)
    op = nkx.operator.FermionOperator2nd(hi, terms=("0^ 1", "1^ 0"), weights=(2, 1))
    mat = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_allclose(mat, op.to_dense())

    op = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0", "1^ 1"), weights=(2, 1))
    mat = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]])
    np.testing.assert_allclose(mat, op.to_dense())

    op = nkx.operator.FermionOperator2nd(hi, terms=("0^",), weights=(2,))
    mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0]])
    np.testing.assert_allclose(mat, op.to_dense())

    # check the jordan-wigner sign !
    op = nkx.operator.FermionOperator2nd(hi, terms=("1^",), weights=(2,))
    mat = np.array([[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, -2, 0]])
    np.testing.assert_allclose(mat, op.to_dense())

    # check the jordan-wigner sign !
    op = nkx.operator.FermionOperator2nd(hi, terms=("1^ 0^",), weights=(2 + 1j,))
    mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-(2 + 1j), 0, 0, 0]])
    np.testing.assert_allclose(mat, op.to_dense())

    op = nkx.operator.FermionOperator2nd(hi, terms=("0^ 1^",), weights=(2 + 1j,))
    mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [+(2 + 1j), 0, 0, 0]])
    np.testing.assert_allclose(mat, op.to_dense())

    # with fermion constraints
    hi = nkx.hilbert.SpinOrbitalFermions(2, n_fermions=1)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 1", "1^ 0"), weights=(2, 1))
    mat1 = np.array(
        [
            [0, 1],
            [2, 0],
        ]
    )
    np.testing.assert_allclose(mat1, op1.to_dense())


def test_fermion_mode_indices():
    hi = nkx.hilbert.SpinOrbitalFermions(5)
    nkx.operator.FermionOperator2nd(hi, terms=("0^ 4", "2", "3"))

    with pytest.raises(ValueError):
        nkx.operator.FermionOperator2nd(hi, terms=("0^ 5",))
    with pytest.raises(ValueError):
        nkx.operator.FermionOperator2nd(hi, terms=(((-1, 0)),))


def test_fermion_create_annihilate():
    # testing the example
    hi = nkx.hilbert.SpinOrbitalFermions(2, s=1 / 2)

    with pytest.raises(IndexError):
        c1 = nkx.operator.fermion.create(hi, 2, sz=-1 / 2)  # index not in hilbert

    c1 = nkx.operator.fermion.create(hi, 1, sz=-1)
    c2 = nkx.operator.FermionOperator2nd(hi, terms=("1^",))
    np.testing.assert_allclose(c1.to_dense(), c2.to_dense())

    c1 = nkx.operator.fermion.destroy(hi, 1, sz=+1)
    c2 = nkx.operator.FermionOperator2nd(hi, terms=("3",))
    np.testing.assert_allclose(c1.to_dense(), c2.to_dense())

    c1 = nkx.operator.fermion.number(hi, 0, sz=-1)
    c2 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0",))
    np.testing.assert_allclose(c1.to_dense(), c2.to_dense())


def test_fermi_hubbard():
    L = 4  # take a 2x2 lattice
    D = 2
    t = 1  # tunneling/hopping
    U = 0.01  # coulomb

    # create the graph our fermions can hop on
    g = nk.graph.Hypercube(length=L, n_dim=D, pbc=True)
    n_sites = g.n_nodes

    # create a hilbert space with 2 up and 2 down spins
    hi = nkx.hilbert.SpinOrbitalFermions(n_sites, s=1 / 2, n_fermions_per_spin=(2, 2))

    # create an operator representing fermi hubbard interactions
    # -t (i^ j + h.c.) + U (i^ i j^ j)
    # we will create a helper function to abbreviate the creation, destruction and number operators
    # each operator has a site and spin projection (sz) in order to find the right position in the hilbert space samples
    def c(site, sz):
        return nkx.operator.fermion.create(hi, site, sz=sz)

    def cdag(site, sz):
        return nkx.operator.fermion.destroy(hi, site, sz=sz)

    def nc(site, sz):
        return nkx.operator.fermion.number(hi, site, sz=sz)

    up = +1
    down = -1
    ham = 0.0
    for sz in (up, down):
        for u, v in g.edges():
            ham += -t * cdag(u, sz) * c(v, sz) - t * cdag(v, sz) * c(u, sz)
    for u in g.nodes():
        ham += U * nc(u, up) * nc(u, down)

    print("Hamiltonian =", ham.operator_string())


def test_fermion_reduce():
    from netket.experimental.operator._fermion_operator_2nd_utils import _dict_compare

    hi = nkx.hilbert.SpinOrbitalFermions(2)

    # order
    op1 = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 1", "0^ 1", "0^ 1^", "0 1^", "1 1^"),
        weights=(1, 1, 3, 4j, 7j),
        constant=1,
    )
    op1_ordered = op1.copy()
    op1_ordered.reduce(order=True)
    op2 = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 1", "1^ 0^", "1^ 0", "1^ 1"),
        weights=(2, -3, -4j, -7j),
        constant=1 + 7j,
    )
    np.testing.assert_allclose(op1_ordered.to_dense(), op1.to_dense())
    np.testing.assert_allclose(op1_ordered.to_dense(), op2.to_dense())
    _dict_compare(op1_ordered.operators, op2.operators)

    # no ordering
    op1 = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 1", "0^ 1", "0^ 1^", "0 1^", "1 1^"),
        weights=(1, 1, 0, 4j, 7j),
        constant=1,
    )
    op1_operators = op1.operators.copy()
    op1_ordered = op1.copy()
    op1_ordered.reduce(order=False)
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "0 1^", "1 1^"), weights=(2, 4j, 7j), constant=1
    )
    np.testing.assert_allclose(op1_ordered.to_dense(), op1.to_dense())
    np.testing.assert_allclose(op1_ordered.to_dense(), op2.to_dense())
    _dict_compare(op1_ordered.operators, op2.operators)
    _dict_compare(op1_operators, op1.operators)  # check they didn't change

    # manually check some ordering
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op1 = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0 0^ 0 0^ 1^ 1 0^ 0 2 2^ 2 2^ 2 2^",),
        constant=1,
    )
    op1_ordered = op1.to_normal_order()
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("2^ 1^ 0^ 2 1 0", "1^ 0^ 1 0"), weights=(-1, 1), constant=1
    )
    np.testing.assert_allclose(op1_ordered.to_dense(), op2.to_dense())
    _dict_compare(op1_ordered.operators, op2.operators)

    op1_ordered = op1.to_pair_order()
    op2 = nkx.operator.FermionOperator2nd(
        hi,
        terms=(
            "2^ 1^ 0^ 2 1 0",
            "1^ 0^ 1 0",
        ),
        weights=(-1, 1),
        constant=1,
    )
    print("op1 = ", op1.operator_string())
    print("op1ordered = ", op1_ordered.operator_string())
    np.testing.assert_allclose(op1_ordered.to_dense(), op2.to_dense())
    _dict_compare(op1_ordered.operators, op2.operators)
    for op_term in ["0^ 0", "1^ 1", "0^ 1", "1^ 0", "0 0", "0 1^", "1 1^"]:
        op1 = nkx.operator.FermionOperator2nd(
            hi,
            terms=(op_term,),
            constant=0,
        )
        op1_ordered = op1.to_normal_order()
        np.testing.assert_allclose(op1_ordered.to_dense(), op1.to_dense())


def test_fermion_ordering():
    from netket.experimental.operator._fermion_operator_2nd_utils import _dict_compare

    hi = nkx.hilbert.SpinOrbitalFermions(2)
    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "0^ 1^", "0 1^", "1 1^"), weights=(2, 3, 4j, 7j), constant=1
    )
    op1_ordered = op1.to_normal_order()
    op2 = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 1", "1^ 0^", "1^ 0", "1^ 1"),
        weights=(2, -3, -4j, -7j),
        constant=1 + 7j,
    )
    np.testing.assert_allclose(op1_ordered.to_dense(), op1.to_dense())
    np.testing.assert_allclose(op1_ordered.to_dense(), op2.to_dense())
    _dict_compare(op1_ordered.operators, op2.operators)

    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "0^ 1^", "0 1^", "1 1^"), weights=(2, 3, 4j, 7j), constant=1
    )
    op1_ordered = op1.to_pair_order()
    op2 = nkx.operator.FermionOperator2nd(
        hi,
        terms=("1 0^", "1^ 0^", "1^ 0", "1^ 1"),
        weights=(-2, -3, -4j, -7j),
        constant=1 + 7j,
    )
    np.testing.assert_allclose(op1_ordered.to_dense(), op1.to_dense())
    np.testing.assert_allclose(op1_ordered.to_dense(), op2.to_dense())
    _dict_compare(op1_ordered.operators, op2.operators)
