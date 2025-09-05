import pytest

import jax.numpy as jnp

import netket as nk


hilbert_space = nk.hilbert.Qubit(3)
graph = nk.graph.Chain(hilbert_space.size)

translation_1 = nk.utils.group.Permutation(
    permutation_array=jnp.array([1, 2, 0]), name="translation_1"
)
transposition_12 = nk.utils.group.Permutation(
    permutation_array=jnp.array([0, 2, 1]), name="transposition_12"
)

sigmax_local = nk.operator.spin.sigmax(hilbert_space, 0)
sigmaz_pauli = nk.operator.spin.sigmaz(hilbert_space, 0).to_pauli_strings()
ising = nk.operator.IsingJax(hilbert_space, graph, 1, 1)
translation_1_operator = nk.operator.permutation.PermutationOperator(
    hilbert_space, translation_1
)
transposition_12_operator = nk.operator.permutation.PermutationOperator(
    hilbert_space, transposition_12
)

operator_list = [
    sigmax_local,
    sigmaz_pauli,
    ising,
    translation_1_operator,
    transposition_12_operator,
]

coefficient_list = [[1, 1.0j], [1.0j, -jnp.real(3.5)]]


@pytest.mark.parametrize("operator_1", operator_list)
@pytest.mark.parametrize("operator_2", operator_list)
@pytest.mark.parametrize("coefficient", coefficient_list)
def test_operator_sum(operator_1, operator_2, coefficient):

    sum_op = coefficient[0] * operator_1 + coefficient[1] * operator_2

    sum_dense_1 = sum_op.to_dense()
    sum_dense_2 = (
        coefficient[0] * operator_1.to_dense() + coefficient[1] * operator_2.to_dense()
    )

    assert jnp.linalg.norm(sum_dense_1 - sum_dense_2) < 1e-14


def test_type_promotion():
    # Ensure that we treat dtypes correctly in basic mul
    hilbert_space = nk.hilbert.Qubit(3)
    op = nk.operator.permutation.PermutationOperator(
        hilbert_space,
        nk.utils.group.Permutation(permutation_array=jnp.array([1, 2, 0])),
    )
    assert op.dtype == jnp.float32

    op2 = op * 2.0
    assert op2.dtype == jnp.float32
    op2 = op * jnp.float64(2.0)
    assert op2.dtype == jnp.float64

    op2 = 2.0 * op
    assert op2.dtype == jnp.float32
    op2 = jnp.float64(2.0) * op
    assert op2.dtype == jnp.float64
