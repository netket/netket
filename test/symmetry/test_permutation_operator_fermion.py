import jax.numpy as jnp
import jax

from netket.symmetry.permutation_operator_fermion import get_antisymmetric_signs
from netket.symmetry.permutation_operator_fermion import PermutationOperatorFermion
from netket.hilbert import SpinOrbitalFermions
import netket as nk

seed = jax.random.PRNGKey(77)


def test_fermionic_sign():
    x = jnp.array(
        [
            [[0, 0, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1]],
            [[1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0, 1]],
        ],
        dtype=int,
    )

    permutation_array = jnp.array([7, 6, 3, 2, 0, 1, 5, 4], dtype=int)

    signs = jnp.array([[-1, 1], [1, 1]], dtype=int)

    assert jnp.all(get_antisymmetric_signs(x, permutation_array, n_fermions=4) == signs)


def test_perm_op_fermion():


    key = jax.random.PRNGKey(0)
    hilbert = SpinOrbitalFermions(n_orbitals=4, s=1/2, n_fermions_per_spin=(2, 2))
    x = hilbert.random_state(key, size=2)

    permutation_array = jax.random.permutation(key, hilbert.size)


    permutation = nk.utils.group.Permutation(
    permutation_array=permutation_array, name="test_permutation"
    )


    permop = PermutationOperatorFermion(hilbert, permutation)
    x_2, signs = permop.get_conn_padded(x) 

    assert jnp.all(x_2[0] == x[0][permutation_array]) 
    assert jnp.all(x_2[1] == x[1][permutation_array])

    assert jnp.all(signs == get_antisymmetric_signs(x, jnp.argsort(permutation_array), hilbert.n_fermions))
