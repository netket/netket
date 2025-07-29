import jax.numpy as jnp


from netket.symmetry.permutation_operator_fermion import get_antisymmetric_signs


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
