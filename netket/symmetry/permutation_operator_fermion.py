import jax
import jax.numpy as jnp

from functools import partial
from jax.tree_util import register_pytree_node_class

from netket.hilbert import SpinOrbitalFermions
from netket.operator import DiscreteJaxOperator
from netket.utils.group import Permutation, Identity


def get_parity(array: jax.Array) -> jax.Array:
    """
    Count the parity of an array.
    This is the number of inversions in the array modulo 2.
    An inversion is a pair (i, j) such that i < j
    and array[i] > array[j].
    """
    batch_dims = array.shape[:-1]
    inversion_matrix = array[..., :, jnp.newaxis] > array[..., jnp.newaxis, :]
    upper_triangular_mask = jnp.triu(
        jnp.ones((*batch_dims, array.shape[-1], array.shape[-1]), dtype=bool), k=1
    )
    inversion_count = jnp.sum(inversion_matrix & upper_triangular_mask, axis=(-2, -1))
    return inversion_count % 2


def get_occupied_orbitals(x: jax.Array, n_fermions: int) -> jax.Array:
    """Return the indices of the occupied orbitals
    in a given SpinOrbitalFermions state n"""
    batch_dims, physical_dims = x.shape[:-1], x.shape[-1]
    x = x.reshape(-1, physical_dims)
    occupied_orbitals = _get_occupied_orbitals(x, n_fermions)
    return occupied_orbitals.reshape(*batch_dims, n_fermions)


@partial(jax.vmap, in_axes=(0, None))
def _get_occupied_orbitals(x: jax.Array, n_fermions: int) -> jax.Array:
    return x.nonzero(size=n_fermions)[0]


@partial(jax.jit, static_argnames=("n_fermions",))
def get_antisymmetric_signs(
    x: jax.Array, permutation: jax.Array, n_fermions: int
) -> jax.Array:
    """Return the sign of the permutation for a batch of fermionic Fock states x."""
    occupied = get_occupied_orbitals(x, n_fermions)
    permuted = permutation[occupied]
    parity = get_parity(permuted)
    sign = 1 - 2 * parity
    return sign


@register_pytree_node_class
class PermutationOperatorFermion(DiscreteJaxOperator):
    """Be careful about the permutation being possibly inverted."""

    def __init__(self, hilbert, permutation):

        assert isinstance(hilbert, SpinOrbitalFermions)

        super().__init__(hilbert)

        if isinstance(permutation, Identity):
            permutation = Permutation(
                permutation_array=jnp.arange(hilbert.size), name="Identity"
            )

        if not isinstance(permutation, Permutation):
            raise TypeError("permutation must be a Permutation object.")
        if not hilbert.size == permutation.permutation_array.size:
            raise ValueError(
                "Permutation size does not correspond to Hilbert space size."
            )

        self.permutation = permutation

    def tree_flatten(self):
        struct_data = {"hilbert": self.hilbert, "permutation": self.permutation}
        return (), struct_data

    @classmethod
    def tree_unflatten(cls, struct_data, array_data):
        return cls(**struct_data)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def dtype(self):
        return jnp.float32

    def __repr__(self):
        if self.permutation._name is not None:
            return f"PermutationOperatorFermion({self.permutation._name}: {self.permutation.permutation_array})"
        else:
            return f"PermutationOperatorFermion({self.permutation.permutation_array})"

    def __eq__(self, other):
        if isinstance(other, PermutationOperatorFermion):
            return (
                self.hilbert == other.hilbert and self.permutation == other.permutation
            )
        else:
            return False

    def __hash__(self):
        return hash((self.hilbert, self.permutation))

    def get_signs(self, x):
        return get_antisymmetric_signs(
            x, self.permutation.inverse_permutation_array, self.hilbert.n_fermions
        )

    def get_conn_padded(self, x):
        r"""
        This function computes <n|Ug = <n o g| \xi_{g^{-1}}(n).
        where n is a batch of fermionic Fock states,
        n o g are the permuted occupation numbers and
        \xi_{g^{-1}}(n) is the sign of the permutation.
        """

        batch_shape, phys_dim = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, phys_dim)
        connected_elements = x.T[self.permutation.permutation_array].T
        connected_elements = connected_elements.reshape((*batch_shape, 1, phys_dim))
        signs = self.get_signs(x)
        return connected_elements, signs

    def __matmul__(self, other):
        if isinstance(other, PermutationOperatorFermion):
            return PermutationOperatorFermion(
                self.hilbert, self.permutation @ other.permutation
            )
        else:
            return super().__mul__(other)
