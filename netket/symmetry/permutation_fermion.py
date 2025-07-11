import jax
import jax.numpy as jnp
from netket.hilbert import SpinOrbitalFermions
from functools import partial
from jax.tree_util import register_pytree_node_class
import netket as nk



@jax.jit
def flatten_samples(n):
    # return x.reshape(-1, x.shape[-1])
    return jax.lax.collapse(n, 0, n.ndim - 1)





@jax.jit
def mergeCount(permutation: jax.Array) -> int:

    """Counts the number of inversions in a permutation using a variant of the merge
    sort algorithm. The complexity is O(n log n) where n is the size of the permutation."""

    if jnp.size(permutation) == 1:
        return 0

    mid = jnp.size(permutation) // 2
    left = permutation[:mid]
    right = permutation[mid:]

    invCountLeft = mergeCount(left)
    invCountRight = mergeCount(right)

    invCountCross = mergeAndCount(left,right)

    
    return (invCountLeft + invCountRight + invCountCross)&1


def mergeAndCount(left: jax.Array, right: jax.Array) -> int:

    n1 = jnp.size(left)
    n2 = jnp.size(right)

    def _cond_fun(val) -> bool:
        i, j, _ = val
        return jnp.logical_and(i < n1, j < n2)

    def _body_fun(val):
        i, j, count = val
        cond = left[i] <= right[j]
        i = jnp.where(cond, i + 1, i)
        j = jnp.where(cond, j, j + 1)
        count = jnp.where(cond, count, count + (n1 - i))
        return i, j, count

    _,_,count = jax.lax.while_loop(_cond_fun, _body_fun, (0, 0, 0))

    return count




#fonction to count the number of inversions in a permutation (parity of the permutation)
def get_parity(permutation: jax.Array) -> int:
    """Counts the parity of a permutation.
    This is the number of inversions in the permutation modulo 2.
    An inversion is a pair (i, j) such that i < j 
    and permutation[i] > permutation[j].
    """
    n = jnp.size(permutation)

    inversion_matrix = (permutation[:, None] > permutation[None, :])

    upper_triangular_mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)

    inversion_count = jnp.sum(inversion_matrix & upper_triangular_mask)

    return inversion_count&1



def occupied_orbitals(n: jax.Array, hilbert: SpinOrbitalFermions) -> jax.Array:
    """Returns the indices of the occupied orbitals in a given SpinOrbitalFermions state n"""
    
    R = n.nonzero(size=hilbert.n_fermions)[0]
    return R




@partial(jax.jit , static_argnames=("hilbert",))
@partial(jax.vmap, in_axes=(0, None,None))
def antisymmetric_sign(n: jax.Array, permutation: jax.Array, hilbert: SpinOrbitalFermions) -> int:
    """Returns the sign of a permutation in the SpinOrbitalFermions Hilbert space.
    The sign is given by the parity of the permutation restricted to the occupied orbitals.
    """

    R = occupied_orbitals(n, hilbert)
    permuted_indices = permutation[R]
    parity = get_parity(permuted_indices)
    sign = (-1)**parity

    return sign




@register_pytree_node_class
class PermutationOperatorFermion(nk.operator.DiscreteJaxOperator):
    """Be careful about the permutation being possibly inverted.
    """

    def __init__(self, hilbert: SpinOrbitalFermions, permutation: jax.Array):
        super().__init__(hilbert)
        self.permutation = permutation
        self.get_signs = antisymmetric_sign

        self.inverse_permutation = jnp.argsort(permutation)

    def tree_flatten(self):
        array_data = (self.permutation)
        struct_data = {'hilbert': self.hilbert}
        return array_data, struct_data

    @classmethod
    def tree_unflatten(cls, struct_data, array_data):
        ...
        return cls(array_data['hilbert'], ...)

    @property
    def max_conn_size(self) -> int:
        return 1

    def dtype(self):
        return int

    def get_conn_padded(self, x):
        r"""
        The action of a permutation operator on a state x is defined as Ps |x> = |x o s^{-1}> where 
        s^{-1} is the inverse permutation. This function computes <x|Ps = <x o s|, 
        which is the same as applying the permutation to the state x.

        For a fermionic state, one must also take into account the sign of the permutation,
        which is given by the parity of the inverse permutation restricted to the occupied orbitals.

        <n |P_s = <n o s | \xi_{s^{-1}}(n).
        """


        batch_shape, phys_dim = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, phys_dim)
        connected_elements = x.T[self.permutation].T
        print(connected_elements.shape)
        connected_elements = connected_elements.reshape((*batch_shape, 1, phys_dim))
        signs = self.get_signs(x, self.inverse_permutation, self.hilbert)
        return connected_elements, signs
