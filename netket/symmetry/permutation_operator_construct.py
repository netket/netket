from netket.hilbert import AbstractHilbert, Qubit, Spin, Fock, SpinOrbitalFermions
from netket.symmetry import PermutationOperator, PermutationOperatorFermion

from netket.utils.group import Permutation


def construct_permutation_operator(
    hilbert_space: AbstractHilbert, permutation: Permutation
):
    """
    Return the appropriate permutation operator depending on the type of Hilbert space.
    If the Hilbert space is a spin or boson Hilbert space, a PermutationOperator will be returned,
    if it is a fermion Hilbert space, a PermutationOperatorFermion will be returned.
    """

    if isinstance(hilbert_space, (Qubit, Spin, Fock)):
        return PermutationOperator(hilbert_space, permutation)

    elif isinstance(hilbert_space, SpinOrbitalFermions):
        return PermutationOperatorFermion(hilbert_space, permutation)

    else:
        raise TypeError(
            "hilbert_space should be a Hilbert space of one of the following types:"
            "Qubit, Spin, Fock, SpinOrbitalFermions"
        )
