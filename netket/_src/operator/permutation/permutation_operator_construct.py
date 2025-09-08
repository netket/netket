# Copyright 2025 The NetKet Authors - All rights reserved.
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

from netket.hilbert import AbstractHilbert, Qubit, Spin, Fock, SpinOrbitalFermions
from netket.operator.permutation import PermutationOperator, PermutationOperatorFermion

from netket.symmetry.group import Permutation


def construct_permutation_operator(
    hilbert_space: AbstractHilbert, permutation: Permutation
):
    """
    Return the appropriate permutation operator depending on the type of Hilbert space.

    If the Hilbert space is a spin or boson Hilbert space, a PermutationOperator will be returned,
    if it is a fermion Hilbert space, a PermutationOperatorFermion will be returned.

    For mathematical details on permutation operators, see :doc:`/advanced/symmetry`.

    Args:
        hilbert_space: The Hilbert space on which the permutation acts.
        permutation: The permutation to be represented as an operator.

    Returns:
        Either a PermutationOperator or PermutationOperatorFermion depending on the Hilbert space type.

    Raises:
        TypeError: If the Hilbert space type is not supported.
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
