# Copyright 2022 The NetKet Authors - All rights reserved.
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

from typing import Optional, TYPE_CHECKING


from netket.operator._pauli_strings.base import _count_of_locations
from netket.hilbert.abstract_hilbert import AbstractHilbert

from netket.experimental.hilbert import SpinOrbitalFermions

from ._fermion_operator_2nd_utils import (
    _convert_terms_to_spin_blocks,
    _collect_constants,
)

if TYPE_CHECKING:
    from ._fermion_operator_2nd import FermionOperator2ndBase


def from_openfermion(
    hilbert: AbstractHilbert,
    of_fermion_operator=None,  # : "openfermion.ops.FermionOperator" type
    *,
    n_orbitals: Optional[int] = None,
    convert_spin_blocks: bool = False,
) -> "FermionOperator2ndBase":
    r"""
    Converts an openfermion FermionOperator into a netket FermionOperator2nd.

    The hilbert first argument can be dropped, see __init__ for details and default
    value.
    Warning: convention of openfermion.hamiltonians is different from ours: instead
    of strong spin components as subsequent hilbert state outputs (i.e. the 1/2 spin
    components of spin-orbit i are stored in locations (2*i, 2*i+1)), we concatenate
    blocks of definite spin (i.e. locations (i, n_orbitals+i)).

    Args:
        hilbert: (optional) hilbert of the resulting FermionOperator2nd object
        of_fermion_operator (openfermion.ops.FermionOperator):
            `FermionOperator object <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`_ .
            More information about those objects can be found in
            `OpenFermion's documentation <https://quantumai.google/reference/python/openfermion>`_
        n_orbitals: (optional) total number of orbitals in the system, default
            None means inferring it from the FermionOperator2nd. Argument is
            ignored when hilbert is given.
        convert_spin_blocks: whether or not we need to convert the FermionOperator
            to our convention. Only works if hilbert is provided and if it has
            spin != 0

    """
    from openfermion.ops import FermionOperator

    if hilbert is None:
        raise ValueError(
            "The first argument `from_openfermion` must either be an "
            "openfermion operator or an Hilbert space, followed by "
            "an openfermion operator"
        )

    if not isinstance(hilbert, AbstractHilbert):
        # if first argument is not Hilbert, then shift all arguments by one
        hilbert, of_fermion_operator = None, hilbert

    if not isinstance(of_fermion_operator, FermionOperator):
        raise NotImplementedError()

    if convert_spin_blocks and hilbert is None:
        raise ValueError("if convert_spin_blocks, the hilbert must be specified")

    terms = list(of_fermion_operator.terms.keys())
    weights = list(of_fermion_operator.terms.values())
    terms, weights, constant = _collect_constants(terms, weights)

    if hilbert is not None:
        # no warning, just overwrite
        n_orbitals = hilbert.n_orbitals

        if convert_spin_blocks:
            if not hasattr(hilbert, "spin") or hilbert.spin is None:
                raise ValueError(
                    f"cannot convert spin blocks for hilbert space {hilbert} without spin"
                )
            n_spin = hilbert._n_spin_states
            terms = _convert_terms_to_spin_blocks(terms, n_orbitals, n_spin)
    if n_orbitals is None:
        # we always start counting from 0, so we only determine the maximum location
        n_orbitals = _count_of_locations(of_fermion_operator)
    if hilbert is None:
        hilbert = SpinOrbitalFermions(n_orbitals)  # no spin splitup assumed

    return hilbert, terms, weights, constant
