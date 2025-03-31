# Copyright 2023 The NetKet Authors - All rights reserved.
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

import numpy as np
from netket.operator import AbstractOperator
from netket.hilbert import AbstractHilbert

def to_quspin_format(operator: AbstractOperator):
    """
    Convert a NetKet operator to QuSpin format.

    Args:
        operator: The NetKet operator to convert.

    Returns:
        A QuSpin operator.
    """
    from quspin.operators import hamiltonian

    hilbert = operator.hilbert
    size = hilbert.size

    static_list = []
    dynamic_list = []

    for term in operator.terms:
        coeff = term.coeff
        ops = term.ops
        sites = term.sites

        if isinstance(coeff, complex):
            dynamic_list.append([ops, sites, lambda t: np.exp(-1j * coeff * t)])
        else:
            static_list.append([ops, sites, coeff])

    return hamiltonian(static_list, dynamic_list, basis=hilbert)

def target_symmetry_subsector(operator: AbstractOperator, subsector: str):
    """
    Target a specific symmetry subsector for a NetKet operator.

    Args:
        operator: The NetKet operator.
        subsector: The symmetry subsector to target.

    Returns:
        A new NetKet operator targeting the specified symmetry subsector.
    """
    # This is a placeholder implementation. The actual implementation will depend on the specific
    # symmetry subsector and how it is represented in NetKet and QuSpin.
    raise NotImplementedError("Targeting specific symmetry subsectors is not yet implemented.")
