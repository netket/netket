#!/usr/bin/env python
# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Base class for NetKet input driver Hamiltonian objects.

'''

from pynetket.python_utils import set_mand_pars
from pynetket.python_utils import set_opt_pars


class Hamiltonian(object):
    '''
    Driver for input hamiltonian parameters.

    Simple Usage::

        >>> ham = Hamiltonian("BoseHubbard", Nmax=3, U=4.0, Nbosons=12)
        >>> print(ham._pars)
        {'Name': 'BoseHubbard', 'Nmax': 3, 'U': 4.0, 'Nbosons': 12}
    '''

    _name = "Hamiltonian"

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Arguments
        ---------

        name: string
            BoseHubbard, Graph, Heisenberg, Ising, and Custom.


        kwargs
        ------

        Nmax : int
            Used in the BoseHubbard Hamiltonian. Maximum number of bosons per
            site. Default is 3.

        U : float
            Used in the BoseHubbard Hamiltonian. The Hubbard interaction
            strength. Default is 4.0.

        Nbosons : int
            Used in the BoseHubbard Hamiltonian. Number of bosons.

        V : float
            Used in the BoseHubbard Hamiltonian. Nearest neighbor interaction
            strength.

        Mu : float
            Used in the BoseHubbard Hamiltonian. Chemical potential.

        SiteOps : 3D list of floats
            Used in Graph Hamiltonians. List of site operators. A list of numpy
            arrays should be converted to a list of 2D lists.

        BondOps : 3D list of floats
            Used in Graph Hamiltonians. Collection of operators that act on
            certain colors of bonds, specified by BondOp. A list of numpy arrays
            should be converted to a list of 2D lists.

        BondOp : List of ints
            Used in Graph Hamiltonians. List of colors for each operator in
            BondOps.

        TotalSz : float
            Used in Heisenberg Hamiltonian. Restricts sampling of spins.

        Operators : 3D list of floats
            Used in Custom Hamiltonians. List of operators. A list of numpy
            arrays should be converted to a list of 2D lists.

        ActingOn : 2D list of ints
            Used in Custom Hamiltonians. List of edges that Operators act on.


        '''

        self._pars = {}

        if name == "BoseHubbard":
            self._pars['Name'] = name

            set_mand_pars(self._pars, "Nmax", kwargs, 3)  # TODO choose def
            set_mand_pars(self._pars, "U", kwargs, 4.0)
            set_opt_pars(self._pars, "Nbosons", kwargs)  # TODO
            set_opt_pars(self._pars, "V", kwargs)
            set_opt_pars(self._pars, "Mu", kwargs)

        elif name == "Graph":
            self._pars['Name'] = name

            set_mand_pars(self._pars, "SiteOps", kwargs, [])
            set_mand_pars(self._pars, "BondOps", kwargs, [])
            set_mand_pars(self._pars, "BondOp", kwargs, [])

        elif name == "Heisenberg":
            self._pars['Name'] = name

            set_opt_pars(self._pars, "TotalSz", kwargs)  # TODO choose def

        elif name == "Ising":
            self._pars['Name'] = name

            set_mand_pars(self._pars, "h", kwargs, 1.0)  # TODO choose def
            set_opt_pars(self._pars, "J", kwargs)  # TODO choose def

        elif name == "Custom":

            set_mand_pars(self._pars, "Operators", kwargs, [])
            set_mand_pars(self._pars, "ActingOn", kwargs, [])

        else:
            raise ValueError("%s Hamiltonian not supported" % name)


if __name__ == '__main__':
    ham = Hamiltonian("BoseHubbard", Nmax=3, U=4.0, Nbosons=12)
    print(ham._pars)
