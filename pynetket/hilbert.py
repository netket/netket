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
Base class for NetKet input driver Hilbert objects.

'''

from pynetket.python_utils import set_opt_pars


class Hilbert(object):
    '''
    Driver for input hilbert space parameters.

    Simple Usage::

        >>> hil = Hilbert("Spin", NSpins=10, TotalSz=0, S=0)
        >>> print(hil._pars)
        {'Name': 'Spin', 'Nspins': 10, 'S': 0.0, 'TotalSz': 0}
    '''

    _name = "Hilbert"

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Arguments
        ---------

        name : string
            Boson, Spin, Qubit, and Custom.


        kwargs
        ------

        Nsites : int
            Used with the Boson Hilbert space. Number of sites. The default is
            10.

        Nmax : int
            Used with the Boson Hilbert space. Maximum number of bosons per
            site. Default is 3.

        Nbosons : int
            Used with the Boson Hilbert space. Number of bosons.

        Nspins : int
            Used with the Spin Hilbert space. Number of sites. The default is
            10.

        S : float
            Used with the Spin Hilbert space. Total spin. The default is 0.0.

        TotalSz : float
            Used with the Spin Hilbert space. Restricts sampling of spins.

        Nqubits : int
            Used with the Qubit Hilbert space. The number of qubits. The default
            is 10.

        QuantumNumbers : list of floats
            Used with the Custom Hilbert space. A list of the quantum numbers.
            The default is [-1, 1].

        Size : int
            Used with the Custom Hilbert space. The size of the Hilbert space.
            The default is 10
        '''

        self._pars = {}

        if name == "Boson":
            self._pars['Name'] = name
            set_opt_pars(self._pars, "Nsites", kwargs)
            set_opt_pars(self._pars, "Nmax", kwargs)
            set_opt_pars(self._pars, "Nbosons", kwargs)

        elif name == "Spin":
            self._pars['Name'] = name
            set_opt_pars(self._pars, "Nspins", kwargs)
            set_opt_pars(self._pars, "S", kwargs)
            set_opt_pars(self._pars, "TotalSz", kwargs)

        elif name == "Qubit":
            self._pars['Name'] = name
            set_opt_pars(self._pars, "Nqubits", kwargs)

        elif name == "Custom":
            set_opt_pars(self._pars, "QuantumNumbers", kwargs)
            set_opt_pars(self._pars, "Size", kwargs)

        else:
            raise ValueError("%s Hilbert space not supported" % name)


if __name__ == '__main__':
    hil = Hilbert("Spin", NSpins=10, TotalSz=0, S=0)
    print(hil._pars)
