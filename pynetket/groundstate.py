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
Base class for NetKet input driver Learning objects.

'''

from pynetket.python_utils import set_mand_pars
from pynetket.python_utils import set_opt_pars


class GroundState(object):
    '''
    Driver for input Ground State parameters.

    Simple Usage::

        >>> gs = GroundState("Gd")
        >>> print(gs._pars)
        {'Method': 'Gd', 'Nsamples': 1000, 'NiterOpt': 1000, 'OutputFile': 'test'}
    '''

    _name = "GroundState"

    def __init__(self, method, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Arguments
        ---------

        method : string
            Name of method. Currently supported methods are gradient descent
            ("Gd"), stochastic reconfiguration ("Sr"), and exact diagonalization
            ("Ed").



        kwargs
        ------

        Nsamples : int

        NiterOpt : int

        DiscardedSamplesOnInit : float

        DiscaredSamples : float

        OutputFile : str

        SaveEvery : int

        Diagshift : float

        RescaleShift : float

        UseIterative : bool


        '''

        self._pars = {}

        if method in ["Gd", "Sr"]:
            self._pars["Method"] = method
            set_mand_pars(self._pars, "Nsamples", kwargs, 1000)  # TODO
            set_mand_pars(self._pars, "NiterOpt", kwargs, 1000)  # TODO
            set_opt_pars(self._pars, "DiscardedSamplesOnInit", kwargs)
            set_opt_pars(self._pars, "DiscardedSamples", kwargs)
            set_mand_pars(self._pars, "OutputFile", kwargs, "test")
            set_opt_pars(self._pars, "SaveEvery", kwargs)

            set_opt_pars(self._pars, "Diagshift", kwargs)
            set_opt_pars(self._pars, "RescaleShift", kwargs)
            set_opt_pars(self._pars, "UseIterative", kwargs)

        elif method == "Ed":
            self._pars["Method"] = method

            set_mand_pars(self._pars, "OutputFile", kwargs, "test")

        else:
            raise ValueError("%s Learning not supported" % name)


if __name__ == '__main__':
    learn = Learning("Gd")
    print(learn._pars)
