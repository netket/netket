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
Base class for NetKet input driver Sampler objects.

'''

from pynetket.python_utils import set_mand_pars
from pynetket.python_utils import set_opt_pars


class Sampler(object):
    '''
    Driver for input sampler parameters.

    Simple Usage::

        >>> samp = Sampler("MetropolisLocal")
        >>> print(samp._pars)
        {'Name': 'MetropolisLocal'}
    '''

    _name = "Sampler"

    _samplers = [
        "MetropolisLocal", "MetropolisLocalPt", "MetropolisExchange",
        "MetropolisExchangePt", "MetropolisHamiltonian",
        "MetropolisHamiltonianPt", "MetropolisHop"
    ]

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Args.
        -----




        kwargs.
        -------



        '''

        self._pars = {}

        if name in self._samplers:
            self._pars["Name"] = name

        elif name == "Custom":
            set_opt_pars(self._pars, "ActingOn", kwargs)  # TODO
            set_opt_pars(self._pars, "MoveOperators", kwargs)  # TODO

        else:
            raise ValueError("%s Sampler not supported" % name)


if __name__ == '__main__':
    samp = Sampler("MetropolisLocal")
    print(samp._pars)
