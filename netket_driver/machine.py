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
Base class for NetKet input driver machine objects.

'''

from netket_driver.python_utils import set_mand_pars
from netket_driver.python_utils import set_opt_pars


class Machine(object):
    '''
    Driver for input machine parameters.

    Simple Usage::

    >>> mach = Machine("RbmSpin")
    >>> print(mach._pars)
    {'Name': 'RbmSpin', 'Alpha': 1.0}
    '''

    _name = "Machine"

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Arguments
        ---------

        name : str
            RBM or RBMSpinSymm.


        kwargs
        ------

        Alpha : float
            Used with all machines. Alternative to Nhidden, here it is the M/N
            where M is the number of hidden units and N is the total number of
            units.

        InitFile : str
            Used with all machines. If specified, network parameters are loaded
            from the given file.

        InitRandom : bool
            Used with all machines. Whether to initialize the parameters with
            random gaussian-distributed values.

        Nhidden : int
            Used with all machines. The number of hidden units M.

        SigmaRand : float
            Used with all machines. If InitRandom is chosen, this is the
            standard deviation of the gaussian.

        UseHiddenBias : bool
            Used with all machines. Whether to use the hidden bias bj.

        UseVisibleBias : bool
            Used with all machines. Whether to use the visible bias ai.

        '''

        self._pars = {}

        if name == "RbmSpin" or name == "RbmSpinSymm" or name == "RbmMultival":
            self._pars["Name"] = name

            set_mand_pars(self._pars, "Alpha", kwargs, 1.0)  # TODO
            set_opt_pars(self._pars, "InitFile", kwargs)
            set_opt_pars(self._pars, "InitRandom", kwargs)
            set_opt_pars(self._pars, "Nhidden", kwargs)
            set_opt_pars(self._pars, "SigmaRand", kwargs)
            set_opt_pars(self._pars, "UseHiddenBias", kwargs)
            set_opt_pars(self._pars, "UseVisibleBias", kwargs)

        else:
            raise ValueError("%s Machine not supported" % name)


if __name__ == '__main__':
    mach = Machine("RbmSpin")
    print(mach._pars)
