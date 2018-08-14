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
Base class for NetKet input driver Optimizer objects.

'''

from netket_driver.python_utils import set_mand_pars
from netket_driver.python_utils import set_opt_pars


class Optimizer(object):
    '''
    Driver for input optimizer parameters.

    Simple Usage::

        >>> opt = Optimizer("Sgd")
        >>> print(opt._pars)
        {'Name': 'Sgd', 'LearningRate': 0.1}
    '''

    _name = "Optimizer"

    # optmizers = ["Sgd", "AdaMax", "AdaDelta", "Momentum", "AMSGrad", "RMSProp"]

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Arguments
        ---------

        name : str
            Optimizer name. Currently supported: Stochastic gradient descent
            ("Sgd"), "AdaMax", "AdaDelta", "Momentum", "AMSGrad", and "RMSProp".


        kwargs
        ------

        LearningRate : float

        L2Reg :

        DecayFactor : float

        Alpha :

        Beta1 :

        Beta2 :



        '''

        self._pars = {}

        if name == "Sgd":
            self._pars["Name"] = name
            set_mand_pars(self._pars, "LearningRate", kwargs, 0.1)  # TODO
            set_opt_pars(self._pars, "L2Reg", kwargs)
            set_opt_pars(self._pars, "DecayFactor", kwargs)

        elif name == "AdaMax":
            self._pars["Name"] = name
            set_opt_pars(self._pars, "Alpha", kwargs)
            set_opt_pars(self._pars, "Beta1", kwargs)
            set_opt_pars(self._pars, "Beta2", kwargs)
            set_opt_pars(self._pars, "Epscut", kwargs)

        elif name == "AdaDelta":
            self._pars["Name"] = name
            set_opt_pars(self._pars, "Rho", kwargs)
            set_opt_pars(self._pars, "Epscut", kwargs)

        elif name == "Momentum":
            self._pars["Name"] = name
            set_opt_pars(self._pars, "LearningRate", kwargs)
            set_opt_pars(self._pars, "Beta", kwargs)

        elif name == "AMSGrad":
            self._pars["Name"] = name
            set_opt_pars(self._pars, "LearningRate", kwargs)
            set_opt_pars(self._pars, "Beta1", kwargs)
            set_opt_pars(self._pars, "Beta2", kwargs)
            set_opt_pars(self._pars, "Epscut", kwargs)

        elif name == "RMSProp":
            self._pars["Name"] = name
            set_opt_pars(self._pars, "LearningRate", kwargs)
            set_opt_pars(self._pars, "Beta", kwargs)
            set_opt_pars(self._pars, "Epscut", kwargs)

        else:
            raise ValueError("%s Optimizer not supported" % name)


if __name__ == '__main__':
    opt = Optimizer("Sgd")
    print(opt._pars)
