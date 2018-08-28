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

from pynetket.python_utils import set_opt_pars


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


        Universal kwargs
        ----------------

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

        SigmaRand : float
            Used with all machines. If InitRandom is chosen, this is the
            standard deviation of the gaussian.


        RBM kwargs
        ----------

        Nhidden : int
            Used with RBM machines. The number of hidden units M.

        UseHiddenBias : bool
            Used with RBM machines. Whether to use the hidden bias bj.

        UseVisibleBias : bool
            Used with RBM machines. Whether to use the visible bias ai.


        Feed-Forward Neural Network kwargs
        ----------------------------------

        layers : list of dict
            List of dictionaries containing information about the each layer.
            Currently, three types of layers are supporjted: FullyConnected,
            Convolutional, and Sum.


        '''

        self._pars = {}

        if name == "RbmSpin" or name == "RbmSpinSymm" or name == "RbmMultival":
            self._pars["Name"] = name

            set_opt_pars(self._pars, "Alpha", kwargs)
            set_opt_pars(self._pars, "InitFile", kwargs)
            set_opt_pars(self._pars, "InitRandom", kwargs)
            set_opt_pars(self._pars, "Nhidden", kwargs)
            set_opt_pars(self._pars, "SigmaRand", kwargs)
            set_opt_pars(self._pars, "UseHiddenBias", kwargs)
            set_opt_pars(self._pars, "UseVisibleBias", kwargs)

        elif name == "FFNN":
            self._pars["Name"] = name
            self._pars["Layers"] = []

            # If there are layers passed by kwargs, add them via add_layer
            if "Layers" in kwargs:
                for layer in kwargs["Layers"]:
                    self.add_layer(layer)

        elif name == "Jastrow":
            self._pars["Name"] = name
            set_opt_pars(self._pars, "Nvisible", kwargs)
            set_opt_pars(self._pars, "W", kwargs)

        elif name == "JastrowSymm":
            self._pars["Name"] = name
            set_opt_pars(self._pars, "Nvisible", kwargs)
            set_opt_pars(self._pars, "Wsymm", kwargs)

        else:
            raise ValueError("%s Machine not supported" % name)

    def add_layer(self, raw_layer):
        '''
        Adds layer (in the form of a dictionary) to the list of layers stored
        in _pars.
        '''
        layer = {}
        layer_types = ["FullyConnected", "Convolutional", "Sum"]

        # Test and make sure the layer has a name
        try:
            if raw_layer["Name"] in layer_types:
                layer["Name"] = raw_layer["Name"]
        except KeyError:
            raise KeyError("Name of layer not found.")

        # Add relevant feature based on the type of layer
        if layer["Name"] == "FullyConnected":
            set_opt_pars(layer, "Inputs", raw_layer)
            set_opt_pars(layer, "Outputs", raw_layer)
            set_opt_pars(layer, "Activation", raw_layer)
            set_opt_pars(layer, "Bias", raw_layer)

        elif layer["Name"] == "Convolutional":
            set_opt_pars(layer, "InputChannels", raw_layer)
            set_opt_pars(layer, "OutputChannels", raw_layer)
            set_opt_pars(layer, "Distance", raw_layer)
            set_opt_pars(layer, "Activation", raw_layer)
            set_opt_pars(layer, "Bias", raw_layer)

        elif layer["Name"] == "Sum":
            set_mand_pars(layer, "Inputs", raw_layer)

        else:
            raise ValueError("%s Layer type not supported" % name)

        self._pars["Layers"].append(layer)


if __name__ == '__main__':
    mach = Machine("RbmSpin")
    print(mach._pars)

    layers = [{
        'Name': 'FullyConnected',
        'Inputs': 20,
        'Outputs': 20,
        'Activation': 'Lncosh'
    }, {
        'Name': 'FullyConnected',
        'Inputs': 20,
        'Outputs': 10,
        'Activation': 'Lncosh'
    }]

    mach = Machine("FFNN", Layers=layers)
    print(mach._pars)

    mach2 = Machine("FFNN")
    mach2.add_layer(layers[0])
    mach2.add_layer(layers[1])
    print(mach2._pars)
    print(mach2._pars == mach._pars)

    m = Machine("FFNN")
    for i in range(4):
        m.add_layer({"Name": "FullyConnected"})

    print(m._pars)
