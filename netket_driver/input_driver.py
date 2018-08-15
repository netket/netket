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
Netket input driver to create json input files.

'''

import os
import json
import subprocess
from netket_driver.python_utils import Message
from netket_driver.python_utils import plot_output
from netket_driver.python_utils import encode_complex
from netket_driver.graph import Graph
from netket_driver.hamiltonian import Hamiltonian
from netket_driver.hilbert import Hilbert
from netket_driver.machine import Machine
from netket_driver.sampler import Sampler
from netket_driver.optimizer import Optimizer
from netket_driver.groundstate import GroundState


class NetKetInput(object):
    '''
    Main input driver for NetKet.

    NetKetInput objects take the other objects in netket_driver as inputs and
    can write the input json and run NetKet within python.

    Simple Usage::

        >>> g = Graph("Hypercube", L=20, Dimension=1, Pbc=True)
        >>> h = Hamiltonian("Ising", h=1.0)
        >>> m = Machine("RbmSpin", Alpha=1.0)
        >>> s = Sampler("MetropolisLocal")
        >>> o = Optimizer("Sgd", LearningRate=0.1)
        >>> gs = GroundState("Sr", Niteropt=300, Diagshift=0.1, UseIterative=False)
        >>> input = NetKetInput(g, h, m, s, o, gs)
    '''

    def __init__(self, *args):
        '''
        This object takes all of the parameters from the args and adds them to
        the _pars attribute of this object.
        '''

        self._pars = {}

        for arg in args:
            self._pars[arg._name] = arg._pars

    def write_json_input(self, json_file='input.json'):
        with open(json_file, 'w') as outfile:
            json.dump(self._pars, outfile, default=encode_complex)

    def run(self, n=4, json_file='input.json', plot=False, exact=0):
        '''
        Writes json input file and calls the netket executable.

        Arguments
        ---------

        n : int
            The number of processors to use when running netket. Default is 4.

        json_file : string
            The input json file name. Default is input.json.

        plot : bool
            Toggle for "live" plotting of output as it's generated. This allows
            users to the energy as a function of iteration. Default is False.

        exact : float
            The exact answer that the energy in the plot should be compared to.
            Default is 0.


        Simple Usage::

            >>> input = NetKetInput(g, h, m, s, o, gs)
            >>> input.run( n=8, json_file="my_input.json" )
        '''

        self.write_json_input(json_file=json_file)

        sts = subprocess.Popen(
            "mpirun -n %d netket %s" % (n, json_file), shell=True)

        if plot:
            try:
                plot_output(exact, self._pars["Learning"]["OutputFile"])
            except:
                Message("Warning", "Plot closed.")
                Message("Warning", "NetKet will coninue to run.")


if __name__ == "__main__":
    g = Graph("Hypercube", L=20, Dimension=1, Pbc=True)
    h = Hamiltonian("Ising", h=1.0)
    m = Machine("RbmSpin", Alpha=1.0)
    s = Sampler("MetropolisLocal")
    o = Optimizer("Sgd", LearningRate=0.1)
    gs = GroundState("Sr", Niteropt=300, Diagshift=0.1, UseIterative=False)
    input = NetKetInput(g)
    # input.write_json_input()
    # input.run()
