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
"""
Netket input driver to create json input files.

"""

import json
import subprocess
from tkinter import TclError
import pynetket as nk
from pynetket.python_utils import message
from pynetket.python_utils import plot_observable
from pynetket.python_utils import encode_complex


class NetKetInput(object):
    """
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
    """

    def __init__(self, *args):
        """
        This object takes all of the parameters from the args and adds them to
        the _pars attribute of this object.
        """

        self._pars = {}
        self._complete = False

        for arg in args:
            self._pars[arg._name] = arg._pars

    def write_json_input(self, json_file='input.json'):
        with open(json_file, 'w') as outfile:
            json.dump(self._pars, outfile, default=encode_complex)

    def run(self, n=4, json_file='input.json', plot=False, exact=0):
        """
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
        """

        self.write_json_input(json_file=json_file)

        self._sts = subprocess.Popen(
            "mpirun -n %d netket %s" % (n, json_file), shell=True)

    def plot(self, observable, exact=None):
        """

        Arguments
        ---------

            observable : string
                The name of an observable written to the json output file.

            exact : float
                The exact answer to compare to. This is used to calculated error
                bars in plot_observable. Default is None.
        """

        # Enclosing the plot_observable function in a try statement so it
        # doesn't throw and error when the user closes the plot.
        try:
            plot_observable(
                self._pars["GroundState"]["OutputFile"],
                observable,
                exact=exact)
        except TclError:
            message("Warning", "Plot closed.")
            message("Warning", "NetKet will coninue to run.")


if __name__ == "__main__":
    g = nk.Graph("Hypercube", L=20, Dimension=1, Pbc=True)
    h = nk.Hamiltonian("Ising", h=1.0)
    m = nk.Machine("RbmSpin", Alpha=1.0)
    s = nk.Sampler("MetropolisLocal")
    o = nk.Optimizer("Sgd", LearningRate=0.1)
    gs = nk.GroundState("Sr", Niteropt=300, Diagshift=0.1, UseIterative=False)
    nk_input = nk.NetKetInput(g)
    # input.write_json_input()
    # input.run()
