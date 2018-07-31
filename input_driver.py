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

import json
import subprocess
from netket.python_utils import Message
from netket.python_utils import plot_output

try:
    import networkx as nx
    import_nx = True
except ImportError:
    # Message("Warning", "No networkx found.")
    import_nx = False


def set_mand_pars(params, key, kwargs, def_value):
    try:
        params[key] = kwargs[key]
    except KeyError:
        # Message("Info", "Couldn't find kwargs with name %s." % key)
        # Message("Info",
        #         "Setting %s to default value of %s" % (key, str(def_value)))
        params[key] = def_value


def set_opt_pars(params, key, kwargs):
    try:
        params[key] = kwargs[key]
    except KeyError:
        pass


class Graph(object):
    '''
    Driver for input graph parameters.

    Simple Usage::

    TODO
    '''

    _name = "Graph"

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Args.
        -----

        name (sting): Hypercube or Custom.


        kwargs.
        -------

        L (int): Used with Hypercube graphs. Sets the length per dimension.
            Default value is 10.

        Dimension (int): Used with Hypercube graphs. Sets the number of
            dimensions. Default value is 1.

        Pbc (bool): Used with Hypercube graphs. Determines periodicity. Default
            value is True.

        graph (nx.graph or igraph.graph): Used with Custom graphs. Graph object
            that is used to populate the edges and edge colors (if applicable)
            values in our input parameters. The graph must have edges.
        '''

        self._pars = {}

        # Hypercube option
        if name == "Hypercube":

            self._pars['Name'] = name

            set_mand_pars(self._pars, "L", kwargs, 10)
            set_mand_pars(self._pars, "Dimension", kwargs, 1)
            set_mand_pars(self._pars, "Pbc", kwargs, True)

        elif name == "Custom":
            if "graph" in kwargs:
                if import_nx:
                    if type(kwargs["graph"]) == type(nx.Graph()):
                        # Grab edges
                        print("Found a networkx graph")
                        assert (len(kwargs["graph"].edges) > 0)
                        self._pars['Edges'] = list(kwargs["graph"].edges)

                        # Grab edge colors
                        try:
                            self._pars['EdgeColors'] = [[
                                u, v, kwargs["graph"][u][v]['color']
                            ] for u, v in kwargs["graph"].edges]

                        except KeyError:
                            print("No edge colors found.")

                # TODO add import igraph

        else:
            raise ValueError("%s graph not supported" % name)


class Hamiltonian(object):
    '''
    Driver for input hamiltonian parameters.

    Simple Usage::

    TODO
    '''

    _name = "Hamiltonian"

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Args.
        -----

        name (string): BoseHubbard, Graph, Heisenberg, Ising, and Custom.


        kwargs.
        -------

        Nmax (int): Used in the BoseHubbard Hamiltonian. Maximum number of
            bosons per site. Default is 3.

        U (float): Used in the BoseHubbard Hamiltonian. The Hubbard interaction
            strength. Default is 4.0.

        Nbosons (int): Used in the BoseHubbard Hamiltonian. Number of bosons.

        V (float): Used in the BoseHubbard Hamiltonian. Nearest neighbor
            interaction strength.

        Mu (float): Used in the BoseHubbard Hamiltonian. Chemical potential.

        SiteOps ([[[float]]]): Used in Graph Hamiltonians. List of site
            operators. A list of numpy arrays should be converted to a list of
            2D lists.

        BondOps ([[[float]]]): Used in Graph Hamiltonians. Collection of
            operators that act on certain colors of bonds, specified by BondOp.
            A list of numpy arrays should be converted to a list of 2D lists.

        BondOp ([int]): Used in Graph Hamiltonians. List of colors for each
            operator in BondOps.

        TotalSz (float): Used in Heisenberg Hamiltonian. Restricts sampling of
            spins.

        Operators ([[[float]]]): Used in Custom Hamiltonians. List of operators.
            A list of numpy arrays should be converted to a list of 2D lists.

        ActingOn: ([[int]]): Used in Custom Hamiltonians. List of edges
            that Operators act on.


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


class Hilbert(object):
    '''
    Driver for input hilbert space parameters.

    Simple Usage::

    TODO
    '''

    _name = "Hilbert"

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Args.
        -----

        name (string): Boson, Spin, Qubit, and Custom.


        kwargs.
        -------

        Nsites (int): Used with the Boson Hilbert space. Number of sites. The
            default is 10.

        Nmax (int): Used with the Boson Hilbert space. Maximum number of
            bosons per site. Default is 3.

        Nbosons (int): Used with the Boson Hilbert space. Number of bosons.

        Nspins (int): Used with the Spin Hilbert space. Number of sites. The
            default is 10.

        S (float): Used with the Spin Hilbert space. Total spin. The default is
            0.0.

        TotalSz (float): Used with the Spin Hilbert space. Restricts sampling of
            spins.

        Nqubits (int): Used with the Qubit Hilbert space. The number of
            qubits. The default is 10.

        QuantumNumbers ([float]): Used with the Custom Hilbert space. A list of
            the quantum numbers. The default is [-1, 1].

        Size (int): Used with the Custom Hilbert space. The size of the Hilbert
            space. The default is 10
        '''

        self._pars = {}

        if name == "Boson":
            self._pars['Name'] = name
            set_mand_pars(self._pars, "Nsites", kwargs, 10)  # TODO
            set_mand_pars(self._pars, "Nmax", kwargs, 3)  # TODO
            set_opt_pars(self._pars, "Nbosons", kwargs)

        elif name == "Spin":
            self._pars['Name'] = name
            set_mand_pars(self._pars, "Nspins", kwargs, 10)  # TODO
            set_mand_pars(self._pars, "S", kwargs, 0.0)  # TODO
            set_opt_pars(self._pars, "TotalSz", kwargs)

        elif name == "Qubit":
            self._pars['Name'] = name
            set_mand_pars(self._pars, "Nqubits", kwargs, 10)  # TODO

        elif name == "Custom":
            set_mand_pars(self._pars, "QuantumNumbers", kwargs,
                          [-1, 1])  # TODO
            set_mand_pars(self._pars, "Size", kwargs, 10)  # TODO

        else:
            raise ValueError("%s Hilbert space not supported" % name)


class Machine(object):
    '''
    Driver for input machine parameters.

    Simple Usage::

    TODO
    '''

    _name = "Machine"

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Args.
        -----

        name (sting): RBM or RBMSpinSymm.


        kwargs.
        -------

        Alpha (float): Used with all machines. Alternative to Nhidden, here it
            is the M/N where M is the number of hidden units and N is the total
            number of units.

        InitFile (string): Used with all machines. If specified, network
            parameters are loaded from the given file.

        InitRandom (bool): Used with all machines. Whether to initialize the parameters with random
            gaussian-distributed values.

        Nhidden (int): Used with all machines. The number of hidden units M.

        SigmaRand (float): Used with all machines. If InitRandom is chosen, this is the standard deviation of the gaussian.

        UseHiddenBias (bool): Used with all machines. Whether to use the hidden bias bj.

        UseVisibleBias (bool): Used with all machines. Whether to use the visible bias ai.

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


class Sampler(object):
    '''
    Driver for input sampler parameters.

    Simple Usage::

    TODO
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


class Optimizer(object):
    '''
    Driver for input optimizer parameters.

    Simple Usage::

    TODO
    '''

    _name = "Optimizer"

    # optmizers = ["Sgd", "AdaMax", "AdaDelta", "Momentum", "AMSGrad", "RMSProp"]

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Args.
        -----




        kwargs.
        -------



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


class Learning(object):
    '''
    Driver for input Learning parameters.

    Simple Usage::

    TODO
    '''

    _name = "Learning"

    def __init__(self, method, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Args.
        -----




        kwargs.
        -------



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

        elif name == "Ed":
            self._pars["Method"] = method

            set_mand_pars(self._pars, "OutputFile", kwargs, "test")

        else:
            raise ValueError("%s Learning not supported" % name)


class NetKetInput(object):
    '''
    '''

    def __init__(self, *args):
        '''
        '''

        self._pars = {}

        for arg in args:
            self._pars[arg._name] = arg._pars

        # print(self._pars) # TODO

    def write_json_input(self, json_file='input.json'):
        with open(json_file, 'w') as outfile:
            json.dump(self._pars, outfile)

    def run(self, n=4, json_file='input.json', plot=False, exact=0):
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
    l = Learning("Sr", Niteropt=300, Diagshift=0.1, UseIterative=False)
    input = NetKetInput(g)
    # input.write_json_input()
    # input.run()
