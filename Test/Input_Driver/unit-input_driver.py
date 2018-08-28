#!/usr/bin/env python
'''
 Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import unittest
import os
import json
import numpy as np
import pynetket as nk
from pynetket.python_utils import encode_complex

try:
    import networkx as nx
    import_nx = True
except ImportError:
    print("No networkx found.")
    import_nx = False


class KnownOutput(unittest.TestCase):
    def test1_Graph(self):
        pars = {}
        pars['Graph'] = {
            'Name': 'Hypercube',
            'L': 20,
            'Dimension': 1,
            'Pbc': True,
        }

        # Test properly created hypercube
        graph = nk.Graph("Hypercube", L=20, Dimension=1, Pbc=True)
        self.assertEqual(pars['Graph'] == graph._pars)

        # Test networkx custom graph
        if import_nx:
            pars = {}
            G = nx.star_graph(10)

            pars['Graph'] = {
                'Edges': list(G.edges),
            }

            graph = nk.Graph("Custom", graph=G)
            self.assertEqual(pars['Graph'] == graph._pars)

    def test2_Hamiltonian(self):
        # BoseHubbard
        pars = {}
        pars['Hamiltonian'] = {
            'Name': 'BoseHubbard',
            'U': 4.0,
            'Nmax': 3,
            'Nbosons': 12,
        }

        ham = nk.Hamiltonian("BoseHubbard", Nmax=3, U=4.0, Nbosons=12)
        self.assertEqual(pars['Hamiltonian'] == ham._pars)

        # Test writing custom Hamiltonians with complex numbers
        sigmay = np.array([[0, -1j], [1j, 0]]).tolist()
        ham = nk.Hamiltonian("Custom", Operators=[sigmay], ActingOn=[])
        with open("test.json", 'w') as outfile:
            json.dump(ham._pars, outfile, default=encode_complex)

        os.remove("test.json")

    def test9_NetKetInput(self):
        pars = {}

        # defining the lattice
        pars['Graph'] = {
            'Name': 'Hypercube',
            'L': 20,
            'Dimension': 1,
            'Pbc': True,
        }

        # defining the hamiltonian
        pars['Hamiltonian'] = {
            'Name': 'Ising',
            'h': 1.0,
        }

        # defining the wave function
        pars['Machine'] = {
            'Name': 'RbmSpin',
            'Alpha': 1.0,
        }

        # defining the sampler
        # here we use Metropolis sampling with single spin flips
        pars['Sampler'] = {
            'Name': 'MetropolisLocal',
        }

        # defining the Optimizer
        # here we use the Stochastic Gradient Descent
        pars['Optimizer'] = {
            'Name': 'Sgd',
            'LearningRate': 0.1,
        }

        # defining the GroundState method
        # here we use the Stochastic Reconfiguration Method
        pars['GroundState'] = {
            'Method': 'Sr',
            'Nsamples': 1000,
            'NiterOpt': 300,
            'Diagshift': 0.1,
            'UseIterative': False,
            'OutputFile': "test",
        }

        g = nk.Graph("Hypercube", L=20, Dimension=1, Pbc=True)
        h = nk.Hamiltonian("Ising", h=1.0)
        m = nk.Machine("RbmSpin", Alpha=1.0)
        s = nk.Sampler("MetropolisLocal")
        o = nk.Optimizer("Sgd", LearningRate=0.1)
        gs = nk.GroundState(
            "Sr",
            Nsamples=1000,
            NiterOpt=300,
            Diagshift=0.1,
            UseIterative=False,
            OutputFile="test")

        input = nk.NetKetInput(g, h, m, s, o, gs)
        input.write_json_input()

        self.assertEqual(input._pars == pars)
        os.remove("input.json")

    def test10_NetKetInput2(self):
        sigmax = [[0, 1], [1, 0]]
        sigmaz = [[1, 0], [0, -1]]
        mszsz = (np.kron(sigmaz, sigmaz)).tolist()
        operators = []
        sites = []
        L = 20
        for i in range(L):
            operators.append(sigmax)
            sites.append([i])
            operators.append(mszsz)
            sites.append([i, (i + 1) % L])

            pars = {}

            # first we choose a hilbert space for our custom hamiltonian
            pars['Hilbert'] = {
                'QuantumNumbers': [1, -1],
                'Size': L,
            }

            # defining a custom hamiltonian
            pars['Hamiltonian'] = {
                'Operators': operators,
                'ActingOn': sites,
            }

            # defining the wave function
            pars['Machine'] = {
                'Name': 'RbmSpin',
                'Alpha': 1,
            }

            # defining the sampler
            # here we use Metropolis sampling with single spin flips
            pars['Sampler'] = {
                'Name': 'MetropolisLocal',
            }

            # defining the Optimizer
            # here we use AdaMax
            pars['Optimizer'] = {
                'Name': 'AdaMax',
            }

            # defining the GroundState method
            # here we use a Gradient Descent with AdaMax
            pars['GroundState'] = {
                'Method': 'Gd',
                'Nsamples': 1.0e3,
                'NiterOpt': 40000,
                'OutputFile': "test",
            }

        hil = nk.Hilbert("Custom", QuantumNumbers=[1, -1], Size=L)
        h = nk.Hamiltonian("Custom", Operators=operators, ActingOn=sites)
        m = nk.Machine("RbmSpin", Alpha=1)
        s = nk.Sampler("MetropolisLocal")
        o = nk.Optimizer("AdaMax")
        gs = nk.GroundState(
            "Gd", Nsamples=1e3, NiterOpt=40000, OutputFile="test")
        input = nk.NetKetInput(hil, h, m, s, o, gs)
        input.write_json_input()

        self.assertEqual(input._pars == pars)
        os.remove("input.json")


if __name__ == "__main__":
    print("Testing Python Input Driver")
    unittest.main()
