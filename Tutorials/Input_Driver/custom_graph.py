#!/usr/bin/env python
# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Demonstrates the use of the pynetket to calculate ground state of simple
Ising1D model.
'''

import pynetket as nk
import igraph as ig
import networkx as nx

g = nk.Graph("Custom", graph=nx.star_graph(10))
h = nk.Hamiltonian("Ising", h=1.0)
m = nk.Machine("RbmSpin", Alpha=1.0)
s = nk.Sampler("MetropolisLocal")
o = nk.Optimizer("Sgd", LearningRate=0.05)
gs = nk.GroundState(
    "Sr",
    Nsamples=1000,
    NiterOpt=1000,
    Diagshift=0.5,
    UseIterative=False,
    OutputFile="test")
calc = nk.NetKetInput(g, h, m, s, o, gs)

calc.run()
calc.plot("EnergyVariance", exact=None)
