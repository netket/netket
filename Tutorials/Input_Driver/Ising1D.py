#!/usr/bin/env python
'''
Demonstrates the use of the input driver to calculate ground state of simple
Ising1D model.
'''

import netket.input_driver as nk

g = nk.Graph("Hypercube", L=20, Dimension=1, Pbc=True)
h = nk.Hamiltonian("Ising", h=1.0)
m = nk.Machine("RbmSpin", Alpha=1.0)
s = nk.Sampler("MetropolisLocal")
o = nk.Optimizer("Sgd", LearningRate=0.1)
gs = nk.GroundState("Sr", Niteropt=300, Diagshift=0.1, UseIterative=False)
input = nk.NetKetInput(g, h, m, s, o, gs)

input.run(plot=True, exact=(-1.274549484318e+00 * 20))
