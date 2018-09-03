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

import pynetket as nk

sigmaxop = []
sites = []
L = 20
for i in range(L):
    # \sum_i sigma^x(i)
    sigmaxop.append([[0, 1], [1, 0]])
    sites.append([i])

g = nk.Graph("Hypercube", L=L, Dimension=1, Pbc=True)
h = nk.Hamiltonian("Ising", h=1.0)
m = nk.Machine("RbmSpin", Alpha=1.0)
s = nk.Sampler("MetropolisLocal")
obs = nk.Observable("SigmaX", ActingOn=sites, Operators=sigmaxop)
o = nk.Optimizer("Sgd", LearningRate=0.1)
gs = nk.GroundState(
    "Sr",
    Nsamples=1e3,
    NiterOpt=500,
    Diagshift=0.1,
    UseIterative=False,
    OutputFile="test")
calc = nk.NetKetInput(g, h, m, s, o, obs, gs)

calc.run()
calc.plot("SigmaX", exact=0.637275 * 20)
