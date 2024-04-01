# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk
import numpy as np

import netket.experimental as nkx

# 1D chain
L = 10

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, use_visible_bias=True, param_dtype=complex)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisHamiltonian(hi, ha, n_chains=16)

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1024, n_discard_per_chain=16)

# Optimizer
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=1e-4)

# Variational monte carlo driver
gs = nk.VMC(ha, op, variational_state=vs)

# Create observable
Sx = sum([nk.operator.spin.sigmax(hi, i) for i in range(L)])

# Run the optimization for 300 iterations to determine the ground state, used as
# initial state of the time-evolution
gs.run(n_iter=300, out="example_ising1d_GS", obs={"Sx": Sx})

# Create integrator for time propagation
integrator = nkx.dynamics.RK23(dt=0.01, adaptive=True, rtol=1e-3, atol=1e-3)
print(integrator)

# Quenched hamiltonian: this has a different transverse field than `ha`
ha1 = nk.operator.Ising(hilbert=hi, graph=g, h=0.5)
te = nkx.TDVP(
    ha1,
    variational_state=vs,
    integrator=integrator,
    t0=0.0,
    qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True, diag_shift=1e-4),
    error_norm="qgt",
)

log = nk.logging.JsonLog("example_ising1d_TE")

# perform the time-evolution saving the observable Sx at every `tstop` time
te.run(
    T=1.0,
    out=log,
    show_progress=True,
    obs={"Sx": Sx},
    tstops=np.linspace(0.0, 1.0, 101, endpoint=True),
)
