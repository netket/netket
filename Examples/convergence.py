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


import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from netket._src.vqs.check_mc_convergence import *


import netket as nk
from netket import experimental as nkx
import optax

nk.config.netket_experimental_fft_autocorrelation = True

# 1D Lattice
L = 20
nchains = 256
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=nchains)

# Optimizer with a decreasing learning rate
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

# Variational monte carlo driver with a variational state
gs = nk.driver.VMC_SR(
    ha,
    op,
    variational_state=vs,
    diag_shift=0.01,
)

# Run the optimization for 500 iterations
gs.run(
    n_iter=30,
    out=(
        nk.logging.SaveVariationalState(path="ost", interval=10),
        nk.logging.RuntimeLog(),
    ),
    timeit=True,
)

# sampler_state = vs.sampler_state
# vs.sampler = nk.sampler.MetropolisLocal(hi, n_chains=nchains, sweep_size=1)
# vs.sampler_state = sampler_state

# os = nk.stats.online_statistics(vs.local_estimators(ha))
# print(0, os, "tau:", os.tau_corr)
# for i in range(0, 100):
#     vs.sample()
#     os = nk.stats.online_statistics(vs.local_estimators(ha), old_estimator=os)
#     print(i, os, "tau:", os.tau_corr)


# vs.chain_length = 5000
# n_blocks = 25

# s = vs.samples
# eloc = vs.local_estimators(ha)
# block_size = vs.chain_length / n_blocks
# os = None
# for i in range(n_blocks):
#     block = eloc[:, int(i * block_size) : int((i + 1) * block_size)]
#     os = nk.stats.online_statistics(block, old_estimator=os, max_lag=256)
#     print(i, os, "tau:", os.tau_corr)

# print(nk.stats.statistics(eloc))
# print(os.get_stats())

# c_ti = []
# tau_i = []
# for el in tqdm(eloc):
#     rho, tau = autocorr_and_tau(el)
#     c_ti.append(rho)
#     tau_i.append(tau)

# c_ti = np.array(c_ti)
# tau_i = np.array(tau_i)

# plt.ion()
# plt.plot(c_ti.mean(axis=0))
# plt.plot(os.acf)


# print("tau from netket:", os.tau_corr)
# print("tau from acf:", os.tau_corr_acf)
# print("tau from autocorr:", tau_i.mean())
