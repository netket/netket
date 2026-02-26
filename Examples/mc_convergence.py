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
from netket import experimental as nkx
import optax
import matplotlib.pyplot as plt

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
ma = nk.models.RBM(alpha=1, param_dtype=float)

sa = nk.sampler.MetropolisLocal(hi, n_chains=512, sweep_size=8)
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

op = nk.optimizer.Sgd(learning_rate=0.000)
gs = nk.driver.VMC_SR(
    ha,
    op,
    variational_state=vs,
    diag_shift=0.01,
)

# Run the optimization for 500 iterations
log = nk.logging.JsonLog("test.log")
gs.run(n_iter=500, out=log, timeit=True)

# Look at the convergence information at the end of the optimization to make sure chains
# are thermalized and well mixed
vs.check_mc_convergence(ha, min_chain_length=500, plot=True)
# you should observe the Rhat is below 1.05, the integrated autocorrelation time
# is well below sampler.sweep_size, and the mean and error are stable.

# Compute an observable to a 5% precision
mag = sum(nk.operator.spin.sigmaz(hi, i) for i in range(L)) / L
mag_mean = vs.expect_to_precision(mag, atol=1e-3)
print(mag_mean)
