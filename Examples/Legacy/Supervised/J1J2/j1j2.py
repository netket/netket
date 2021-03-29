# Copyright 2020 The Netket Authors. - All Rights Reserved.
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


from netket import legacy as nk
from ed import load_ed_data

L = 10
J2 = 0.4

# Load the Hilbert space info and data
hi, training_samples, training_targets = load_ed_data(L, J2)

# Machine
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Optimizer
op = nk.optimizer.AdaDelta()


spvsd = nk.supervised.Supervised(
    machine=ma,
    optimizer=op,
    n_chains=400,
    samples=training_samples,
    targets=training_targets,
)

n_iter = 4000

# Run with "Overlap_phi" loss. Also available currently is "MSE, Overlap_uni"
spvsd.run(
    n_iter=n_iter,
    loss_function="Overlap_phi",
    out="output",
    save_params_every=50,
)
