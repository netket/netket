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


import netket as nk
from load_data import load


path_to_samples = 'ising1d_train_samples.txt'
path_to_bases = 'ising1d_train_bases.txt'

# Load the data
hi, rotations, training_samples, training_bases = load(
    path_to_samples, path_to_bases)

# Machine
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(seed=1234, sigma=0.001)

# Sampler
sa = nk.sampler.MetropolisLocal(machine=ma)

# Optimizer
op = nk.optimizer.AdaDelta()

ha = nk.operator.Ising(h=1.0, hilbert=hi)

# Quantum State Reconstruction
qst = nk.unsupervised.Qsr(
    sampler=sa,
    optimizer=op,
    batch_size=1000,
    n_samples=10000,
    niter_opt=10000,
    rotations=rotations,
    output_file="output",
    samples=training_samples,
    bases=training_bases)

qst.add_observable(ha, "Energy")

qst.run()
