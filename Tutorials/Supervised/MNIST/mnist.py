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
import numpy as np
from mpi4py import MPI
from load_MNIST import load_training

# Load the Hilbert space info and data
num_images = 1000
hi, training_samples, training_targets = load_training(num_images)

for i in range(num_images):
    training_targets[i] = np.log(training_targets[i] + 1)

# Machine
ma = nk.machine.RbmMultiVal(hilbert=hi, alpha=10)

## Layers
#L = 28*28
#layers = (
#    nk.layer.FullyConnected(input_size=L, output_size=100),
#    nk.layer.Relu(input_size=100),
#    nk.layer.FullyConnected(input_size=100, output_size=10),
#)
#
#ma = nk.machine.FFNN(hilbert=hi, layers=layers)

ma.init_random_parameters(seed=1234, sigma=0.001)

# Sampler
sa = nk.sampler.MetropolisLocal(machine=ma)

# Optimizer
op = nk.optimizer.Sgd(1e-2)

# Quantum State Reconstruction
spvsd = nk.supervised.supervised(
    sampler=sa,
    optimizer=op,
    batch_size=64,
    niter_opt=10,
    output_file="output",
    samples=training_samples,
    targets=training_targets)

spvsd.run()
