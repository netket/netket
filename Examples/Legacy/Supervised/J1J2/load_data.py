# Copyright 2020 The Netket Authors. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  # You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math as m
import netket.graph as gr
import netket.hilbert as hs


def load(path_to_samples, path_to_targets):
    tsamples = np.loadtxt(path_to_samples)
    ttargets = np.loadtxt(path_to_targets, dtype=complex)

    # Create the hilbert space
    # TODO remove Hypercube here and put customgraph
    g = gr.Hypercube(length=len(tsamples[0]), n_dim=1)
    hi = hs.Qubit(N=g.n_nodes)

    training_samples = []
    training_targets = []
    for i in range(len(tsamples)):
        training_samples.append(tsamples[i].tolist())
        # training_targets.append([ttargets[i]])
        training_targets.append([np.log(ttargets[i])])

    return hi, training_samples, training_targets
