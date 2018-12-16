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

import numpy as np
import math as m
import netket.operator as op


def load(hi, path_to_samples, path_to_bases):
    tsamples = np.loadtxt(path_to_samples)
    fin_bases = open(path_to_bases, 'r')
    lines = fin_bases.readlines()
    bases = []
    N = hi.size
    for b in lines:
        basis = ""
        for j in range(N):
            basis += b[j]
        bases.append(basis)
    index_list = sorted(range(len(bases)), key=lambda k: bases[k])
    bases.sort()

    training_samples = []
    training_bases = []
    for i in range(len(tsamples)):
        training_samples.append(tsamples[index_list[i]].tolist())

    U_X = (1. / (m.sqrt(2)) *
           np.asarray([[1., 1.], [1., -1.]])).tolist()
    U_Y = (1. / (m.sqrt(2)) *
           np.asarray([[1., -1j], [1., 1j]])).tolist()

    rotations = []

    tmp = 'void'
    b_index = -1
    for b in bases:
        if (b != tmp):
            tmp = b
            localop = op.LocalOperator(hi, 1.0)

            for j in range(N):
                if (tmp[j] == 'X'):
                    localop = localop * op.LocalOperator(hi, U_X, [j])
                if (tmp[j] == 'Y'):
                    localop = localop * op.LocalOperator(hi, U_Y, [j])

            rotations.append(localop)
            b_index += 1
        training_bases.append(b_index)

    return tuple(rotations), training_samples, training_bases
