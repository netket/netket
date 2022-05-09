# Copyright 2022 The NetKet Authors - All rights reserved.
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

from netket.operator import LocalOperator
import math
import numpy as np


def _one_norm(op: LocalOperator) -> float:
    """
    Computes the 1-norm of the matrix representation of the LocalOperator

    Parameters
    ----------
    op : LocalOperator
        A local operator.

    Returns
    -------
    float
        1-norm of the local operator.
    """
    hilbert = op.hilbert
    operators_dict = op._operators_dict
    local_sizes = np.asarray(hilbert.shape)
    one_norm = 0
    # TODO: this can be hugely accelerated
    for i in range(math.prod(local_sizes)):
        ket = np.asarray(np.unravel_index(i, local_sizes))
        value = 0
        for acting_on, operator in operators_dict.items():
            acting_on = np.atleast_1d(acting_on)
            col_idx = np.ravel_multi_index(
                tuple(ket[acting_on]),
                tuple(local_sizes[acting_on]),
            )
            value += np.abs(np.sum(operator[:, col_idx]))
        if value > one_norm:
            one_norm = value
    return one_norm


def propagator(op: LocalOperator, t: float) -> LocalOperator:
    ...


if __name__ == "__main__":
    import netket as nk
    import time

    hi = nk.hilbert.Fock(N=6, n_max=3)
    op = nk.operator.BoseHubbard(hi, nk.graph.Chain(6), U=1)
    ft = time.time()
    print("One norm of dense", np.linalg.norm(op.to_dense(), ord=1))
    st = time.time()
    print("Dense took", st - ft)
    print("One norm by counting", _one_norm(op.to_local_operator()))
    print("Counting took", time.time() - st)
