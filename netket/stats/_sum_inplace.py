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

from functools import singledispatch
import numpy as _np
import jax

from netket.utils.mpi import (
    mpi_available as _mpi_available,
    n_nodes as _n_nodes,
    mpi_sum,
    mpi_sum_jax,
)


@singledispatch
def sum_inplace(x):
    """
    Computes the elementwise sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy
    might be returned.

    Args:
        a: The input array, which will usually be overwritten in place.
    Returns:
        out: The reduced array.
    """
    raise TypeError("Unknown type to perform dispatch upon: {}".format(type(x)))


#######
# Scalar and numpy
@sum_inplace.register(_np.ndarray)
@sum_inplace.register(complex)
@sum_inplace.register(_np.float64)
@sum_inplace.register(_np.float32)
@sum_inplace.register(_np.complex64)
@sum_inplace.register(_np.complex128)
@sum_inplace.register(float)
@sum_inplace.register(int)
def sum_inplace_scalar(a):
    return mpi_sum(a)


# Jax
#
@sum_inplace.register(jax.interpreters.xla.DeviceArray)
@sum_inplace.register(jax.core.Tracer)
def sum_inplace_jax(x):
    res, _ = mpi_sum_jax(x)
    return res
