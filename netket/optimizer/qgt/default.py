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

from typing import Callable, Optional, Union, Tuple, Any
from functools import partial

import jax
from jax import numpy as jnp
from flax import struct

from netket.utils.types import PyTree
from netket.utils import n_nodes
from netket.stats import sum_inplace
import netket.jax as nkjax

from .qgt_jacobian_dense import QGTJacobianDense
from .qgt_jacobian_pytree import QGTJacobianPyTree
from .qgt_onthefly import QGTOnTheFly


def default_qgt_matrix(variational_state, solver=False):
    # arbitrary heuristic: for more than 2

    # an rbm has 3
    n_param_blocks = len(jax.tree_leaves(variational_state.parameters))
    n_params = variational_state.n_parameters

    # Completely arbitrary
    if n_param_blocks > 6 and n_params > 800:
        return QGTJacobianPyTree
    else:
        return QGTOnTheFly


class QGTAuto:
    """
    Automatically select the 'best' Quantum Geometric Tensor
    computing format acoording to some rather untested heuristic.
    """

    _last_vstate = None
    _last_matrix = None

    def __init__(self, solver=None):
        self._solver = solver

    def __call__(self, variational_state, *args, **kwargs):
        if self._last_vstate != variational_state:
            self._last_vstate = variational_state

            self._last_matrix = default_qgt_matrix(
                variational_state, solver=self._solver
            )

        return self._last_matrix(variational_state, *args, **kwargs)
