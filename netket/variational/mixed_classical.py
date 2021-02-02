# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from functools import partial
from typing import Any, Optional, Callable, Iterable, Union, Tuple, List

import numpy as np

import jax
from jax import numpy as jnp
from jax import tree_map
from jax.util import as_hashable_function

import flax
from flax import linen as nn

import netket
from netket import jax as nkjax
from netket import utils
from netket.hilbert import AbstractHilbert
from netket.sampler import Sampler, SamplerState, ExactSampler
from netket.stats import Stats, statistics, mean, sum_inplace
from netket.utils import flax as flax_utils, maybe_wrap_module
from netket.optim import SR
from netket.operator import (
    AbstractOperator,
    define_local_cost_function,
    local_cost_function,
    local_value_cost,
    local_value_op_op_cost,
)

from .base import VariationalState, VariationalMixedState
from .classical import MCState

PyTree = Any
PRNGKey = jnp.ndarray
InitFunType = Callable[
    [nn.Module, Iterable[int], PRNGKey, np.dtype], Tuple[Optional[PyTree], PyTree]
]
AFunType = Callable[[nn.Module, PyTree, jnp.ndarray], jnp.ndarray]
ATrainFunType = Callable[
    [nn.Module, PyTree, jnp.ndarray, Union[bool, PyTree]], jnp.ndarray
]


def apply_diagonal(bare_afun, w, x, *args, **kwargs):
    x = jnp.hstack((x, x))
    return bare_afun(w, x, *args, **kwargs)


class MCMixedState(VariationalMixedState, MCState):
    def __init__(
        self,
        sampler,
        model=None,
        *,
        sampler_diag: Sampler = None,
        n_samples_diag: int = 1000,
        n_discard_diag: Optional[int] = None,
        seed=nkjax.PRNGKey(),
        sampler_seed: Optional[int] = None,
        **kwargs,
    ):
        seed, seed_diag = jax.random.split(nkjax.PRNGKey(seed))
        if sampler_seed is None:
            sampler_seed_diag = None
        else:
            sampler_seed, sampler_seed_diag = jax.random.split(
                nkjax.PRNGKey(sampler_seed)
            )

        self.diagonal = None

        hilbert_physical = sampler.hilbert.physical

        super().__init__(
            sampler.hilbert.physical,
            sampler,
            model,
            **kwargs,
            seed=seed,
            sampler_seed=sampler_seed,
        )

        if sampler_diag is None:
            sampler_diag = sampler.replace(hilbert=hilbert_physical)

        sampler_diag = sampler_diag.replace(machine_pow=1)

        diagonal_apply_fun = nkjax.HashablePartial(apply_diagonal, self._apply_fun)

        for kw in ["n_samples", "n_discard"]:
            if kw in kwargs:
                kwargs.pop(kw)

        self.diagonal = MCState(
            sampler_diag,
            apply_fun=diagonal_apply_fun,
            n_samples=n_samples_diag,
            n_discard=n_discard_diag,
            variables=self.variables,
            seed=seed_diag,
            sampler_seed=sampler_seed_diag,
            **kwargs,
        )

        # build the

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        if self.diagonal is not None:
            self.diagonal.variables = variables

    @property
    def sampler_diag(self):
        return self.diagonal.sampler

    @sampler_diag.setter
    def sampler_diag(self, sampler):
        self.diagonal.sampler = sampler

    @property
    def n_samples_diag(self):
        return self.diagonal.n_samples

    @n_samples_diag.setter
    def n_samples_diag(self, n_samples):
        self.diagonal.n_samples = n_samples

    @property
    def chain_length_diag(self) -> int:
        return self.diagonal.chain_length_diag

    @chain_length_diag.setter
    def chain_length_diag(self, length: int):
        self.diagonal.chain_length_diag = length

    @property
    def n_discard_diag(self) -> int:
        return self.diagonal.n_discard_diag

    @n_discard_diag.setter
    def n_discard_diag(self, n_discard: Optional[int]):
        self.diagonal.n_discard_diag = n_discard

    @MCState.parameters.setter
    def parameters(self, pars: PyTree):
        MCState.parameters.fset(self, pars)
        if self.diagonal is not None:
            self.diagonal.parameters = pars

    @MCState.model_state.setter
    def model_state(self, state: PyTree):
        MCState.model_state.fset(self, state)
        if self.diagonal is not None:
            self.diagonal.model_state = state

    def reset(self):
        super().reset()
        if self.diagonal is not None:
            self.diagonal.reset()

    def expect_operator(self, Ô: AbstractOperator) -> Stats:
        σ = self.diagonal.samples
        σ_shape = σ.shape
        σ = σ.reshape((-1, σ.shape[-1]))

        σ_np = np.asarray(σ)
        σp, mels = Ô.get_conn_padded(σ_np)

        # now we have to concatenate the two
        O_loc = local_cost_function(
            local_value_op_op_cost,
            self._apply_fun,
            self.variables,
            σp,
            mels,
            σ,
        ).reshape(σ_shape[:-1])

        # notice that loc.T is passed to statistics, since that function assumes
        # that the first index is the batch index.
        return statistics(O_loc.T)

    def expect_and_grad_operator(self, Ô: AbstractOperator, centered=True) -> Stats:
        raise NotImplementedError
