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
from typing import Any, Optional, Callable, Iterable, Union, Tuple, List, Dict

import numpy as np

import jax
from jax import numpy as jnp
from jax import tree_map
from jax.util import as_hashable_function

import flax
from flax import linen as nn
from flax import serialization

import netket
from netket import jax as nkjax
from netket import utils
from netket.hilbert import AbstractHilbert
from netket.sampler import Sampler, SamplerState, ExactSampler
from netket.stats import Stats, statistics, mean, sum_inplace
from netket.utils import flax as flax_utils
from netket.optim import SR
from netket.operator import (
    AbstractOperator,
    define_local_cost_function,
    local_cost_function,
    local_value_cost,
)

from .base import VariationalState

PyTree = Any
PRNGKey = jnp.ndarray
SEED = Union[int, PRNGKey]
InitFunType = Callable[
    [nn.Module, Iterable[int], PRNGKey, np.dtype], Tuple[Optional[PyTree], PyTree]
]
AFunType = Callable[[nn.Module, PyTree, jnp.ndarray], jnp.ndarray]
ATrainFunType = Callable[
    [nn.Module, PyTree, jnp.ndarray, Union[bool, PyTree]], jnp.ndarray
]


def compute_chain_length(n_chains, n_samples):
    if n_samples <= 0:
        raise ValueError("Invalid number of samples: n_samples={}".format(n_samples))

    n_chains = n_chains * utils.n_nodes
    chain_length = int(np.ceil(n_samples / n_chains))
    return chain_length


class MCState(VariationalState):
    """Variational State for a Variational Neural Quantum State.

    The state is sampled according to the provided sampler.
    """

    # model: Any
    # """The model"""
    model_state: Optional[PyTree]
    """An Optional PyTree encoding a mutable state of the model that is not trained."""

    _sampler: Sampler
    """The sampler used to sample the hilbert space."""
    sampler_state: SamplerState
    """The current state of the sampler"""

    _n_samples: int = 0
    """Total number of samples across all mpi processes."""
    _n_discard: int = 0
    """Number of samples discarded at the beginning of every Monte-Carlo chain."""
    _samples: Optional[jnp.ndarray] = None
    """Cached samples obtained with the last sampling."""

    _init_fun: Callable = None
    _apply_fun: Callable = None
    _apply_train_fun: Callable = None

    def __init__(
        self,
        sampler: Sampler,
        model=None,
        *,
        n_samples: int = 1000,
        n_discard: Optional[int] = None,
        variables: Optional[PyTree] = None,
        init_fun: InitFunType = None,
        apply_fun: Callable = None,
        sample_fun: Callable = None,
        seed: Optional[SEED] = None,
        sampler_seed: Optional[SEED] = None,
        mutable: bool = False,
        training_kwargs: Dict = {},
    ):
        """
        Constructs the MCState.

        Arguments:
            sampler: The sampler
            model: (Optional) The model. If not provided, you must provide init_fun and apply_fun.

        Additional Arguments:
            n_samples: the total number of samples across chains and processes when sampling (default=1000).
            n_discard: number of discarded samples at the beginning of each monte-carlo chain (default=n_samples/10).
            parameters: Optional PyTree of weights from which to start.
            seed: rng seed used to generate a set of parameters (only if parameters is not passed). Defaults to a random one.
            sampler_seed: rng seed used to initialise the sampler. Defaults to a random one.
            mutable: Dict specifing mutable arguments. Use it to specify if the model has a state that can change
                during evaluation, but that should not be optimised. See also flax.linen.module.apply documentation
                (default=False)
            init_fun: Function of the signature f(model, shape, rng_key, dtype) -> Optional_state, parameters used to
                initialise the parameters. Defaults to the standard flax initialiser. Only specify if your network has
                a non-standard init method.
            apply_fun: Function of the signature f(model, variables, σ) that should evaluate the model. Defafults to
                `model.apply(variables, σ)`. specify only if your network has a non-standard apply method.
            training_kwargs: a dict containing the optionaal keyword arguments to be passed to the apply_fun during training.
                Useful for example when you have a batchnorm layer that constructs the average/mean only during training.
        """
        super().__init__(sampler.hilbert)

        # Init type 1: pass in a model
        if model is not None:
            # exetract init and apply functions
            # Wrap it in an HashablePartial because if two instances of the same model are provided,
            # model.apply and model2.apply will be different methods forcing recompilation, but
            # model and model2 will have the same hash.
            self._init_fun = nkjax.HashablePartial(
                lambda model, *args, **kwargs: model.init(*args, **kwargs), model
            )
            self._apply_fun = nkjax.HashablePartial(
                lambda model, *args, **kwargs: model.apply(*args, **kwargs), model
            )

        elif apply_fun is not None:
            self._apply_fun = apply_fun

            if init_fun is not None:
                self._init_fun = init_fun
            elif variables is None:
                raise ValueError(
                    "If you don't provide variables, you must pass a valid init_fun."
                )
        else:
            raise ValueError(
                "Must either pass the model or apply_fun, otherwise how do you think we"
                "gonna evaluate the model?"
            )

        if sample_fun is not None:
            self._sample_fun = sample_fun
        else:
            self._sample_fun = self._apply_fun

        self.mutable = mutable
        self.training_kwargs = flax.core.freeze(training_kwargs)

        if variables is not None:
            self.variables = variables
        else:
            self.init(seed)

        self._sampler_seed = sampler_seed
        self.sampler = sampler

        self.n_samples = n_samples
        self.n_discard = n_discard

    def init(self, seed=None, dtype=jnp.float32):
        """
        Initialises the variational parameters of the variational state.
        """
        if self._init_fun is None:
            raise RuntimeError(
                "Cannot initialise the parameters of this state"
                "because you did not supply a valid init_function."
            )

        key = nkjax.PRNGKey(seed)

        dummy_input = jnp.zeros((1, self.hilbert.size), dtype=dtype)

        variables = self._init_fun({"params": key}, dummy_input)
        self.variables = variables

    @property
    def sampler(self) -> Sampler:
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: Sampler):
        if not isinstance(sampler, Sampler):
            raise TypeError(
                "The sampler should be a subtype of netket.sampler.Sampler, but {} is not.".format(
                    type(sampler)
                )
            )

        self._sampler = sampler
        self.sampler_state = self.sampler.init_state(
            self._apply_fun, self.variables, seed=self._sampler_seed
        )
        self.reset()

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n_samples: int):
        chain_length = compute_chain_length(self.sampler.n_chains, n_samples)

        n_samples_per_node = chain_length * self.sampler.n_chains
        n_samples = n_samples_per_node * utils.n_nodes

        self._n_samples = n_samples
        self._chain_length = chain_length
        self._n_samples_per_node = n_samples_per_node
        self.reset()

    @property
    def chain_length(self) -> int:
        """
        Length of the markov chain used for sampling configurations.

        If running under MPI, the total samples will be n_nodes * chain_length * n_batches.
        """
        return self._chain_length

    @chain_length.setter
    def chain_length(self, length: int):
        self.n_samples = length * self.sampler.n_chains * utils.n_nodes
        self.reset()

    @property
    def n_discard(self) -> int:
        """
        Number of discarded samples at the beginning of the markov chain.
        """
        return self._n_discard

    @n_discard.setter
    def n_discard(self, n_discard: Optional[int]):
        if n_discard is not None and n_discard < 0:
            raise ValueError(
                "Invalid number of discarded samples: n_discard={}".format(n_discard)
            )

        # don't discard if ExactSampler
        if isinstance(self.sampler, ExactSampler):
            if n_discard is not None and n_discard > 0:
                warnings.warn(
                    "Exact Sampler does not need to discard samples. Setting n_discard to 0."
                )
            n_discard = 0

        self._n_discard = (
            int(n_discard) if n_discard != None else self.chain_length // 10
        )

    def reset(self):
        """
        Resets the sampled states. This method is called automatically every time
        that the parameters/state is updated.
        """
        self._samples = None

    def sample(
        self,
        *,
        chain_length: Optional[int] = None,
        n_samples: Optional[int] = None,
        n_discard: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Sample a certain number of configurations.

        If one among chain_leength or n_samples is defined, that number of samples
        are gen erated. Otherwise the value set internally is used.

        Args:
            chain_length: The length of the markov chains.
            n_samples: The total number of samples across all MPI ranks.
            n_discard: Number of discarded samples at the beginning of the markov chain.
        """
        if n_samples is None and chain_length is None:
            chain_length = self.chain_length
        elif chain_length is None:
            chain_length = compute_chain_length(self.sampler.n_chains, n_samples)

        if n_discard is None:
            n_discard = self.n_discard

        self.sampler_state = self.sampler.reset(
            self._apply_fun, self.variables, self.sampler_state
        )

        if self.n_discard > 0:
            _, self.sampler_state = self.sampler.sample(
                self._apply_fun,
                self.variables,
                state=self.sampler_state,
                chain_length=n_discard,
            )

        self._samples, self.sampler_state = self.sampler.sample(
            self._apply_fun,
            self.variables,
            state=self.sampler_state,
            chain_length=chain_length,
        )
        return self._samples

    @property
    def samples(self) -> jnp.ndarray:
        """
        Returns the set of cached samples.

        The samples returnede are guaranteed valid for the current state of
        the variational state. If no cached parameters are available, then
        they are sampled first and then cached.

        To obtain a new set of samples either use :ref:`reset` or :ref:`sample`.
        """
        if self._samples is None:
            self.sample()
        return self._samples

    def evaluate(self, σ) -> jnp.ndarray:
        return self._apply_fun(self.variables, σ)

    def expect(self, Ô: AbstractOperator) -> Stats:
        if not self.hilbert == Ô.hilbert:
            return NotImplemented

        σ = self.samples
        σ_shape = σ.shape
        σ = σ.reshape((-1, σ_shape[-1]))

        σp, mels = Ô.get_conn_padded(np.asarray(σ))
        O_loc = local_cost_function(
            local_value_cost,
            self._apply_fun,
            self.variables,
            σp,
            mels,
            σ,
        ).reshape(σ_shape[:-1])

        # notice that loc.T is passed to statistics, since that function assumes
        # that the first index is the batch index.
        return statistics(O_loc.T)

    def expect_and_grad(
        self,
        Ô: AbstractOperator,
        *,
        mutable=None,
        is_hermitian=None,
        centered=True,
    ) -> Tuple[Stats, PyTree]:
        if not self.hilbert == Ô.hilbert:
            return NotImplemented

        # should check if it is hermitian
        # if hermitian...

        # By default the mutable variables are defined in the variationalstate itself.
        if mutable is None:
            mutable = self.mutable

        if is_hermitian is None:
            is_hermitian = Ô.is_hermitian

        σ = self.samples
        σ_shape = σ.shape
        σ = σ.reshape((-1, σ_shape[-1]))

        # Compute local terms....
        # TODO Run this in jitted block and compute gradient directly
        # maybe return the stat object as aux so we can do only one
        # forward pass.
        σp, mels = Ô.get_conn_padded(np.asarray(σ))
        O_loc = local_cost_function(
            local_value_cost,
            self._apply_fun,
            self.variables,
            σp,
            mels,
            σ,
        )
        Ō = statistics(O_loc.reshape(σ_shape[:-1]).T)

        if is_hermitian and centered:
            out = grad_expect_hermitian(
                self._apply_fun,
                self.parameters,
                self.model_state,
                σ,
                O_loc,
                mutable,
                self.n_samples,
            )
        else:
            if centered:
                out = grad_expect_non_hermitian(
                    self._apply_fun,
                    self.parameters,
                    self.model_state,
                    σ,
                    mutable,
                    self.n_samples,
                    σp,
                    mels,
                )
            else:
                out = grad_expect_non_hermitian_non_centered(
                    self._apply_fun,
                    self.parameters,
                    self.model_state,
                    σ,
                    mutable,
                    self.n_samples,
                    σp,
                    mels,
                )

        if mutable is False:
            Ō_grad, _ = out
        else:
            Ō_grad, self.model_state = out

        return Ō, Ō_grad

    def quantum_geometric_tensor(self, sr: SR):
        r"""Computes an estimate of the quantum geometric tensor G_ij.
        This function returns a linear operator that can be used to apply G_ij to a given vector
        or can be converted to a full matrix.

        Returns:
            scipy.sparse.linalg.LinearOperator: A linear operator representing the quantum geometric tensor.
        """

        return sr.create(
            apply_fun=self._apply_fun,
            params=self.parameters,
            samples=self.samples,
            model_state=self.model_state,
            x0=None,
            sr=sr,
        )


@partial(jax.jit, static_argnums=(0, 5))
def grad_expect_hermitian(
    model_apply_fun, parameters, model_state, σ, O_loc, mutable, n_samples
) -> PyTree:

    if jnp.ndim(σ) != 2:
        σ_shape = σ.shape
        σ = σ.reshape((-1, σ_shape[-1]))

    O_loc -= mean(O_loc)

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    if mutable is False:
        _, vjp_fun = nkjax.vjp(
            lambda w: model_apply_fun({"params": w, **model_state}, σ),
            parameters,
            conjugate=True,
        )
        new_model_state = None
    else:
        _, vjp_fun, new_model_state = nkjax.vjp(
            lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
            parameters,
            conjugate=True,
            has_aux=True,
        )
    Ō_grad = vjp_fun(O_loc / n_samples)[0]

    return tree_map(sum_inplace, Ō_grad), new_model_state


@partial(jax.jit, static_argnums=(0, 4))
def grad_expect_non_hermitian(
    model_apply_fun,
    parameters,
    model_state,
    σ,
    mutable,
    n_samples,
    σp,
    mels,
) -> PyTree:
    if jnp.ndim(σ) != 2:
        σ_shape = σ.shape
        σ = σ.reshape((-1, σ_shape[-1]))

    has_aux = mutable is not False
    if not has_aux:
        out_axes = (0, 0)
    else:
        out_axes = (0, 0, 0)

    if not has_aux:
        logpsi = lambda w, σ: model_apply_fun({"params": w, **model_state}, σ)
    else:
        # TODO: output the mutable state
        logpsi = lambda w, σ: model_apply_fun(
            {"params": w, **model_state}, σ, mutable=mutable
        )[0]

    def local_value(pars, σp, mel, σ):
        return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))

    grad_fun = jax.vmap(
        nkjax.value_and_grad(local_value, argnums=0, has_aux=has_aux),
        in_axes=(None, 0, 0, 0),
        out_axes=out_axes,
    )

    if not has_aux:
        Ō, Ō_grad = grad_fun(
            parameters,
            σp,
            mels,
            σ,
        )
        new_model_state = None
    else:
        Ō, Ō_grad = grad_fun(
            parameters,
            σp,
            mels,
            σ,
        )
        Ō, new_model_state = Ō

    return tree_map(lambda x: sum_inplace(x) / n_samples, Ō_grad), new_model_state


@partial(jax.jit, static_argnums=(0, 4))
def grad_expect_non_hermitian_non_centered(
    model_apply_fun,
    parameters,
    model_state,
    σ,
    mutable,
    n_samples,
    σp,
    mels,
) -> PyTree:
    if jnp.ndim(σ) != 2:
        σ_shape = σ.shape
        σ = σ.reshape((-1, σ_shape[-1]))

    has_aux = mutable is not False
    if not has_aux:
        out_axes = (0, 0)
    else:
        out_axes = (0, 0, 0)

    if not has_aux:
        logpsi = lambda w, σ: model_apply_fun({"params": w, **model_state}, σ)
    else:
        # TODO: output the mutable state
        logpsi = lambda w, σ: model_apply_fun(
            {"params": w, **model_state}, σ, mutable=mutable
        )[0]

    from netket.operator._der_local_values import (
        _local_value_and_grad_notcentered_kernel,
    )

    # TODO Properely support has_aux
    Ō, Ō_grad = jax.vmap(
        _local_value_and_grad_notcentered_kernel(logpsi, pars, σp, mel, σ),
        in_axes=(None, 0, 0, 0),
        out_axes=(0, 0),
    )

    return tree_map(lambda x: sum_inplace(x) / n_samples, Ō_grad), model_state


# serialization


def serialize_classical_variational_state(vstate):
    state_dict = {
        "variables": vstate.variables,
        "sampler_state": vstate.sampler_state,
        "n_samples": vstate.n_samples,
        "n_discard": vstate.n_discard,
    }
    return state_dict


def deserialize_classical_variational_state(vstate, state_dict):
    import copy

    new_vstate = copy.copy(vstate)

    new_vstate.variables = state_dict["variables"]
    new_vstate.sampler_state = state_dict["sampler_state"]
    new_vstate.n_samples = state_dict["n_samples"]
    new_vstate.n_discard = state_dict["n_discard"]

    return new_vstate


serialization.register_serialization_state(
    MCState,
    serialize_classical_variational_state,
    deserialize_classical_variational_state,
)
