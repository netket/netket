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
from netket import config
from netket.hilbert import AbstractHilbert
from netket.sampler import Sampler, SamplerState, ExactSampler
from netket.stats import Stats, statistics, mean, sum_inplace
from netket.utils import flax as flax_utils, n_nodes, maybe_wrap_module, deprecated
from netket.utils.types import PyTree, PRNGKeyT, SeedT, Shape, NNInitFunc
from netket.optimizer import SR
from netket.operator import (
    AbstractOperator,
    AbstractSuperOperator,
    define_local_cost_function,
    local_cost_function,
    local_value_cost,
    Squared,
)

from .base import VariationalState

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


def local_value_kernel(logpsi, pars, σ, σp, mel):
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


def local_value_squared_kernel(logpsi, pars, σ, σp, mel):
    return jnp.abs(local_value_kernel(logpsi, pars, σ, σp, mel)) ** 2


@partial(jax.jit, static_argnums=0)
def apply(fun, *args):
    return fun(*args)


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
    """The function used to initialise the parameters and model_state"""
    _apply_fun: Callable = None
    """The function used to evaluate the model"""

    def __init__(
        self,
        sampler: Sampler,
        model=None,
        *,
        n_samples: int = 1000,
        n_discard: Optional[int] = None,
        variables: Optional[PyTree] = None,
        init_fun: NNInitFunc = None,
        apply_fun: Callable = None,
        sample_fun: Callable = None,
        seed: Optional[SeedT] = None,
        sampler_seed: Optional[SeedT] = None,
        mutable: bool = False,
        training_kwargs: Dict = {},
    ):
        """
        Constructs the MCState.

        Arguments:
            sampler: The sampler
            model: (Optional) The model. If not provided, you must provide init_fun and apply_fun.

        Keyword Arguments:
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
            _, model = maybe_wrap_module(model)

            self.model = model

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
            self.init(seed, dtype=sampler.dtype)

        if sampler_seed is None and seed is not None:
            key, key2 = jax.random.split(nkjax.PRNGKey(seed), 2)
            sampler_seed = key2

        self._sampler_seed = sampler_seed
        self.sampler = sampler

        self.n_samples = n_samples
        self.n_discard = n_discard

    def init(self, seed=None, dtype=None):
        """
        Initialises the variational parameters of the variational state.
        """
        if self._init_fun is None:
            raise RuntimeError(
                "Cannot initialise the parameters of this state"
                "because you did not supply a valid init_function."
            )

        if dtype is None:
            dtype = self.sampler.dtype

        key = nkjax.PRNGKey(seed)

        dummy_input = jnp.zeros((1, self.hilbert.size), dtype=dtype)

        variables = self._init_fun({"params": key}, dummy_input)
        self.variables = variables

    @property
    def sampler(self) -> Sampler:
        """The Monte Carlo sampler used by this Monte Carlo variational state."""
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
        """The total number of samples generated at every sampling step."""
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
            int(n_discard) if n_discard is not None else self.n_samples // 10
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

    def log_value(self, σ: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the variational state for a batch of states and returns
        the logarithm of the amplitude of the quantum state. For pure states,
        this is :math:`log(<σ|ψ>)`, whereas for mixed states this is
        :math:`log(<σr|ρ|σc>)`, where ψ and ρ are respectively a pure state
        (wavefunction) and a mixed state (density matrix).
        For the density matrix, the left and right-acting states (row and column)
        are obtained as :code:`σr=σ[::,0:N]` and :code:`σc=σ[::,N:]`.

        Given a batch of inputs (Nb, N), returns a batch of outputs (Nb,).
        """
        return apply(self._apply_fun, self.variables, σ)

    @deprecated("Use MCState.log_value(σ) instead.")
    def evaluate(self, σ: jnp.ndarray) -> jnp.ndarray:
        """
        DEPRECATED: use log_value instead.
        """
        return self.log_value(σ)

    def expect(self, Ô: AbstractOperator) -> Stats:
        if not self.hilbert == Ô.hilbert:
            return NotImplemented

        σ = self.samples

        if isinstance(Ô, Squared):
            Ô = Ô.parent
            kernel = local_value_squared_kernel
        else:
            kernel = local_value_kernel

        σp, mels = Ô.get_conn_padded(np.asarray(σ).reshape((-1, σ.shape[-1])))

        return _expect(
            self.sampler.machine_pow,
            self._apply_fun,
            kernel,
            self.parameters,
            self.model_state,
            σ,
            σp,
            mels,
        )

    def expect_and_grad(
        self,
        Ô: AbstractOperator,
        *,
        mutable=None,
        is_hermitian=None,
    ) -> Tuple[Stats, PyTree]:
        if not self.hilbert == Ô.hilbert:
            return NotImplemented

        # By default the mutable variables are defined in the variationalstate itself.
        if mutable is None:
            mutable = self.mutable

        if is_hermitian is None:
            is_hermitian = Ô.is_hermitian

        if isinstance(Ô, Squared):
            squared_operator = True
        else:
            squared_operator = False

        if squared_operator:
            Ô = Ô.parent

        σ = self.samples
        σp, mels = Ô.get_conn_padded(np.asarray(σ.reshape((-1, σ.shape[-1]))))

        if is_hermitian:
            if squared_operator:
                if isinstance(Ô, AbstractSuperOperator):
                    Ō, Ō_grad, new_model_state = grad_expect_operator_Lrho2(
                        self._apply_fun,
                        mutable,
                        self.parameters,
                        self.model_state,
                        self.samples,
                        σp,
                        mels,
                    )
                else:
                    Ō, Ō_grad, new_model_state = grad_expect_operator_kernel(
                        self.sampler.machine_pow,
                        self._apply_fun,
                        local_value_squared_kernel,
                        mutable,
                        self.parameters,
                        self.model_state,
                        self.samples,
                        σp,
                        mels,
                    )

            else:
                Ō, Ō_grad, new_model_state = grad_expect_hermitian(
                    self._apply_fun,
                    mutable,
                    self.parameters,
                    self.model_state,
                    σ,
                    σp,
                    mels,
                )
        else:
            Ō, Ō_grad, new_model_state = grad_expect_operator_kernel(
                self.sampler.machine_pow,
                self._apply_fun,
                local_value_kernel,
                mutable,
                self.parameters,
                self.model_state,
                self.samples,
                σp,
                mels,
            )

        if mutable is not False:
            self.model_state = new_model_state

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

    def to_array(self, normalize: bool = True) -> jnp.ndarray:
        return netket.nn.to_array(
            self.hilbert, self._apply_fun, self.variables, normalize=normalize
        )

    def __repr__(self):
        return (
            "MCState("
            + "\n  hilbert = {},".format(self.hilbert)
            + "\n  sampler = {},".format(self.sampler)
            + "\n  n_samples = {},".format(self.n_samples)
            + "\n  n_discard = {},".format(self.n_discard)
            + "\n  sampler_state = {},".format(self.sampler_state)
            + "\n  n_parameters = {})".format(self.n_parameters)
        )

    def __str__(self):
        return (
            "MCState("
            + "hilbert = {}, ".format(self.hilbert)
            + "sampler = {}, ".format(self.sampler)
            + "n_samples = {})".format(self.n_samples)
        )


@partial(jax.jit, static_argnums=(1, 2))
def _expect(
    machine_pow: int,
    model_apply_fun: Callable,
    local_value_kernel: Callable,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Stats:
    σ_shape = σ.shape

    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    logpsi = lambda w, σ: model_apply_fun({"params": w, **model_state}, σ)

    log_pdf = (
        lambda w, σ: machine_pow * model_apply_fun({"params": w, **model_state}, σ).real
    )

    local_value_vmap = jax.vmap(
        partial(local_value_kernel, logpsi),
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )

    _, Ō_stats = nkjax.expect(
        log_pdf, local_value_vmap, parameters, σ, σp, mels, n_chains=σ_shape[0]
    )

    return Ō_stats


@partial(jax.jit, static_argnums=(0, 1))
def grad_expect_hermitian(
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Tuple[PyTree, PyTree]:

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples = σ.shape[0] * utils.n_nodes

    O_loc = local_cost_function(
        local_value_cost,
        model_apply_fun,
        {"params": parameters, **model_state},
        σp,
        mels,
        σ,
    )

    Ō = statistics(O_loc.reshape(σ_shape[:-1]).T)

    O_loc -= Ō.mean

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
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]

    Ō_grad = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    return Ō, tree_map(sum_inplace, Ō_grad), new_model_state


@partial(jax.jit, static_argnums=(1, 2, 3))
def grad_expect_operator_kernel(
    machine_pow: int,
    model_apply_fun: Callable,
    local_kernel: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Tuple[PyTree, PyTree, Stats]:

    if not config.FLAGS["NETKET_EXPERIMENTAL"]:
        raise RuntimeError(
            """
                           Computing the gradient of a squared or non hermitian 
                           operator is an experimental feature under development 
                           and is known not to return wrong values sometimes.

                           If you want to debug it, set the environment variable
                           NETKET_EXPERIMENTAL=1
                           """
        )

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
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

    log_pdf = (
        lambda w, σ: machine_pow * model_apply_fun({"params": w, **model_state}, σ).real
    )

    def expect_closure(*args):
        local_kernel_vmap = jax.vmap(
            partial(local_kernel, logpsi), in_axes=(None, 0, 0, 0), out_axes=0
        )

        return nkjax.expect(log_pdf, local_kernel_vmap, *args, n_chains=σ_shape[0])

    def expect_closure_pars(pars):
        return expect_closure(pars, σ, σp, mels)

    Ō, Ō_pb, Ō_stats = nkjax.vjp(expect_closure_pars, parameters, has_aux=True)
    Ō_pars_grad = Ō_pb(jnp.ones_like(Ō))

    return (
        Ō_stats,
        tree_map(lambda x: sum_inplace(x) / utils.n_nodes, Ō_pars_grad),
        model_state,
    )


@partial(jax.jit, static_argnums=(0, 1))
def grad_expect_operator_Lrho2(
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Tuple[PyTree, PyTree, Stats]:
    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples_node = σ.shape[0]

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

    local_kernel_vmap = jax.vmap(
        partial(local_value_kernel, logpsi), in_axes=(None, 0, 0, 0), out_axes=0
    )

    # _Lρ = local_kernel_vmap(parameters, σ, σp, mels).reshape((σ_shape[0], -1))
    (
        Lρ,
        der_loc_vals,
    ) = netket.operator._der_local_values_jax._local_values_and_grads_notcentered_kernel(
        logpsi, parameters, σp, mels, σ
    )
    # netket.operator._der_local_values_jax._local_values_and_grads_notcentered_kernel returns a loc_val that is conjugated
    Lρ = jnp.conjugate(Lρ)

    LdagL_stats = statistics((jnp.abs(Lρ) ** 2).T)
    LdagL_mean = LdagL_stats.mean

    # old implementation
    # this is faster, even though i think the one below should be faster
    # (this works, but... yeah. let's keep it here and delete in a while.)
    grad_fun = jax.vmap(nkjax.grad(logpsi, argnums=0), in_axes=(None, 0), out_axes=0)
    der_logs = grad_fun(parameters, σ)
    der_logs_ave = tree_map(lambda x: mean(x, axis=0), der_logs)

    # TODO
    # NEW IMPLEMENTATION
    # This should be faster, but should benchmark as it seems slower
    # to compute der_logs_ave i can just do a jvp with a ones vector
    # _logpsi_ave, d_logpsi = nkjax.vjp(lambda w: logpsi(w, σ), parameters)
    # TODO: this ones_like might produce a complexXX type but we only need floatXX
    # and we cut in 1/2 the # of operations to do.
    # der_logs_ave = d_logpsi(
    #    jnp.ones_like(_logpsi_ave).real / (n_samples_node * utils.n_nodes)
    # )[0]
    der_logs_ave = tree_map(sum_inplace, der_logs_ave)

    def gradfun(der_loc_vals, der_logs_ave):
        par_dims = der_loc_vals.ndim - 1

        _lloc_r = Lρ.reshape((n_samples_node,) + tuple(1 for i in range(par_dims)))

        grad = mean(der_loc_vals.conjugate() * _lloc_r, axis=0) - (
            der_logs_ave.conjugate() * LdagL_mean
        )
        return grad

    LdagL_grad = jax.tree_util.tree_multimap(gradfun, der_loc_vals, der_logs_ave)

    return (
        LdagL_stats,
        LdagL_grad,
        model_state,
    )


# serialization


def serialize_MCState(vstate):
    state_dict = {
        "variables": serialization.to_state_dict(vstate.variables),
        "sampler_state": serialization.to_state_dict(vstate.sampler_state),
        "n_samples": vstate.n_samples,
        "n_discard": vstate.n_discard,
    }
    return state_dict


def deserialize_MCState(vstate, state_dict):
    import copy

    new_vstate = copy.copy(vstate)
    new_vstate.reset()

    new_vstate.variables = serialization.from_state_dict(
        vstate.variables, state_dict["variables"]
    )
    new_vstate.sampler_state = serialization.from_state_dict(
        vstate.sampler_state, state_dict["sampler_state"]
    )
    new_vstate.n_samples = state_dict["n_samples"]
    new_vstate.n_discard = state_dict["n_discard"]

    return new_vstate


serialization.register_serialization_state(
    MCState,
    serialize_MCState,
    deserialize_MCState,
)
