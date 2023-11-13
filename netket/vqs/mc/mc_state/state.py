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
from typing import Any, Callable, Optional

import numpy as np

import jax
from jax import numpy as jnp

import flax
from flax import serialization
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket import nn
from netket.stats import Stats
from netket.operator import AbstractOperator, Squared
from netket.sampler import Sampler, SamplerState
from netket.utils import (
    maybe_wrap_module,
    mpi,
    wrap_afun,
    wrap_to_support_scalar,
)
from netket.utils.types import PyTree, SeedT, NNInitFunc
from netket.optimizer import LinearOperator
from netket.optimizer.qgt import QGTAuto

from netket.vqs.base import VariationalState, expect, expect_and_grad, expect_and_forces
from netket.vqs.mc import get_local_kernel, get_local_kernel_arguments


def compute_chain_length(n_chains, n_samples):
    if n_samples <= 0:
        raise ValueError(f"Invalid number of samples: n_samples={n_samples}")

    chain_length = int(np.ceil(n_samples / n_chains))

    n_samples_new = chain_length * n_chains
    n_samples_per_rank_new = n_samples_new // mpi.n_nodes

    if n_samples_new != n_samples:
        n_samples_per_rank = n_samples // mpi.n_nodes
        warnings.warn(
            f"n_samples={n_samples} ({n_samples_per_rank} per MPI rank) does not "
            f"divide n_chains={n_chains}, increased to {n_samples_new} "
            f"({n_samples_per_rank_new} per MPI rank)",
            UserWarning,
            stacklevel=3,
        )

    return chain_length


def check_chunk_size(n_samples, chunk_size):
    n_samples_per_rank = n_samples // mpi.n_nodes

    if chunk_size is not None:
        if chunk_size < n_samples_per_rank and n_samples_per_rank % chunk_size != 0:
            raise ValueError(
                f"chunk_size={chunk_size}`<`n_samples_per_rank={n_samples_per_rank}, "
                "chunk_size is not an integer fraction of `n_samples_per rank`. This is"
                "unsupported. Please change `chunk_size` so that it divides evenly the"
                "number of samples per rank or set it to `None` to disable chunking."
            )


def _is_power_of_two(n: int) -> bool:
    return (n != 0) and (n & (n - 1) == 0)


@partial(jax.jit, static_argnums=0)
def jit_evaluate(fun: Callable, *args):
    """
    call `fun(*args)` inside of a `jax.jit` frame.

    Args:
        fun: the hashable callable to be evaluated.
        args: the arguments to the function.
    """
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
    """The sampler used to sample the Hilbert space."""
    sampler_state: SamplerState
    """The current state of the sampler."""
    _previous_sampler_state: SamplerState = None
    """The sampler state before the last sampling has been effected.

    This field is used so that we don't need to serialize the current samples
    but we can always regenerate them.
    """

    _chain_length: int = 0
    """Length of the Markov chain used for sampling configurations."""
    _n_discard_per_chain: int = 0
    """Number of samples discarded at the beginning of every Markov chain."""

    _samples: Optional[jnp.ndarray] = None
    """Cached samples obtained with the last sampling."""

    _init_fun: Callable = None
    """The function used to initialise the parameters and model_state."""
    _apply_fun: Callable = None
    """The function used to evaluate the model."""

    _chunk_size: Optional[int] = None

    __module__ = "netket.vqs"

    def __init__(
        self,
        sampler: Sampler,
        model=None,
        *,
        n_samples: Optional[int] = None,
        n_samples_per_rank: Optional[int] = None,
        n_discard_per_chain: Optional[int] = None,
        chunk_size: Optional[int] = None,
        variables: Optional[PyTree] = None,
        init_fun: Optional[NNInitFunc] = None,
        apply_fun: Optional[Callable] = None,
        sample_fun: Optional[Callable] = None,
        seed: Optional[SeedT] = None,
        sampler_seed: Optional[SeedT] = None,
        mutable: CollectionFilter = False,
        training_kwargs: dict = {},
    ):
        """
        Constructs the MCState.

        Args:
            sampler: The sampler
            model: (Optional) The neural quantum state ansatz, encoded into a model.
                This should be a :class:`flax.linen.Module` instance, or any other supported
                neural network framework. If not provided, you must specify init_fun and apply_fun.
            n_samples: the total number of samples across chains and processes when sampling (default=1000).
            n_samples_per_rank: the total number of samples across chains on one process when sampling. Cannot be
                specified together with n_samples (default=None).
            n_discard_per_chain: number of discarded samples at the beginning of each monte-carlo chain (default=0 for exact sampler,
                and n_samples/10 for approximate sampler).
            parameters: Optional PyTree of weights from which to start.
            seed: rng seed used to generate a set of parameters (only if parameters is not passed). Defaults to a random one.
            sampler_seed: rng seed used to initialise the sampler. Defaults to a random one.
            mutable: Name or list of names of mutable arguments. Use it to specify if the model has a state that can change
                during evaluation, but that should not be optimised. See also :meth:`flax.linen.Module.apply` documentation
                (default=False)
            init_fun: Function of the signature f(model, shape, rng_key, dtype) -> Optional_state, parameters used to
                initialise the parameters. Defaults to the standard flax initialiser. Only specify if your network has
                a non-standard init method.
            variables: Optional initial value for the variables (parameters and model state) of the model.
            apply_fun: Function of the signature f(model, variables, σ) that should evaluate the model. Defaults to
                `model.apply(variables, σ)`. specify only if your network has a non-standard apply method.
            sample_fun: Optional function used to sample the state, if it is not the same as `apply_fun`.
            training_kwargs: a dict containing the optional keyword arguments to be passed to the apply_fun during training.
                Useful for example when you have a batchnorm layer that constructs the average/mean only during training.
            chunk_size: (Defaults to `None`) If specified, calculations are split into chunks where the neural network
                is evaluated at most on :code:`chunk_size` samples at once. This does not change the mathematical results,
                but will trade a higher computational cost for lower memory cost.
        """
        super().__init__(sampler.hilbert)

        # Init type 1: pass in a model
        if model is not None:
            # extract init and apply functions
            # Wrap it in an HashablePartial because if two instances of the same model are provided,
            # model.apply and model2.apply will be different methods forcing recompilation, but
            # model and model2 will have the same hash.
            _, model = maybe_wrap_module(model)

            self._model = model

            self._init_fun = nkjax.HashablePartial(
                lambda model, *args, **kwargs: model.init(*args, **kwargs), model
            )
            self._apply_fun = wrap_to_support_scalar(
                nkjax.HashablePartial(
                    lambda model, pars, x, **kwargs: model.apply(pars, x, **kwargs),
                    model,
                )
            )

        elif apply_fun is not None:
            self._apply_fun = wrap_to_support_scalar(apply_fun)

            if init_fun is not None:
                self._init_fun = init_fun
            elif variables is None:
                raise ValueError(
                    "If you don't provide variables, you must pass a valid init_fun."
                )

            self._model = wrap_afun(apply_fun)

        else:
            raise ValueError(
                "Must either pass the model or apply_fun, otherwise how do you think we"
                "gonna evaluate the model?"
            )

        # default argument for n_samples/n_samples_per_rank
        if n_samples is None and n_samples_per_rank is None:
            # get the first multiple of sampler.n_chains above 1000 to avoid
            # printing a warning on construction
            n_samples = int(np.ceil(1000 / sampler.n_chains) * sampler.n_chains)
        elif n_samples is not None and n_samples_per_rank is not None:
            raise ValueError(
                "Only one argument between `n_samples` and `n_samples_per_rank`"
                "can be specified at the same time."
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

        if n_samples is not None:
            self.n_samples = n_samples
        else:
            self.n_samples_per_rank = n_samples_per_rank

        self.n_discard_per_chain = n_discard_per_chain

        self.chunk_size = chunk_size

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

        variables = jit_evaluate(self._init_fun, {"params": key}, dummy_input)
        self.variables = variables

    @property
    def model(self) -> Optional[Any]:
        """Returns the model definition of this variational state.

        This field is optional, and is set to `None` if the variational state has
        been initialized using a custom function.
        """
        return self._model

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

        # Save the old `n_samples` before the new `sampler` is set.
        # `_chain_length == 0` means that this `MCState` is being constructed.
        if self._chain_length > 0:
            n_samples_old = self.n_samples

        self._sampler = sampler
        self.sampler_state = self.sampler.init_state(
            self.model, self.variables, seed=self._sampler_seed
        )
        self._sampler_state_previous = self.sampler_state

        # Update `n_samples`, `n_samples_per_rank`, and `chain_length` according
        # to the new `sampler.n_chains`.
        # If `n_samples` is divisible by the new `sampler.n_chains`, it will be
        # unchanged; otherwise it will be rounded up.
        # If the new `n_samples_per_rank` is not divisible by `chunk_size`, a
        # `ValueError` will be raised.
        # `_chain_length == 0` means that this `MCState` is being constructed.
        if self._chain_length > 0:
            self.n_samples = n_samples_old

        self.reset()

    @property
    def n_samples(self) -> int:
        """The total number of samples generated at every sampling step."""
        return self.chain_length * self.sampler.n_chains

    @n_samples.setter
    def n_samples(self, n_samples: int):
        chain_length = compute_chain_length(self.sampler.n_chains, n_samples)
        self.chain_length = chain_length

    @property
    def n_samples_per_rank(self) -> int:
        """The number of samples generated on one MPI rank at every sampling step."""
        return self.chain_length * self.sampler.n_chains_per_rank

    @n_samples_per_rank.setter
    def n_samples_per_rank(self, n_samples_per_rank: int):
        self.n_samples = n_samples_per_rank * mpi.n_nodes

    @property
    def chain_length(self) -> int:
        """
        Length of the markov chain used for sampling configurations.

        If running under MPI, the total samples will be n_nodes * chain_length * n_batches.
        """
        return self._chain_length

    @chain_length.setter
    def chain_length(self, chain_length: int):
        if chain_length <= 0:
            raise ValueError(f"Invalid chain length: chain_length={chain_length}")

        n_samples = chain_length * self.sampler.n_chains
        check_chunk_size(n_samples, self.chunk_size)

        self._chain_length = chain_length
        self.reset()

    @property
    def n_discard_per_chain(self) -> int:
        """
        Number of discarded samples at the beginning of the markov chain.
        """
        return self._n_discard_per_chain

    @n_discard_per_chain.setter
    def n_discard_per_chain(self, n_discard_per_chain: Optional[int]):
        if n_discard_per_chain is not None and n_discard_per_chain < 0:
            raise ValueError(
                "Invalid number of discarded samples: n_discard_per_chain={}".format(
                    n_discard_per_chain
                )
            )

        # don't discard if the sampler is exact
        if self.sampler.is_exact:
            if n_discard_per_chain is not None and n_discard_per_chain > 0:
                warnings.warn(
                    "An exact sampler does not need to discard samples. Setting n_discard_per_chain to 0.",
                    stacklevel=2,
                )
            n_discard_per_chain = 0

        self._n_discard_per_chain = (
            int(n_discard_per_chain)
            if n_discard_per_chain is not None
            else self.n_samples // 10
        )

    @property
    def chunk_size(self) -> int:
        """
        Suggested *maximum size* of the chunks used in forward and backward evaluations
        of the Neural Network model.

        If your inputs are smaller than the chunk size this setting is ignored.

        This can be used to lower the memory required to run a computation with a very
        high number of samples or on a very large lattice. Notice that inputs and
        outputs must still fit in memory, but the intermediate computations will now
        require less memory.

        This option comes at an increased computational cost. While this cost should
        be negligible for large-enough chunk sizes, don't use it unless you are memory
        bound!

        This option is an hint: only some operations support chunking. If you perform
        an operation that is not implemented with chunking support, it will fall back
        to no chunking. To check if this happened, set the environment variable
        `NETKET_DEBUG=1`.
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size: Optional[int]):
        # disable chunks if it is None
        if chunk_size is None:
            self._chunk_size = None
            return

        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer. ")

        if not _is_power_of_two(chunk_size):
            warnings.warn(
                "For performance reasons, we suggest to use a power-of-two chunk size.",
                stacklevel=2,
            )

        check_chunk_size(self.n_samples, chunk_size)

        self._chunk_size = chunk_size

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
        n_discard_per_chain: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Sample a certain number of configurations.

        If one among chain_length or n_samples is defined, that number of samples
        are generated. Otherwise the value set internally is used.

        Args:
            chain_length: The length of the markov chains.
            n_samples: The total number of samples across all MPI ranks.
            n_discard_per_chain: Number of discarded samples at the beginning of the markov chain.
        """

        if n_samples is None and chain_length is None:
            chain_length = self.chain_length
        else:
            if chain_length is None:
                chain_length = compute_chain_length(self.sampler.n_chains, n_samples)

            if self.chunk_size is not None:
                check_chunk_size(chain_length * self.sampler.n_chains, self.chunk_size)

        if n_discard_per_chain is None:
            n_discard_per_chain = self.n_discard_per_chain

        # Store the previous sampler state, for serialization purposes
        self._sampler_state_previous = self.sampler_state

        self.sampler_state = self.sampler.reset(
            self.model, self.variables, self.sampler_state
        )

        if self.n_discard_per_chain > 0:
            _, self.sampler_state = self.sampler.sample(
                self.model,
                self.variables,
                state=self.sampler_state,
                chain_length=n_discard_per_chain,
            )

        self._samples, self.sampler_state = self.sampler.sample(
            self.model,
            self.variables,
            state=self.sampler_state,
            chain_length=chain_length,
        )
        return self._samples

    @property
    def samples(self) -> jnp.ndarray:
        """
        Returns the set of cached samples.

        The samples returned are guaranteed valid for the current state of
        the variational state. If no cached parameters are available, then
        they are sampled first and then cached.

        To obtain a new set of samples either use
        :meth:`~MCState.reset` or :meth:`~MCState.sample`.
        """
        if self._samples is None:
            self.sample()
        return self._samples

    def log_value(self, σ: jnp.ndarray) -> jnp.ndarray:
        r"""
        Evaluate the variational state for a batch of states and returns
        the logarithm of the amplitude of the quantum state.

        For pure states,
        this is :math:`\log(\langle\sigma|\psi\rangle)`, whereas for mixed states
        this is :math:`\log(\langle\sigma_r|\rho|\sigma_c\rangle)`, where
        :math:`\psi` and :math:`\rho` are respectively a pure state
        (wavefunction) and a mixed state (density matrix).
        For the density matrix, the left and right-acting states (row and column)
        are obtained as :code:`σr=σ[::,0:N]` and :code:`σc=σ[::,N:]`.

        Given a batch of inputs :code:`(Nb, N)`, returns a batch of outputs
        :code:`(Nb,)`.
        """
        return jit_evaluate(self._apply_fun, self.variables, σ)

    def local_estimators(
        self, op: AbstractOperator, *, chunk_size: Optional[int] = None
    ):
        r"""
        Compute the local estimators for the operator :code:`op` (also known as local energies
        when :code:`op` is the Hamiltonian) at the current configuration samples :code:`self.samples`.

        .. math::

            O_\mathrm{loc}(s) = \frac{\langle s | \mathtt{op} | \psi \rangle}{\langle s | \psi \rangle}

        .. warning::

            The samples differ between MPI processes, so returned the local estimators will
            also take different values on each process. To compute sample averages and similar
            quantities, you will need to perform explicit operations over all MPI ranks.
            (Use functions like :code:`self.expect` to get process-independent quantities without
            manual reductions.)

        Args:
            op: The operator.
            chunk_size: Suggested maximum size of the chunks used in forward and backward evaluations
                of the model. (Default: :code:`self.chunk_size`)
        """
        return local_estimators(self, op, chunk_size=chunk_size)

    # override to use chunks
    def expect(self, O: AbstractOperator) -> Stats:
        r"""Estimates the quantum expectation value for a given operator
        :math:`O` or generic observable.
        In the case of a pure state :math:`\psi` and an operator, this is
        :math:`\langle O\rangle= \langle \Psi|O|\Psi\rangle/\langle\Psi|\Psi\rangle`
        otherwise for a  mixed state :math:`\rho`, this is
        :math:`\langle O\rangle= \textrm{Tr}[\rho \hat{O}]/\textrm{Tr}[\rho]`.

        Args:
            O: the operator or observable for which to compute the expectation
                value.

        Returns:
            An estimation of the quantum expectation value
            :math:`\langle O\rangle`.
        """
        return expect(self, O, self.chunk_size)

    # override to use chunks
    def expect_and_grad(
        self,
        O: AbstractOperator,
        *,
        mutable: Optional[CollectionFilter] = None,
        **kwargs,
    ) -> tuple[Stats, PyTree]:
        r"""Estimates the quantum expectation value and its gradient
        for a given operator :math:`O`.

        Args:
            O: The operator :math:`O` for which expectation value and
                gradient are computed.
            mutable: Can be bool, str, or list. Specifies which collections in the
                     model_state should be treated as  mutable: bool: all/no collections
                     are mutable. str: The name of a single mutable  collection. list: A
                     list of names of mutable collections. This is used to mutate the state
                     of the model while you train it (for example to implement BatchNorm. Consult
                     `Flax's Module.apply documentation <https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module.apply>`_
                     for a more in-depth explanation).
            use_covariance: whether to use the covariance formula, usually reserved for
                hermitian operators,
                :math:`\textrm{Cov}[\partial\log\psi, O_{\textrm{loc}}\rangle]`

        Returns:
            An estimate of the quantum expectation value <O>.
            An estimate of the gradient of the quantum expectation value <O>.
        """
        if mutable is None:
            mutable = self.mutable

        return expect_and_grad(
            self,
            O,
            self.chunk_size,
            mutable=mutable,
            **kwargs,
        )

    # override to use chunks
    def expect_and_forces(
        self,
        O: AbstractOperator,
        *,
        mutable: Optional[CollectionFilter] = None,
    ) -> tuple[Stats, PyTree]:
        r"""Estimates the quantum expectation value and the corresponding force
        vector for a given operator O.

        The force vector :math:`F_j` is defined as the covariance of log-derivative
        of the trial wave function and the local estimators of the operator. For complex
        holomorphic states, this is equivalent to the expectation gradient
        :math:`\frac{\partial\langle O\rangle}{\partial(\theta_j)^\star} = F_j`. For real-parameter states,
        the gradient is given by :math:`\frac{\partial\partial_j\langle O\rangle}{\partial\partial_j\theta_j} = 2 \textrm{Re}[F_j]`.

        Args:
            O: The operator O for which expectation value and force are computed.
            mutable: Can be bool, str, or list. Specifies which collections in
                the model_state should be treated as  mutable: bool: all/no
                collections are mutable. str: The name of a single mutable
                collection. list: A list of names of mutable collections. This is
                used to mutate the state of the model while you train it (for
                example to implement BatchNorm. Consult
                `Flax's Module.apply documentation <https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module.apply>`_
                for a more in-depth explanation).

        Returns:
            An estimate of the quantum expectation value <O>.
            An estimate of the force vector
            :math:`F_j = \textrm{Cov}[\partial_j\log\psi, O_{\textrm{loc}}]`.
        """
        if isinstance(O, Squared):
            raise NotImplementedError(
                "expect_and_forces not yet implemented for `Squared`"
            )

        if mutable is None:
            mutable = self.mutable

        return expect_and_forces(self, O, self.chunk_size, mutable=mutable)

    def quantum_geometric_tensor(
        self, qgt_T: Optional[LinearOperator] = None
    ) -> LinearOperator:
        r"""Computes an estimate of the quantum geometric tensor G_ij.
        This function returns a linear operator that can be used to apply G_ij to a given vector
        or can be converted to a full matrix.

        Args:
            qgt_T: the optional type of the quantum geometric tensor. By default it's automatically selected.


        Returns:
            nk.optimizer.LinearOperator: A linear operator representing the quantum geometric tensor.
        """
        if qgt_T is None:
            qgt_T = QGTAuto()

        return qgt_T(self)

    def to_array(self, normalize: bool = True) -> jnp.ndarray:
        return nn.to_array(
            self.hilbert,
            self._apply_fun,
            self.variables,
            normalize=normalize,
            chunk_size=self.chunk_size,
        )

    def __repr__(self):
        return (
            "MCState("
            + f"\n  hilbert = {self.hilbert},"
            + f"\n  sampler = {self.sampler},"
            + f"\n  n_samples = {self.n_samples},"
            + f"\n  n_discard_per_chain = {self.n_discard_per_chain},"
            + f"\n  sampler_state = {self.sampler_state},"
            + f"\n  n_parameters = {self.n_parameters})"
        )

    def __str__(self):
        return (
            "MCState("
            + f"hilbert = {self.hilbert}, "
            + f"sampler = {self.sampler}, "
            + f"n_samples = {self.n_samples})"
        )


@partial(jax.jit, static_argnames=("kernel", "apply_fun", "shape"))
def _local_estimators_kernel(kernel, apply_fun, shape, variables, samples, extra_args):
    O_loc = kernel(apply_fun, variables, samples, extra_args)

    return O_loc.reshape(shape)


def local_estimators(
    state: MCState, op: AbstractOperator, *, chunk_size: Optional[int]
):
    s, extra_args = get_local_kernel_arguments(state, op)

    shape = s.shape
    if jnp.ndim(s) != 2:
        s = s.reshape((-1, shape[-1]))

    if chunk_size is None:
        chunk_size = state.chunk_size  # state.chunk_size can still be None

    if chunk_size is None:
        kernel = get_local_kernel(state, op)
    else:
        kernel = nkjax.HashablePartial(
            get_local_kernel(state, op, chunk_size), chunk_size=chunk_size
        )

    return _local_estimators_kernel(
        kernel, state._apply_fun, shape[:-1], state.variables, s, extra_args
    )


# serialization
def serialize_MCState(vstate):
    state_dict = {
        "variables": serialization.to_state_dict(vstate.variables),
        "sampler_state": serialization.to_state_dict(vstate._sampler_state_previous),
        "n_samples": vstate.n_samples,
        "n_discard_per_chain": vstate.n_discard_per_chain,
        "chunk_size": vstate.chunk_size,
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
    new_vstate.n_discard_per_chain = state_dict["n_discard_per_chain"]
    new_vstate.chunk_size = state_dict["chunk_size"]

    return new_vstate


serialization.register_serialization_state(
    MCState,
    serialize_MCState,
    deserialize_MCState,
)
