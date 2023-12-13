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

import jax
from jax import numpy as jnp

import flax
from flax import serialization
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket import nn as nknn
from netket.hilbert import AbstractHilbert
from netket.utils import maybe_wrap_module, wrap_afun, wrap_to_support_scalar
from netket.utils.types import PyTree, SeedT, NNInitFunc
from netket.optimizer import LinearOperator
from netket.optimizer.qgt import QGTAuto

from ..base import VariationalState
from ..mc.mc_state.state import check_chunk_size, _is_power_of_two


@partial(jax.jit, static_argnums=0)
def jit_evaluate(fun: Callable, *args):
    """
    call `fun(*args)` inside of a `jax.jit` frame.

    Args:
        fun: the hashable callable to be evaluated.
        args: the arguments to the function.
    """
    return fun(*args)


@jax.jit
def _array_to_pdf(v):
    return jnp.abs(v) ** 2


class FullSumState(VariationalState):
    """Variational State for a variational quantum state computed on the whole
    Hilbert space without Monte Carlo sampling by summing over the whole Hilbert
    space.

    Expectation values and gradients are deterministic.
    The only non-deterministic part is due to the initialization seed used to generate
    the parameters.
    """

    model_state: Optional[PyTree]
    """An Optional PyTree encoding a mutable state of the model that is not trained."""

    _init_fun: Callable = None
    """The function used to initialise the parameters and model_state"""
    _apply_fun: Callable = None
    """The function used to evaluate the model"""

    _chunk_size: Optional[int] = None

    def __init__(
        self,
        hilbert: AbstractHilbert,
        model=None,
        *,
        chunk_size: Optional[int] = None,
        variables: Optional[PyTree] = None,
        init_fun: NNInitFunc = None,
        apply_fun: Optional[Callable] = None,
        seed: Optional[SeedT] = None,
        mutable: CollectionFilter = False,
        training_kwargs: dict = {},
        dtype=float,
    ):
        """
        Constructs the FullSumState.

        Args:
            hilbert: The Hilbert space
            model: (Optional) The model. If not provided, you must provide init_fun and apply_fun.
            variables: Optional dictionary for the initial values for the variables (parameters and model state) of the model.
            seed: rng seed used to generate a set of parameters (only if parameters is not passed). Defaults to a random one.
            mutable: Name or list of names of mutable arguments. Use it to specify if the model has a state that can change
                during evaluation, but that should not be optimised. See also :meth:`flax.linen.Module.apply` documentation
                (default=False)
            init_fun: Function of the signature f(model, shape, rng_key, dtype) -> Optional_state, parameters used to
                initialise the parameters. Defaults to the standard flax initialiser. Only specify if your network has
                a non-standard init method.
            apply_fun: Function of the signature f(model, variables, σ) that should evaluate the model. Defaults to
                `model.apply(variables, σ)`. specify only if your network has a non-standard apply method.
            training_kwargs: a dict containing the optional keyword arguments to be passed to the apply_fun during training.
                Useful for example when you have a batchnorm layer that constructs the average/mean only during training.
            chunk_size: (Defaults to `None`) If specified, calculations are split into chunks where the neural network
                is evaluated at most on :code:`chunk_size` samples at once. This does not change the mathematical results,
                but will trade a higher computational cost for lower memory cost.
        """
        super().__init__(hilbert)

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
                    lambda model, *args, **kwargs: model.apply(*args, **kwargs), model
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

        self.mutable = mutable
        self.training_kwargs = flax.core.freeze(training_kwargs)

        if variables is not None:
            self.variables = variables
        else:
            self.init(seed, dtype=dtype)

        self._states = None
        """
        Caches the output of `self._all_states()`.
        """

        self._array = None
        """
        Caches the output of `self.to_array()`.
        """

        self._pdf = None
        """
        Caches the output of `self.probability_distribution()`.
        """

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
            dtype = float

        key = nkjax.PRNGKey(seed)

        dummy_input = jnp.zeros((1, self.hilbert.size), dtype=dtype)

        variables = jit_evaluate(self._init_fun, {"params": key}, dummy_input)
        self.variables = variables

    @property
    def chunk_size(self) -> int:
        """
        Suggested *maximum size* of the chunks used in forward and backward evaluations
        of the Neural Network model. If your inputs are smaller than the chunk size
        this setting is ignored.

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
                "For performance reasons, we suggest to use a power-of-two chunk size."
            )

        # TODO MPI aware check for valid size
        check_chunk_size(self.hilbert.n_states, chunk_size)

        self._chunk_size = chunk_size

    def reset(self):
        """
        Resets the sampled states. This method is called automatically every time
        that the parameters/state is updated.
        """
        self._array = None
        self._pdf = None

    @property
    def model(self) -> Optional[Any]:
        """Returns the model definition of this variational state.

        This field is optional, and is set to `None` if the variational state has
        been initialized using a custom function.
        """
        return self._model

    def log_value(self, σ: jnp.ndarray) -> jnp.ndarray:
        r"""
        Evaluate the variational state for a batch of states and returns
        the logarithm of the amplitude of the quantum state.

        For pure states, this is :math:`\log(\langle\sigma|\psi\rangle)`,
        whereas for mixed states
        this is :math:`\log(\langle\sigma_r|\rho|\sigma_c\rangle)`, where
        :math:`\psi` and :math:`\rho` are respectively a pure state
        (wavefunction) and a mixed state (density matrix).
        For the density matrix, the left and right-acting states (row and column)
        are obtained as :code:`σr=σ[::,0:N]` and :code:`σc=σ[::,N:]`.

        Given a batch of inputs :code:`(Nb, N)`, returns a batch of outputs
        :code:`(Nb,)`.
        """
        return jit_evaluate(self._apply_fun, self.variables, σ)

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

    def to_array(self, normalize: bool = True, allgather: bool = True) -> jnp.ndarray:
        if self._array is None and normalize:
            self._array = nknn.to_array(
                self.hilbert,
                self._apply_fun,
                self.variables,
                normalize=normalize,
                allgather=allgather,
                chunk_size=self.chunk_size,
            )

        if normalize:
            arr = self._array
        else:
            arr = nknn.to_array(
                self.hilbert,
                self._apply_fun,
                self.variables,
                normalize=normalize,
                allgather=allgather,
                chunk_size=self.chunk_size,
            )

        return arr

    def probability_distribution(self):
        if self._pdf is None:
            self._pdf = _array_to_pdf(self.to_array())

        return self._pdf

    # cached computations
    @property
    def _all_states(self):
        if self._states is None:
            self._states = self.hilbert.all_states()
        return self._states

    def __repr__(self):
        return (
            "FullSumState("
            + f"\n  hilbert = {self.hilbert},"
            + f"\n  n_parameters = {self.n_parameters})"
        )

    def __str__(self):
        return "FullSumState(" + f"hilbert = {self.hilbert}, "


# serialization


def serialize_FullSumState(vstate):
    state_dict = {
        "variables": serialization.to_state_dict(vstate.variables),
    }
    return state_dict


def deserialize_FullSumState(vstate, state_dict):
    import copy

    new_vstate = copy.copy(vstate)
    new_vstate.reset()

    new_vstate.variables = serialization.from_state_dict(
        vstate.variables, state_dict["variables"]
    )
    return new_vstate


serialization.register_serialization_state(
    FullSumState,
    serialize_FullSumState,
    deserialize_FullSumState,
)
