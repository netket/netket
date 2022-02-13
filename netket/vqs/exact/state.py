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

from functools import partial
from typing import Any, Callable, Dict, Optional

import jax
from jax import numpy as jnp

import flax
from flax import serialization

from netket import jax as nkjax
from netket import nn
from netket.hilbert import AbstractHilbert
from netket.utils import maybe_wrap_module, wrap_afun, wrap_to_support_scalar
from netket.utils.types import PyTree, SeedT, NNInitFunc
from netket.optimizer import LinearOperator
from netket.optimizer.qgt import QGTAuto

from ..base import VariationalState


@partial(jax.jit, static_argnums=0)
def jit_evaluate(fun: Callable, *args):
    """
    call `fun(*args)` inside of a `jax.jit` frame.

    Args:
        fun: the hashable callable to be evaluated.
        args: the arguments to the function.
    """
    return fun(*args)


class ExactState(VariationalState):
    """Variational State for a variational quantum state computed on the whole
    Hilbert space without Monte Carlo sampling.

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

    def __init__(
        self,
        hilbert: AbstractHilbert,
        model=None,
        *,
        variables: Optional[PyTree] = None,
        init_fun: NNInitFunc = None,
        apply_fun: Callable = None,
        seed: Optional[SeedT] = None,
        mutable: bool = False,
        training_kwargs: Dict = {},
        dtype=float,
    ):
        """
        Constructs the ExactState.

        Args:
            hilbert: The Hilbert space
            model: (Optional) The model. If not provided, you must provide init_fun and apply_fun.
            parameters: Optional PyTree of weights from which to start.
            seed: rng seed used to generate a set of parameters (only if parameters is not passed). Defaults to a random one.
            mutable: Dict specifing mutable arguments. Use it to specify if the model has a state that can change
                during evaluation, but that should not be optimised. See also flax.linen.module.apply documentation
                (default=False)
            init_fun: Function of the signature f(model, shape, rng_key, dtype) -> Optional_state, parameters used to
                initialise the parameters. Defaults to the standard flax initialiser. Only specify if your network has
                a non-standard init method.
            variables: Optional initial value for the variables (parameters and model state) of the model.
            apply_fun: Function of the signature f(model, variables, σ) that should evaluate the model. Defafults to
                `model.apply(variables, σ)`. specify only if your network has a non-standard apply method.
            training_kwargs: a dict containing the optionaal keyword arguments to be passed to the apply_fun during training.
                Useful for example when you have a batchnorm layer that constructs the average/mean only during training.
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
        return jit_evaluate(self._apply_fun, self.variables, σ)

    def quantum_geometric_tensor(
        self, qgt_T: LinearOperator = QGTAuto()
    ) -> LinearOperator:
        r"""Computes an estimate of the quantum geometric tensor G_ij.
        This function returns a linear operator that can be used to apply G_ij to a given vector
        or can be converted to a full matrix.

        Args:
            qgt_T: the optional type of the quantum geometric tensor. By default it's automatically selected.


        Returns:
            nk.optimizer.LinearOperator: A linear operator representing the quantum geometric tensor.
        """
        return qgt_T(self)

    def to_array(self, normalize: bool = True, allgather: bool = True) -> jnp.ndarray:

        if self._array is None and normalize:
            self._array = nn.to_array(
                self.hilbert,
                self._apply_fun,
                self.variables,
                normalize=normalize,
                allgather=allgather,
            )

        if normalize:
            arr = self._array
        else:
            arr = nn.to_array(
                self.hilbert,
                self._apply_fun,
                self.variables,
                normalize=normalize,
                allgather=allgather,
            )

        return arr

    def probability_distribution(self):
        if self._pdf is None:
            self._pdf = jnp.abs(self.to_array()) ** 2

        return self._pdf

    # cached computations
    @property
    def _all_states(self):
        if self._states is None:
            self._states = self.hilbert.all_states()
        return self._states

    def __repr__(self):
        return (
            "ExactState("
            + "\n  hilbert = {},".format(self.hilbert)
            + "\n  n_parameters = {})".format(self.n_parameters)
        )

    def __str__(self):
        return "ExactState(" + "hilbert = {}, ".format(self.hilbert)


# serialization


def serialize_ExactState(vstate):
    state_dict = {
        "variables": serialization.to_state_dict(vstate.variables),
    }
    return state_dict


def deserialize_ExactState(vstate, state_dict):
    import copy

    new_vstate = copy.copy(vstate)
    new_vstate.reset()

    new_vstate.variables = serialization.from_state_dict(
        vstate.variables, state_dict["variables"]
    )
    return new_vstate


serialization.register_serialization_state(
    ExactState,
    serialize_ExactState,
    deserialize_ExactState,
)
