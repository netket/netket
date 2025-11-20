# Copyright 2025 The NetKet Authors - All rights reserved.
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
from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp

from flax import serialization

from netket import jax as nkjax
from netket.hilbert import SpinOrbitalFermions
from netket.models import Slater2nd
from netket.sampler import Sampler
from netket.utils import (
    model_frameworks,
    wrap_to_support_scalar,
    _serialization as serialization_utils,
)
from netket.utils.types import PyTree, SeedT, DType

from netket.vqs.base import VariationalState
from netket.vqs.mc import MCState
from netket.vqs.full_summ import FullSumState
from netket import nn as nknn


@partial(jax.jit, static_argnums=0)
def jit_evaluate(fun: Callable, *args):
    """
    call `fun(*args)` inside of a `jax.jit` frame.

    Args:
        fun: the hashable callable to be evaluated.
        args: the arguments to the function.
    """
    return fun(*args)


@partial(jax.jit, static_argnames=("model", "method"))
def jit_apply(model, *args, method):
    return model.apply(*args, method=method)


class DeterminantVariationalState(VariationalState):
    r"""Variational State for fermionic mean-field states (Hartree-Fock ansatz).

    This state represents a Slater determinant wavefunction using a
    :class:`~netket.models.Slater2nd` model and computes expectation values
    analytically using Wick's theorem rather than Monte Carlo sampling, which is
    efficient.

    The state internally constructs and uses a :class:`~netket.models.Slater2nd` model,
    which supports three types of Hartree-Fock ansätze:

    - **Generalized Hartree-Fock (GHF)**: ``generalized=True``
      Uses :math:`n_{\mathrm{M}} \times n_{\mathrm{f}}` parameters where
      :math:`n_{\mathrm{M}}` is the number of fermionic modes and :math:`n_{\mathrm{f}}`
      is the number of fermions. This is the most flexible ansatz but has the most parameters.

    - **Unrestricted Hartree-Fock (UHF)**: ``generalized=False, restricted=False``
      Uses :math:`n_{\mathrm{S}} \times n_{\mathrm{L}} \times n_{\textrm{f, s}}` parameters where
      :math:`n_{\mathrm{S}}` is the number of spin states, :math:`n_{\mathrm{L}}` is the number
      of spatial orbitals, and :math:`n_{\textrm{f, s}}` is the number of fermions per spin.
      Allows different spatial orbitals for different spins (can describe magnetic states).

    - **Restricted Hartree-Fock (RHF)**: ``generalized=False, restricted=True``
      Uses :math:`n_{\mathrm{L}} \times n_{\textrm{f, s}}` parameters.
      All spins share the same spatial orbitals (appropriate for non-magnetic ground states).

    See :class:`~netket.models.Slater2nd` for more details on the different Hartree-Fock types
    and their mathematical formulation.

    """

    _model_framework: model_frameworks.ModuleFramework | None = None
    """The model framework used to define the model."""

    _model: Any
    """The linen-compatible model definition."""

    _init_fun: Callable | None = None
    """The function used to initialise the parameters and model_state"""

    _apply_fun: Callable
    """The function used to evaluate the model"""

    _rdm: jax.Array | None = None
    """Cached one-body reduced density matrix"""

    def __init__(
        self,
        hilbert: SpinOrbitalFermions,
        *,
        generalized: bool = False,
        restricted: bool = True,
        param_dtype: DType = float,
        variables: PyTree | None = None,
        seed: SeedT | None = None,
    ):
        r"""
        Constructs the DeterminantVariationalState.

        This constructor automatically creates an internal :class:`~netket.models.Slater2nd`
        model with the specified Hartree-Fock type.

        .. warning::

            If unspecified the parameters will be completely random, which is
            a very bad initialization for a Slater Determinant. Consider
            initializing using something more reasonable, like the k-space
            tight-binding orbitals or something similar.

        Args:
            hilbert: The Hilbert space. Must be a :class:`~netket.hilbert.SpinOrbitalFermions`
                with fixed particle number (``n_fermions`` must not be None).
            generalized: If True, uses Generalized Hartree-Fock (GHF) with
                :math:`n_{\mathrm{M}} \times n_{\mathrm{f}}` parameters.
                If False, the type depends on the ``restricted`` parameter.
                See :class:`~netket.models.Slater2nd` for details. Default: False.
            restricted: Only used when ``generalized=False``. If True, uses Restricted
                Hartree-Fock (RHF) with :math:`n_{\mathrm{L}} \times n_{\textrm{f, s}}`
                parameters where all spins share spatial orbitals. If False, uses
                Unrestricted Hartree-Fock (UHF) with separate orbitals per spin.
                See :class:`~netket.models.Slater2nd` for details. Default: True.
            param_dtype: The dtype of the variational parameters (float or complex).
                Default: float.
            variables: Optional dictionary for the initial values for the variables
                (parameters and model state) of the model. If not provided, parameters
                are randomly initialized using the seed.
            seed: Random seed used to generate initial parameters (only used if
                ``variables`` is not provided). Defaults to a random seed.

        Example:
            >>> import netket as nk
            >>> # Create a spin-1/2 fermionic Hilbert space
            >>> hi = nk.hilbert.SpinOrbitalFermions(4, s=1/2, n_fermions_per_spin=(2, 2))
            >>> # Create a Restricted Hartree-Fock state (non-magnetic)
            >>> vstate = nk.experimental.vqs.DeterminantVariationalState(
            ...     hi, generalized=False, restricted=True, seed=42
            ... )
            >>> print(vstate.n_parameters)  # Number of variational parameters
        """
        # Validate hilbert space
        if not isinstance(hilbert, SpinOrbitalFermions):
            raise TypeError(
                "DeterminantVariationalState requires a SpinOrbitalFermions hilbert space, "
                f"but got {type(hilbert)}"
            )

        if hilbert.n_fermions is None:
            raise ValueError(
                "DeterminantVariationalState requires a hilbert space with a fixed number "
                "of fermions (n_fermions must not be None)."
            )

        super().__init__(hilbert)

        # Extract init and apply functions
        model = Slater2nd(
            hilbert,
            generalized=generalized,
            restricted=restricted,
            param_dtype=param_dtype,
        )
        self._model_framework = model_frameworks.identify_framework(model)
        _maybe_unwrapped_variables, model = self._model_framework.wrap(model)

        if variables is None:
            if _maybe_unwrapped_variables is not None:
                variables = _maybe_unwrapped_variables

        self._model = model

        self._init_fun = nkjax.HashablePartial(
            lambda model, *args, **kwargs: model.init(*args, **kwargs), model
        )
        self._apply_fun = wrap_to_support_scalar(
            nkjax.HashablePartial(
                lambda model, *args, **kwargs: model.apply(*args, **kwargs), model
            )
        )

        if variables is not None:
            self.variables = variables
        else:
            self.init(seed, dtype=float)

        self._rdm = None
        self.mutable = False

    @property
    def model(self) -> Any | None:
        """Returns the model definition of this variational state.

        This field is optional, and is set to `None` if the variational state has
        been initialized using a custom function.
        """
        if self._model_framework is not None:
            return self._model_framework.unwrap(self._model, self.variables)
        return self._model

    def init(self, seed=None, dtype=None):
        """
        Initialises the variational parameters of the variational state.
        """
        if self._init_fun is None:
            raise RuntimeError(
                "Cannot initialise the parameters of this state "
                "because you did not supply a valid init_function."
            )

        if dtype is None:
            dtype = float

        key = nkjax.PRNGKey(seed)
        dummy_input = self.hilbert.random_state(key, 1, dtype=dtype)

        variables = jit_evaluate(self._init_fun, {"params": key}, dummy_input)
        self.variables = variables

    def reset(self):
        """
        Resets the cached density matrix. Called automatically when
        parameters/state is updated.
        """
        self._rdm = None

    def log_value(self, σ: jnp.ndarray) -> jnp.ndarray:
        return jit_evaluate(self._apply_fun, self.variables, σ)

    @property
    def rdm(self) -> jax.Array:
        """
        The one-body reduced density matrix of the mean-field state.

        This is computed from the Slater determinant parameters and cached
        until parameters are updated.

        Returns:
            The density matrix of shape (n_modes, n_modes) where n_modes is
            hilbert.size.
        """
        if self._rdm is None:
            self._rdm = jit_apply(
                self._model,
                self.variables,
                method=self._model.rdm_one_body,
            )
        return self._rdm  # type: ignore

    def quantum_geometric_tensor(self, qgt_T=None):
        r"""
        Compute the centered quantum geometric tensor (QGT) for the Slater determinant.

        For a Slater determinant, the QGT can be computed analytically using the formula:

        .. math::

            Q_{(\mu i),(\nu j)} = \delta_{ij} \left[\delta_{\mu\nu} - \gamma_{\mu\nu}\right]

        where :math:`\gamma = C C^\dagger` is the one-body reduced density matrix.

        This method returns a :class:`~netket.optimizer.linear_operator.DenseOperator`
        wrapping the dense QGT matrix, compatible with the standard NetKet QGT interface.

        Args:
            qgt_T: Optional QGT type/constructor. If None, defaults to QGTByWick.

        Returns:
            A :class:`~netket.optimizer.linear_operator.DenseOperator` wrapping
            the centered QGT matrix of shape (n_parameters, n_parameters).

        Note:
            - **Generalized HF**: Matrix of shape (n_modes × n_fermions, n_modes × n_fermions)
            - **Restricted HF**: Matrix of shape (n_orbitals × n_fermions_per_spin, n_orbitals × n_fermions_per_spin)
            - **Unrestricted HF**: Block-diagonal matrix with blocks for each spin sector

        Example:
            >>> hi = nk.hilbert.SpinOrbitalFermions(4, s=1/2, n_fermions_per_spin=(2, 2))
            >>> vstate = nk.experimental.vqs.DeterminantVariationalState(
            ...     hi, generalized=False, restricted=True, seed=42
            ... )
            >>> qgt = vstate.quantum_geometric_tensor(diag_shift=0.01)
            >>> print(qgt.matrix.shape)  # (8, 8)
            >>> print(vstate.n_parameters)  # 8
        """
        if qgt_T is None:
            from netket._src.vqs.fermion_mf.qgt import QGTByWick

            qgt_T = QGTByWick

        return qgt_T(self)

    def to_fullsumstate(self, **kwargs):
        """
        Convert this DeterminantVariationalState to a FullSumState.

        Warning:
            Only feasible for small systems due to exponential Hilbert space size.

        Args:
            **kwargs: Additional arguments passed to FullSumState constructor
                     (e.g., chunk_size)

        Returns:
            A FullSumState with the same model and parameters
        """
        return FullSumState(
            hilbert=self.hilbert,
            model=self.model,
            variables=self.variables,
            **kwargs,
        )

    def to_mcstate(
        self,
        sampler: Sampler,
        n_samples: int | None = None,
        n_samples_per_rank: int | None = None,
        **kwargs,
    ):
        """
        Convert this DeterminantVariationalState to an MCState for sampling-based calculations.

        Args:
            sampler: The sampler to use for the MCState (e.g., MetropolisFermionHop)
            n_samples: Number of samples for Monte Carlo estimation
            n_samples_per_rank: Alternative specification of samples per MPI rank
            **kwargs: Additional arguments passed to MCState constructor
                     (e.g., n_discard_per_chain, chunk_size)

        Returns:
            An MCState with the same model and parameters

        Example:
            >>> mf_state = DeterminantVariationalState(hilbert, model)
            >>> sampler = nk.sampler.MetropolisFermionHop(hilbert, graph=g)
            >>> mc_state = mf_state.to_mcstate(sampler, n_samples=1024)
            >>> qgt = mc_state.quantum_geometric_tensor()
        """
        if sampler.hilbert != self.hilbert:
            raise ValueError(
                f"Sampler hilbert space {sampler.hilbert} does not match "
                f"state hilbert space {self.hilbert}"
            )

        return MCState(
            sampler=sampler,
            model=self.model,
            n_samples=n_samples,
            n_samples_per_rank=n_samples_per_rank,
            variables=self.variables,
            **kwargs,
        )

    def to_array(self, normalize: bool = True) -> jax.Array:
        return nknn.to_array(
            self.hilbert,
            self._apply_fun,
            self.variables,
            normalize=normalize,
        )

    def __repr__(self):
        return (
            "DeterminantVariationalState("
            + f"\n  hilbert = {self.hilbert},"
            + f"\n  generalized = {self.model.generalized},"
            + f"\n  restricted = {self.model.restricted},"
            + f"\n  n_parameters = {self.n_parameters})"
        )


# Serialization support


def serialize_DeterminantVariationalState(vstate):
    state_dict = {
        "variables": serialization.to_state_dict(
            serialization_utils.remove_prngkeys(vstate.variables)
        ),
    }
    return state_dict


def deserialize_DeterminantVariationalState(vstate, state_dict):
    import copy

    new_vstate = copy.copy(vstate)
    new_vstate.reset()

    vars = jax.tree_util.tree_map(
        jnp.asarray,
        serialization.from_state_dict(vstate.variables, state_dict["variables"]),
    )
    vars = serialization_utils.restore_prngkeys(vstate.variables, vars)

    new_vstate.variables = vars
    return new_vstate


serialization.register_serialization_state(
    DeterminantVariationalState,
    serialize_DeterminantVariationalState,
    deserialize_DeterminantVariationalState,
)
