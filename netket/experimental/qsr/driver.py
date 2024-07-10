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

from typing import Optional, Union
import warnings

import numpy as np
import jax
import jax.numpy as jnp

from netket import jax as nkjax
from netket.driver import AbstractVariationalDriver
from netket.operator import AbstractOperator
from netket.vqs import VariationalState
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.utils.types import Array
from netket.jax import tree_cast


from netket.stats import statistics

from .dataset import RawQuantumDataset
from .logic_helpers import (
    _grad_local_value_rotated,
    _local_value_rotated_amplitude,
    _compose_grads,
    _grad_negative,
)

BaseType = Union[AbstractOperator, np.ndarray, str]


class QSR(AbstractVariationalDriver):
    r"""
    Quantum state reconstruction driver minimizing KL divergence.

    This driver variationally reconstructs a target state given the measurement data.
    It's achieved by minimizing the average negative log-likelihood, or equivalently,
    the KL divergence between the distributions given by the data and the variational
    state:

    .. math::

        &\min_\theta \frac{1}{N_b} \sum_{b=1}^{N_b} \sum_{\sigma_b} q_b(\sigma_b) \log \left[ \frac{q_b(\sigma_b)}{p_{b\theta}(\sigma_b)} \right] \\
        &\approx \min_\theta \frac{1}{N_b} \sum_{b=1}^{N_b} \frac{1}{|D_b|} \sum_{\sigma_b \in D_b} [-\log p_{b\theta}(\sigma_b)],

    where :math:`\theta` is the variational parameter, :math:`N_b` is the number of
    measurement basis, :math:`q_b(\sigma_b)` is the probability of obtaining the
    outcome state :math:`\sigma_b` in the measurement basis :math:`b` given the
    target state, :math:`p_{b\theta}(\sigma_b)` is the probability of obtaining
    the outcome state :math:`\sigma_b` in the measurement basis :math:`b` given the
    variational state, and :math:`D_b` is the size of the dataset in the measurement
    basis :math:`b`.

    In practice, the noise introduced by mini-batch training hurts the convergence
    of accurate quantum state reconstruction. To alleviate this problem, we use a
    control variate method called `stochastic variance reduced gradient (SVRG) <https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html>`_ to
    reduce the variance of the gradient estimator. Specifically, we update the parameters
    :math:`\theta` according to

    .. math::

        \theta_{i+1} = \theta_{i} -\eta \left\{
        \underbrace{\nabla_\theta \left[\frac{1}{|B_i|}\sum_{\sigma_b\in B_i} \log p_{b\theta_i}(\sigma_b)\right]}_{\text{I: batch gradient}}
        - \underbrace{\nabla_\theta \left[\frac{1}{|B_i|}\sum_{\sigma_b\in B_i} \log p_{b\tilde{\theta}_i}(\sigma_b)\right]}_{\text{II: control variate}}
        + \underbrace{\nabla_\theta \left[\frac{1}{N_b} \sum_{b=1}^{N_b} \frac{1}{|D_b|} \sum_{\sigma_b \in D_b} \log p_{b\tilde{\theta}_i}(\sigma_b)\right]}_{\text{III: expectation of control variate}}
        \right\},

    where term I is the normal batch gradient, term II is the control variate which
    is the batch gradient evaluated with a set of previous parameters

    .. math::

        \tilde{\theta}_i = \begin{cases}
        \theta_i,  &i=0 \mod m, \\
        \tilde{\theta}_{i-1}, &\text{otherwise},
        \end{cases}

    updated for every :math:`m` iterations, and term III is the expectation value of
    the control variate since the mini-batch is sampled uniformly from the whole dataset.
    """

    def __init__(
        self,
        training_data: Union[RawQuantumDataset, tuple[list, list]],
        training_batch_size: int,
        optimizer,
        *,
        variational_state: VariationalState,
        preconditioner: Optional[PreconditionerT] = identity_preconditioner,
        seed: Optional[int] = None,
        batch_sample_replace: Optional[bool] = True,
        control_variate_update_freq: Optional[
            Union[
                int,
                str,
            ]
        ] = None,
        chunk_size: Optional[int] = None,
    ):
        r"""Initializes the QSR driver class.

        Args:
            training_data: A tuple of two arrays (sigma_s, Us). sigma_s is a the
                sampled states and Us is the corresponding rotations.
            training_batch_size: The training batch size.
            optimizer: The optimizer to use. You can use optax optimizers or
                choose from the predefined optimizers netket offers.
            variational_state: The variational state to optimize.
            preconditioner: The preconditioner to use.
                Defaults to identity_preconditioner.
            seed: The RNG seed. Defaults to None.
            batch_sample_replace: Whether to sample with replacement. Defaults to True.
            control_variate_update_freq: The frequency of updating the control variates. Defaults to None.
                "Adaptive" for adaptive update frequency, i.e. n_samples // batch size.
            chunk_size: The chunk size for the control variates. Defaults to None.

        Raises:
            Warning: If the chunk size is not a divisor of the training data size.
            TypeError: If the training data is not a 2 element tuple.
        """
        super().__init__(variational_state, optimizer)
        self.preconditioner = preconditioner

        if not isinstance(training_data, RawQuantumDataset):
            training_data = RawQuantumDataset(training_data)

        self._rng = np.random.default_rng(
            np.asarray(nkjax.mpi_split(nkjax.PRNGKey(seed)))
        )

        # mixed states
        self.mixed_states = variational_state.__class__.__name__ in ["MCMixedState"]

        self.batch_sample_replace = batch_sample_replace
        self.training_batch_size = training_batch_size

        self._raw_dataset = training_data
        self._dataset = training_data.preprocess(
            hilbert=self.state.hilbert, mixed_state_target=self.mixed_states
        )

        # statistical constants
        self._entropy = None

        # control variates
        if control_variate_update_freq == "Adaptive":
            if self.dataset.size <= training_batch_size:
                self._control_variate_update_freq = None
            else:
                self._control_variate_update_freq = (
                    self.dataset.size // training_batch_size
                )
        else:
            self._control_variate_update_freq = control_variate_update_freq
        self._control_variate_expectation = None
        self._control_variate_params = None
        self._chunk_size = chunk_size

        # chunk
        if self._chunk_size is not None:
            self.n_chunk = self.dataset.size // self._chunk_size
            if not self.n_chunk * self._chunk_size == self.dataset.size:
                warnings.warn(
                    "WARNING: chunk size does not divide the number of samples, the last few chunks will be smaller",
                    stacklevel=2,
                )
            self._chunked_indices = np.array_split(
                np.arange(self.dataset.size), self.n_chunk
            )

    @property
    def dataset(self):
        return self._dataset

    def _forward_and_backward(self):
        state = self.state

        if self.mixed_states:
            state_diag = state.diagonal
        else:
            state_diag = self.state

        state.reset()

        # compute the neg gradient of log Z
        self._grad_neg = _grad_negative(state_diag)

        # sample training data for pos grad
        self._batch_data = self.dataset.subsample(
            self.training_batch_size,
            rng=self._rng,
            batch_sample_replace=self.batch_sample_replace,
        )

        # compute the pos gradient of log p
        _log_val_rot, self._grad_pos = _grad_local_value_rotated(
            state._apply_fun,
            state.parameters,
            state.model_state,
            self._batch_data.sigma_p,
            self._batch_data.mels,
            self._batch_data.secs,
        )

        # control variates
        if self._control_variate_update_freq is not None:
            # update control variate
            if self.step_count % self._control_variate_update_freq == 0:
                if self._chunk_size is not None:
                    self._control_variate_expectation = jax.tree_util.tree_map(
                        jnp.zeros_like, self._grad_pos
                    )

                    for i in range(self.n_chunk):
                        chunk_data = self.dataset[self._chunked_indices[i]]
                        _, data = _grad_local_value_rotated(
                            state._apply_fun,
                            state.parameters,
                            state.model_state,
                            chunk_data.sigma_p,
                            chunk_data.mels,
                            chunk_data.secs,
                        )
                        coeff = len(chunk_data) / len(self.dataset)
                        # chunking: accumulate
                        self._control_variate_expectation = jax.tree_util.tree_map(
                            lambda x, y: x + coeff * y,
                            self._control_variate_expectation,
                            data,
                        )

                else:
                    _, self._control_variate_expectation = _grad_local_value_rotated(
                        state._apply_fun,
                        state.parameters,
                        state.model_state,
                        self.dataset.sigma_p,
                        self.dataset.mels,
                        self.dataset.secs,
                    )
                self._control_variate_params = state.parameters

            # control variate gradient
            # it's the graident evaluated at an earlier point
            _, self._grad_pos_cv = _grad_local_value_rotated(
                state._apply_fun,
                self._control_variate_params,
                state.model_state,
                self._batch_data.sigma_p,
                self._batch_data.mels,
                self._batch_data.secs,
            )

            # gather gradient
            # grad <- grad - grad_cv + E[grad_cv]
            self._grad_pos = jax.tree_util.tree_map(
                lambda x, y, Ey: x - y + Ey,
                self._grad_pos,
                self._grad_pos_cv,
                self._control_variate_expectation,
            )

        # compose neg and pos gradient
        # and take complex conjugate
        self._loss_grad = _compose_grads(self._grad_neg, self._grad_pos)

        # restore the square in prob = |psi|^2
        if not self.mixed_states:
            self._loss_grad = jax.tree_util.tree_map(lambda x: x * 2.0, self._loss_grad)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._loss_grad = tree_cast(self._loss_grad, self.state.parameters)

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp = self.preconditioner(self.state, self._loss_grad)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = tree_cast(self._dp, self.state.parameters)

        return self._dp

    def nll(self, return_stats: Optional[bool] = True):
        r"""
        Compute the Negative-Log-Likelihood over a batch of data.

        Args:
            return_stats: if True, return the statistics.

        .. warning::

            Exponentially expensive in the hilbert space size!

        """
        log_val_rot = _local_value_rotated_amplitude(
            self.state._apply_fun,
            self.state.variables,
            self.dataset.sigma_p,
            self.dataset.mels,
            self.dataset.secs,
        )
        if self.mixed_states:
            log_val_rot /= 2

        ce = log_val_rot

        # log norm calculation
        if self.mixed_states:
            log_psi = (
                self.state.diagonal.log_value(self.state.hilbert_physical.all_states())
                / 2
            )
        else:
            log_psi = self.state.log_value(self.state.hilbert.all_states())
        maxl = log_psi.real.max()
        log_n = jnp.log(jnp.exp(2 * (log_psi.real - maxl)).sum()) + 2 * maxl

        # result
        if return_stats:
            return statistics(jnp.real(log_n - ce))
        return jnp.real(log_n - ce)

    def nll_whole_training_set(self, return_stats: Optional[bool] = True):
        r"""
        Compute the Negative-Log-Likelihood over the whole training set.

        Args:
            return_stats: if True, return the statistics.

        .. warning::

            Exponentially expensive in the hilbert space size!
        """
        if self._chunk_size is not None:
            log_val_rot = []
            for i in range(self.n_chunk):
                chunk_data = self.dataset[self._chunked_indices[i]]
                log_val_rot.append(
                    _local_value_rotated_amplitude(
                        self.state._apply_fun,
                        self.state.variables,
                        chunk_data.sigma_p,
                        chunk_data.mels,
                        chunk_data.secs,
                    )
                )
            log_val_rot = jnp.concatenate(log_val_rot)
        else:
            log_val_rot = _local_value_rotated_amplitude(
                self.state._apply_fun,
                self.state.variables,
                self.dataset.sigma_p,
                self.dataset.mels,
                self.dataset.secs,
            )

        # square root <sigma|rho|sigma> to keep in line with the pure state case
        if self.mixed_states:
            log_val_rot /= 2

        ce = log_val_rot

        # log norm calculation
        if self.mixed_states:
            log_psi = (
                self.state.diagonal.log_value(self.state.hilbert_physical.all_states())
                / 2
            )
        else:
            log_psi = self.state.log_value(self.state.hilbert.all_states())
        maxl = log_psi.real.max()
        log_n = jnp.log(jnp.exp(2 * (log_psi.real - maxl)).sum()) + 2 * maxl

        # result
        if return_stats:
            return statistics(jnp.real(log_n - ce))
        return jnp.real(log_n - ce)

    def entropy(
        self,
        target_state: Array,
        n_shots: Optional[int] = 1,
        no_cache: Optional[bool] = False,
    ) -> float:
        r"""
        Compute the average entropy of the probability distributions
        given by the target state in different measurement basis.

        Args:
            target_state: the target state.
            n_shots: number of shots per measurement basis.
            no_cache: if True, do not use the cached value.

        .. warning::

            Exponentially expensive in the hilbert space size!

            Need to know the target state!

        """
        if len(target_state.shape) == 1:
            target_state = target_state.reshape(-1, 1)
            target_state = target_state @ target_state.conj().T
        if self._entropy is not None and not no_cache:
            return self._entropy
        rotations = self._raw_dataset.bases[::n_shots]
        entropy_list = []
        for rot in rotations:
            rho_rot = (rot @ (rot @ target_state).conj().T).conj().T
            prob = np.real(np.diag(rho_rot))
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            entropy_list.append(entropy)
        if no_cache:
            return np.mean(entropy_list)
        self._entropy = np.mean(entropy_list)
        return self._entropy

    def KL(self, target_state: Optional[Array] = None, n_shots: Optional[int] = None):
        r"""
        Compute average KL divergence loss over a batch of data.

        Args:
            target_state: the target state.
            n_shots: number of shots per measurement basis.

        .. warning::

            Exponentially expensive in the hilbert space size!

            Need to know the target state!

        """
        return statistics(
            self.nll(return_stats=False) - self.entropy(target_state, n_shots)
        )

    def KL_whole_training_set(
        self, target_state: Optional[Array] = None, n_shots: Optional[int] = None
    ):
        r"""
        Compute average KL divergence loss over the whole training set.

        Args:
            target_state: the target state.
            n_shots: number of shots per measurement basis.

        .. warning::

            Exponentially expensive in the hilbert space size!

            Need to know the target state!

        """
        return statistics(
            self.nll_whole_training_set(return_stats=False)
            - self.entropy(target_state, n_shots)
        )

    def KL_exact(
        self, target_state: Optional[Array] = None, n_shots: Optional[int] = 1
    ) -> float:
        r"""
        Compute the average KL divergence loss between the variational state and the target state.

        Args:
            target_state: the target state.
            n_shots: number of shots per measurement basis.

        .. warning::

            Exponentially expensive in the hilbert space size!

            Need to know the target state!

        """
        if len(target_state.shape) == 1:
            target_state = target_state.reshape(-1, 1)
            target_state = target_state @ target_state.conj().T
        rotations = self._raw_dataset.bases[::n_shots]
        KL_list = []
        if self.mixed_states:
            vs = self.state.to_matrix(normalize=True)
        else:
            vs = self.state.to_array(normalize=True).reshape(-1, 1)
            vs = vs @ vs.conj().T
        for rot in rotations:
            rho_rot = (rot @ (rot @ vs).conj().T).conj().T
            target_rot = (rot @ (rot @ target_state).conj().T).conj().T
            prob = np.maximum(np.real(np.diag(rho_rot)), 0)
            prob_target = np.maximum(np.real(np.diag(target_rot)), 0)
            KL = np.sum(prob_target * np.log(prob_target / (prob + 1e-10) + 1e-10))
            KL_list.append(KL)
        return np.mean(KL_list)

    def __repr__(self):
        return (
            "QSR("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )
