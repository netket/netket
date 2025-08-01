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

from typing import Any
from functools import partial

import numpy as np

import jax
from jax import numpy as jnp

from netket.utils.types import PyTree, PRNGKeyT, Array
from netket.utils import struct
from netket.jax import dtype_real
from netket.jax.sharding import shard_along_axis

from netket.sampler import MetropolisSamplerState, MetropolisSampler
from netket.sampler.rules import LocalRule, ExchangeRule, HamiltonianRule

# Original C++ Implementation
# https://github.com/netket/netket/blob/1e187ae2b9d2aa3f2e53b09fe743e50763d04c9a/Sources/Sampler/metropolis_hastings_pt.hpp
# https://github.com/netket/netket/blob/1e187ae2b9d2aa3f2e53b09fe743e50763d04c9a/Sources/Sampler/metropolis_hastings_pt.cc
# Python port
# https://github.com/netket/netket/blob/87d469aa8c23f71c4838cf09d7ed7b87ff2ea01f/netket/legacy/sampler/numpy/metropolis_hastings_pt.py


class ParallelTemperingSamplerState(MetropolisSamplerState):
    """
    State for the Metropolis Parallel Tempering sampler.

    Contains the usual quantities, as well as statistics about the paralel tempering.
    """

    beta: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    """The inverse temperatures of the different chains."""

    n_accepted_per_beta: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    """Total number of moves accepted per beta across all JAX processes since the last reset."""
    beta_0_index: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    r"""Index of the position of the chain with :math:`\\beta=1`."""
    beta_position: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    r"""Averaged position of :math:`\\beta=1`."""
    beta_diffusion: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    """Average variance of the position of :math:`\\beta = 1`."""
    exchange_steps: int = struct.field(default_factory=lambda: jnp.zeros((), dtype=int))
    """Number of exchanges between the different temperatures."""

    def __init__(
        self,
        σ: jnp.ndarray,
        rng: jnp.ndarray,
        rule_state: Any | None,
        beta: jnp.ndarray,
        log_prob: jnp.ndarray | None = None,
    ):
        n_chains, n_replicas = beta.shape

        self.beta = beta
        self.n_accepted_per_beta = jnp.zeros((n_chains, n_replicas), dtype=int)
        self.beta_0_index = jnp.zeros((n_chains,), dtype=int)
        self.beta_position = jnp.zeros((n_chains,), dtype=float)
        self.beta_diffusion = jnp.zeros((n_chains,), dtype=float)
        self.exchange_steps = jnp.zeros((), dtype=int)
        super().__init__(σ, rng=rng, rule_state=rule_state, log_prob=log_prob)
        self.n_accepted_proc = jnp.zeros(
            n_chains, dtype=int
        )  # correct shape is (n_chains,) and not (n_batches,)

    def __repr__(self):
        if self.n_steps > 0:
            acc_string = f"# accepted = {self.n_accepted}/{self.n_steps} ({self.acceptance * 100}%), "
        else:
            acc_string = ""

        text = (
            f"ParallelTemperingSamplerState(# replicas = {self.beta.shape[-1]}, "
            + acc_string
            + f"rng state={self.rng}"
        )
        return text

    @property
    def normalized_diffusion(self):
        r"""
        Average variance of the position of :math:`\\beta = 1`.
        In the ideal case, this quantity should be of order ~[0.2, 1.0]
        """
        diffusion = (
            jnp.sqrt(self.beta_diffusion / self.exchange_steps) / self.beta.shape[-1]
        )
        return diffusion.mean()

    @property
    def normalized_position(self):
        r"""
        Average position of :math:`\\beta = 1`, normalized and centered around 0.
        """
        position = self.beta_position / float(self.beta.shape[-1] - 1) - 0.5
        return position.mean()


class ParallelTemperingSampler(MetropolisSampler):
    """
    Metropolis-Hastings with Parallel Tempering sampler.

    This sampler samples an Hilbert space, producing samples off a specific dtype.
    The samples are generated according to a transition rule that must be
    specified.

    The Metropolis Hastings acceptance rule is correted with a temperature.
    """

    n_replicas: int = struct.field(pytree_node=False, default=32)
    """
    The number of replicas evolving with different temperatures for every
    _physical_ markov chain.

    The total number of chains evolved is :code:`n_chains * n_replicas`.
    """
    _beta_sorted: jax.Array = None
    """
    An internal cache for the user-specified betas, sorted.
    """
    _beta_distribution: str = struct.field(pytree_node=False, default="linear")
    """
    An internal for the user-specified distribution of betas.
    """

    def __init__(
        self,
        *args,
        n_replicas: int | None = None,
        betas: str | jax.Array | None = "linear",
        **kwargs,
    ):
        r"""
        :class:`~netket.sampler.ParallelTemperingSampler` is a generic Metropolis-Hastings sampler using
        a transition rule to perform moves in the Markov Chain.
        The transition kernel is used to generate
        a proposed state :math:`s^\prime`, starting from the current state :math:`s`.
        The move is accepted with probability

        .. math::
            A(s\rightarrow s^\prime) = \mathrm{min}\left (1,\frac{P(s^\prime)}{P(s)} e^{L(s,s^\prime)} \right),

        where the probability being sampled from is :math:`P(s)=β|M(s)|^p`. Here :math:`M(s)` is a
        user-provided function (the machine), :math:`p` is also user-provided with default value :math:`p=2`,
        :math:`β` is the temperature of the Markov Chain and :math:`L(s,s^\prime)` is a suitable correcting factor
        computed by the transition kernel.


        Args:
            hilbert: The hilbert space to sample
            rule: A `MetropolisRule` to generate random transitions from a given state as
                    well as uniform random states.
            n_replicas: The number of different temperatures β for the sampling, must be even.
                    (default : 32).
            betas: (Optional) Distribution or list of values of the temperatures β.
                    For the distribution, possibility between "linear" for a linear distribution
                    and "log" for a logarithmic one.
                    For the explicit list of values, the length must be even and the value β=1 must
                    obligatory be an element of betas, all other temperatures must be in (0,1].
                    (default : "lin", i.e. linear distribution between (0,1]).
            n_chains: The number of Markov Chain to be run in parallel on a single JAX process.
            sweep_size: The number of exchanges that compose a single sweep.
                    If None, sweep_size is equal to the number of degrees of freedom being sampled
                    (the size of the input vector s to the machine).
            n_chains: The number of batches of the states to sample (default = 8)
            machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
            dtype: The dtype of the states sampled (default = np.float32).
        """
        if isinstance(betas, str):
            betas = betas.lower()
            if betas not in ["linear", "lin", "logarithmic", "log"]:
                raise ValueError(
                    f"""
                    To initialize the temperatures with a string, you must choose between
                    "lin" for a linear distribution and "log" for a logarithmic distribution.
                    Instead got "{betas}".
                    """
                )
            if n_replicas is None:
                n_replicas = 32

            # truncate to last 3
            beta_distribution = betas[:3]
            betas = None
        elif isinstance(betas, Array) or isinstance(betas, list):
            # TODO: this defaults to float32/64 depending on what is enabled.
            # Might need to think about a better default here.
            betas = jnp.array(betas, dtype=float)
            if betas.ndim != 1:
                raise ValueError("betas must have exactly 1 dimension.")
            if n_replicas is not None:
                raise ValueError(
                    """
                    Cannot specify the list of betas and n_replicas at the same time.
                    The number of replicas will be inferred automatically from the length
                    of the vector of inverse temperatures.
                    """
                )
            # we need beta[0] = 1, so we sort and check that the temperatures are valid
            betas = jnp.sort(betas, descending=True)
            if not (jnp.isclose(betas[0], 1) and betas[-1] > 0):
                raise ValueError(
                    rf"""The values for beta should be in (0,1] and obligatory contain beta=1, instead got [{jnp.min(betas):.2f},{jnp.max(betas):.8f}]."""
                )

            beta_distribution = "custom"
            n_replicas = betas.shape[-1]
        else:
            raise TypeError(
                "`betas` must be a string or a vector of inverse temperatures."
            )

        # verify the number of replicas
        if not (
            isinstance(n_replicas, int)
            and n_replicas > 0
            and np.mod(n_replicas, 2) == 0
        ):
            raise ValueError(
                "n_replicas (or the length of `betas`) must be an even integer > 0."
            )

        self.n_replicas = n_replicas
        self._beta_sorted = betas
        self._beta_distribution = beta_distribution

        super().__init__(*args, **kwargs)

    @property
    def sorted_betas(self):
        """
        The sorted values of the temperatures for each _physical_ markov chain.
        The first value is β = 1 and is the _physical_ temperature.
        """

        if self._beta_sorted is not None:
            return self._beta_sorted
        else:
            if self._beta_distribution == "lin":
                return (
                    1.0
                    - jnp.arange(self.n_replicas, dtype=jnp.float64) / self.n_replicas
                )
            elif self._beta_distribution == "log":
                return -jnp.log(
                    jnp.arange(1, self.n_replicas + 1, dtype=jnp.float64)
                    / (self.n_replicas + 1)
                ) / jnp.log(self.n_replicas + 1)
            else:
                raise NotImplementedError(f"distribution: {self._beta_distribution}")

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"\n  hilbert = {self.hilbert},"
            + f"\n  rule = {self.rule},"
            + f"\n  n_chains = {self.n_chains},"
            + f"\n  n_replicas = {self.n_replicas},"
            + f"\n  beta_distribution = {self._beta_distribution},"
            + f"\n  sweep_size = {self.sweep_size},"
            + f"\n  reset_chains = {self.reset_chains},"
            + f"\n  machine_power = {self.machine_pow},"
            + f"\n  dtype = {self.dtype}"
            + ")"
        )

    @property
    def n_batches(self) -> int:
        r"""
        The batch size of the configuration $\sigma$ used by this sampler on this
        jax process.

        Equal `n_chains * n_replicas`.
        """
        return self.n_chains * self.n_replicas

    @partial(jax.jit, static_argnums=1)
    def _init_state(
        self, machine, parameters: PyTree, key: PRNGKeyT
    ) -> ParallelTemperingSamplerState:
        key_state, key_rule, rng = jax.random.split(key, 3)
        rule_state = self.rule.init_state(self, machine, parameters, key_rule)
        σ = self.rule.random_state(self, machine, parameters, rule_state, rng)
        σ = shard_along_axis(σ, axis=0)

        output_dtype = jax.eval_shape(machine.apply, parameters, σ).dtype
        log_prob = jnp.full((self.n_batches,), -jnp.inf, dtype=dtype_real(output_dtype))
        log_prob = shard_along_axis(log_prob, axis=0)

        beta = jnp.tile(self.sorted_betas, (self.n_batches // self.n_replicas, 1))

        return ParallelTemperingSamplerState(
            σ=σ,
            log_prob=log_prob,
            rng=key_state,
            rule_state=rule_state,
            beta=beta,
        )

    @partial(jax.jit, static_argnums=1)
    def _reset(self, machine, parameters: PyTree, state: ParallelTemperingSamplerState):
        state = super()._reset(machine, parameters, state)
        return state.replace(
            n_accepted_per_beta=jnp.zeros_like(state.n_accepted_per_beta),
            beta_position=jnp.zeros_like(state.beta_position),
            beta_diffusion=jnp.zeros_like(state.beta_diffusion),
            exchange_steps=jnp.zeros_like(state.exchange_steps),
            # beta=beta,
            # beta_0_index=jnp.zeros((self.n_chains,), dtype=jnp.int64),
        )

    def _sample_next(
        self, machine, parameters: PyTree, state: ParallelTemperingSamplerState
    ):
        def loop_body(
            i, state: ParallelTemperingSamplerState
        ) -> ParallelTemperingSamplerState:
            # 1 to propagate for next iteration, 1 for uniform rng and n_chains for transition kernel
            new_rng, key1, key2, key3, key4 = jax.random.split(state.rng, 5)

            beta = state.beta

            ## Usual Metropolis sampling
            σp, log_prob_correction = self.rule.transition(
                self, machine, parameters, state, key1, state.σ
            )
            proposal_log_prob = self.machine_pow * machine.apply(parameters, σp).real

            uniform = jax.random.uniform(key2, shape=(self.n_batches,))
            if log_prob_correction is not None:
                do_accept = uniform < jnp.exp(
                    beta.reshape((-1,))
                    * (proposal_log_prob - state.log_prob + log_prob_correction)
                )
            else:
                do_accept = uniform < jnp.exp(
                    beta.reshape((-1,)) * (proposal_log_prob - state.log_prob)
                )

            # do_accept must match ndim of proposal and state (which is 2)
            new_σ = jnp.where(do_accept.reshape(-1, 1), σp, state.σ)
            n_accepted_per_beta = state.n_accepted_per_beta + do_accept.reshape(
                (self.n_batches // self.n_replicas, self.n_replicas)
            )

            new_log_prob = jax.numpy.where(
                do_accept.reshape(-1), proposal_log_prob, state.log_prob
            )

            ## exchange betas

            # randomly decide if every set of replicas should be swapped in even or odd order
            swap_order = jax.random.randint(
                key3,
                minval=0,
                maxval=2,
                shape=(self.n_batches // self.n_replicas,),
            )  # 0 or 1

            # indices of even swapped elements (per-row)
            idxs = jnp.arange(0, self.n_replicas, 2).reshape(
                (1, -1)
            ) + swap_order.reshape((-1, 1))
            # indices off odd swapped elements (per-row)
            inn = (idxs + 1) % self.n_replicas

            # for every rows of the input, swap elements at idxs with elements at inn
            @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)
            def swap_rows(beta_row, idxs, inn):
                proposed_beta = beta_row.at[idxs].set(
                    beta_row[inn], unique_indices=True, indices_are_sorted=True
                )
                proposed_beta = proposed_beta.at[inn].set(
                    beta_row[idxs], unique_indices=True, indices_are_sorted=False
                )
                return proposed_beta

            proposed_beta = swap_rows(beta, idxs, inn)

            @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)
            def compute_proposed_prob(prob, idxs, inn):
                # prob[idxs] = (beta_i - beta_j) log psi(x_i)
                # prob[inn] = (beta_j - beta_i) log psi(x_j)
                # so we have to add the log probabilities to get the right acceptance
                prob_rescaled = prob[idxs] + prob[inn]
                return prob_rescaled

            # compute the probability of the swaps
            log_prob = (proposed_beta - state.beta) * new_log_prob.reshape(
                (self.n_batches // self.n_replicas, self.n_replicas)
            )

            prob_rescaled = jnp.exp(compute_proposed_prob(log_prob, idxs, inn))

            uniform = jax.random.uniform(
                key4,
                shape=(
                    self.n_batches // self.n_replicas,
                    self.n_replicas // 2,
                ),
            )

            # decide where to swap
            do_swap = uniform < prob_rescaled

            do_swap = jnp.dstack((do_swap, do_swap)).reshape(
                (-1, self.n_replicas)
            )  # concat along last dimension

            # roll if swap_order is odd
            do_swap = jax.vmap(jnp.where, in_axes=(0, 0, 0), out_axes=0)(
                swap_order == 0, do_swap, jnp.roll(do_swap, 1, axis=-1)
            )

            # Do the swap where it has to be done
            new_beta = jax.numpy.where(do_swap, proposed_beta, beta)

            swap_order = swap_order.reshape(-1)

            beta_0_moved = jax.vmap(jnp.take)(
                do_swap, state.beta_0_index
            )  # flag saying if beta_0 should move
            proposed_beta_0_index = jnp.mod(
                state.beta_0_index
                + (-2 * jnp.mod(swap_order, 2) + 1)
                * (-2 * jnp.mod(state.beta_0_index, 2) + 1),
                self.n_replicas,
            )

            new_beta_0_index = jnp.where(
                beta_0_moved, proposed_beta_0_index, state.beta_0_index
            )

            # swap acceptances
            swapped_n_accepted_per_beta = swap_rows(n_accepted_per_beta, idxs, inn)
            new_n_accepted_per_beta = jax.numpy.where(
                do_swap,
                swapped_n_accepted_per_beta,
                n_accepted_per_beta,
            )

            # Update statistics to compute diffusion coefficient of replicas
            # Total exchange steps performed
            new_exchange_steps = state.exchange_steps + 1
            delta = new_beta_0_index - state.beta_position
            new_beta_position = state.beta_position + delta / new_exchange_steps
            delta2 = new_beta_0_index - new_beta_position
            new_beta_diffusion = state.beta_diffusion + delta * delta2

            return state.replace(
                rng=new_rng,
                σ=new_σ,
                log_prob=new_log_prob,
                beta=new_beta,
                beta_0_index=new_beta_0_index,
                n_accepted_per_beta=new_n_accepted_per_beta,
                beta_position=new_beta_position,
                beta_diffusion=new_beta_diffusion,
                exchange_steps=new_exchange_steps,
                n_steps_proc=state.n_steps_proc + self.n_batches // self.n_replicas,
                n_accepted_proc=jax.vmap(jnp.take)(
                    new_n_accepted_per_beta, new_beta_0_index
                ),
            )

        new_state = jax.lax.fori_loop(0, self.sweep_size, loop_body, state)

        σ_flat = new_state.σ
        σ = σ_flat.reshape((-1, self.n_replicas, σ_flat.shape[-1]))
        σ_new = jnp.take_along_axis(σ, new_state.beta_0_index[:, None, None], axis=1)
        σ_new = jax.lax.collapse(σ_new, 0, 2)  # remove dummy replica dim

        log_prob = new_state.log_prob.reshape((-1, self.n_replicas))
        log_prob_new = jnp.take_along_axis(
            log_prob, new_state.beta_0_index[:, None], axis=1
        )
        log_prob_new = jax.lax.collapse(log_prob_new, 0, 2)  # remove dummy replica dim

        return new_state, (σ_new, log_prob_new)


def ParallelTemperingLocal(hilbert, *args, **kwargs):
    r"""
    Sampler acting on one local degree of freedom.

    This sampler acts locally only on one local degree of freedom :math:`s_i`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s_N`,
    where :math:`s^\prime_i \neq s_i`.

    The transition probability associated to this
    sampler can be decomposed into two steps:

    1. One of the site indices :math:`i = 1\dots N` is chosen
    with uniform probability.

    2. Among all the possible (:math:`m`) values that :math:`s_i` can take,
    one of them is chosen with uniform probability.

    For example, in the case of spin :math:`1/2` particles, :math:`m=2`
    and the possible local values are :math:`s_i = -1,+1`.
    In this case then :class:`MetropolisLocal` is equivalent to flipping a random spin.

    In the case of bosons, with occupation numbers
    :math:`s_i = 0, 1, \dots n_{\mathrm{max}}`, :class:`MetropolisLocal`
    would pick a random local occupation number uniformly between :math:`0`
    and :math:`n_{\mathrm{max}}`.

    Args:
        hilbert: The hilbert space to sample
        n_chains: The number of Markov Chain to be run in parallel on a single process.
        sweep_size: The number of exchanges that compose a single sweep.
                If None, sweep_size is equal to the number of degrees of freedom being sampled
                (the size of the input vector s to the machine).
        n_chains: The number of batches of the states to sample (default = 8)
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float32).
    """
    return ParallelTemperingSampler(hilbert, LocalRule(), *args, **kwargs)


def ParallelTemperingExchange(
    hilbert, *args, clusters=None, graph=None, d_max=1, **kwargs
):
    r"""
    This sampler acts locally only on two local degree of freedom :math:`s_i` and :math:`s_j`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N`,
    where in general :math:`s^\prime_i \neq s_i` and :math:`s^\prime_j \neq s_j`.
    The sites :math:`i` and :math:`j` are also chosen to be within a maximum graph
    distance of :math:`d_{\mathrm{max}}`.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A pair of indices :math:`i,j = 1\dots N`, and such
       that :math:`\mathrm{dist}(i,j) \leq d_{\mathrm{max}}`,
       is chosen with uniform probability.

    2. The sites are exchanged, i.e. :math:`s^\prime_i = s_j` and :math:`s^\prime_j = s_i`.

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum numbers
    are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where :math:`\sum_i s_i = \mathrm{constant}` is needed,
    otherwise the sampling would be strongly not ergodic.

    Args:
        hilbert: The hilbert space to sample
        d_max: The maximum graph distance allowed for exchanges.
        n_chains: The number of Markov Chain to be run in parallel on a single process.
        sweep_size: The number of exchanges that compose a single sweep.
                If None, sweep_size is equal to the number of degrees of freedom being sampled
                (the size of the input vector s to the machine).
        n_chains: The number of batches of the states to sample (default = 8)
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float32).


    Examples:
          Sampling from a RBM machine in a 1D lattice of spin 1/2, using
          nearest-neighbours exchanges.

          >>> import netket as nk
          >>>
          >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
          >>> hi=nk.hilbert.Spin(s=0.5, N=g.n_nodes)
          >>>
          >>> # Construct a MetropolisExchange Sampler
          >>> sa = nk.sampler.ParallelTemperingExchange(hi, graph=g)
          >>> print(sa)
          ParallelTemperingSampler(rule = ExchangeRule(# of clusters: 200), n_chains = 16, sweep_size = 100, reset_chains = False, machine_power = 2, dtype = int8)

    """
    rule = ExchangeRule(clusters=clusters, graph=graph, d_max=d_max)
    return ParallelTemperingSampler(hilbert, rule, *args, **kwargs)


def ParallelTemperingHamiltonian(hilbert, hamiltonian, *args, **kwargs):
    r"""
    Sampling based on the off-diagonal elements of a Hamiltonian (or a generic Operator).
    In this case, the transition matrix is taken to be:

    .. math::
       T( \mathbf{s} \rightarrow \mathbf{s}^\prime) = \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),

    where :math:`\theta(x)` is the Heaviside step function, and :math:`\mathcal{N}(\mathbf{s})`
    is a state-dependent normalization.
    The effect of this transition probability is then to connect (with uniform probability)
    a given state :math:`\mathbf{s}` to all those states :math:`\mathbf{s}^\prime` for which the Hamiltonian has
    finite matrix elements.
    Notice that this sampler preserves by construction all the symmetries
    of the Hamiltonian. This is in generally not true for the local samplers instead.

    Args:
       machine: A machine :math:`\Psi(s)` used for the sampling.
                The probability distribution being sampled
                from is :math:`F(\Psi(s))`, where the function
                :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
       hamiltonian: The operator used to perform off-diagonal transition.
       n_chains: The number of Markov Chain to be run in parallel on a single process.
       sweep_size: The number of exchanges that compose a single sweep.
                   If None, sweep_size is equal to the number of degrees of freedom (n_visible).


    Examples:
       Sampling from a RBM machine in a 1D lattice of spin 1/2

       >>> import netket as nk
       >>>
       >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
       >>> hi=nk.hilbert.Spin(s=0.5, N=g.n_nodes)
       >>>
       >>> # Transverse-field Ising Hamiltonian
       >>> ha = nk.operator.Ising(hilbert=hi, h=1.0, graph=g)
       >>>
       >>> # Construct a MetropolisExchange Sampler
       >>> sa = nk.sampler.ParallelTemperingHamiltonian(hi, hamiltonian=ha)
       >>> print(sa)
       ParallelTemperingSampler(rule = HamiltonianRuleJax(operator=IsingJax(J=1.0, h=1.0; dim=100)), n_chains = 16, sweep_size = 100, reset_chains = False, machine_power = 2, dtype = int8)

    """
    rule = HamiltonianRule(hamiltonian)
    return ParallelTemperingSampler(hilbert, rule, *args, **kwargs)
