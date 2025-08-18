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
from typing import Any
from collections.abc import Callable
from textwrap import dedent

import numpy as np

import jax
from flax import linen as nn
from jax import numpy as jnp

from netket.hilbert import AbstractHilbert, SpinOrbitalFermions

from netket import config
from netket.utils import wrap_afun
from netket.utils.types import PyTree, DType
from netket.utils import struct

from netket.jax.sharding import (
    shard_along_axis,
)
from netket.jax import apply_chunked, dtype_real

from .base import Sampler, SamplerState
from .rules import MetropolisRule


class MetropolisSamplerState(SamplerState):
    """
    State for a Metropolis sampler.

    Contains the current configuration, the RNG state and the (optional)
    state of the transition rule.
    """

    σ: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    """Current batch of configurations in the Markov chain."""
    log_prob: jnp.ndarray = struct.field(sharded=True, serialize=False)
    """Log probabilities of the current batch of configurations σ in the Markov chain."""
    rng: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-rng-key"
        )
    )
    """State of the random number generator (key, in jax terms)."""
    rule_state: Any | None
    """Optional state of the transition rule."""

    n_steps_proc: int = struct.field(default_factory=lambda: jnp.zeros((), dtype=int))
    """Number of moves performed along the chains in this process since the last reset."""
    n_accepted_proc: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    """Number of accepted transitions among the chains in this process since the last reset."""

    def __init__(
        self,
        σ: jnp.ndarray,
        rng: jnp.ndarray,
        rule_state: Any | None,
        log_prob: jnp.ndarray | None = None,
    ):
        self.σ = σ
        self.rng = rng
        self.rule_state = rule_state

        if log_prob is None:
            log_prob = jnp.full(self.σ.shape[:-1], -jnp.inf, dtype=float)
        self.log_prob = shard_along_axis(log_prob, axis=0)

        self.n_accepted_proc = shard_along_axis(
            jnp.zeros(σ.shape[0], dtype=int), axis=0
        )
        self.n_steps_proc = jnp.zeros((), dtype=int)
        super().__init__()

    @property
    def acceptance(self) -> float | None:
        """The fraction of accepted moves across all chains.

        The rate is computed since the last reset of the sampler.
        Will return None if no sampling has been performed since then.
        """
        if self.n_steps == 0:
            return None

        return self.n_accepted / self.n_steps

    @property
    def n_steps(self) -> int:
        """Total number of moves performed across all processes since the last reset."""
        return self.n_steps_proc

    @property
    def n_accepted(self) -> int:
        """Total number of moves accepted across all processes since the last reset."""
        # jit sum for gda
        return jnp.sum(self.n_accepted_proc)

    def __repr__(self):
        try:
            if self.n_steps > 0:
                acc_string = f"# accepted = {self.n_accepted}/{self.n_steps} ({self.acceptance * 100}%), "
            else:
                acc_string = ""

            return f"{type(self).__name__}({acc_string}rng state={self.rng})"
        except TypeError:
            return f"{type(self).__name__}(???, rng state={self.rng})"

    def __process_deserialization_updates__(self, updates):
        # In netket 3.15 we changed the default dtype of samples
        # to integer dtypes in most of the time. Without this,
        # deserialization of old files would be broken.
        if self.σ.dtype != updates["σ"].dtype:
            updates["σ"] = updates["σ"].astype(self.σ.dtype)
        return updates


def _assert_good_sample_shape(samples, shape, dtype, obj=""):
    canonical_dtype = jax.dtypes.canonicalize_dtype(dtype)
    if samples.shape != shape or samples.dtype != canonical_dtype:
        raise ValueError(
            dedent(
                f"""

            The samples returned by the {obj} have `shape={samples.shape}` and
            `dtype={samples.dtype}`, but the sampler requires `shape={shape} and
            `dtype={canonical_dtype}` (canonicalized from {dtype}).

            If you are using a custom transition rule, check that it returns the
            correct shape and dtype.

            If you are using a built-in transition rule, there might be a mismatch
            between hilbert spaces, or it's a bug in NetKet.

            """
            )
        )


def _assert_good_log_prob_shape(log_prob, n_chains_per_rank, machine):
    if log_prob.shape != (n_chains_per_rank,):
        raise ValueError(
            dedent(
                f"""

            The output of the model {machine} has `shape={log_prob.shape}`, but
            `shape=({n_chains_per_rank},)` was expected.

            This might be because of an hilbert space mismatch or because your
            model is ill-configured.

            """
            )
        )


def _round_n_chains_to_next_multiple(
    n_chains, n_chains_per_whatever, n_devices, whatever_str
):
    # small helper function to round the number of chains to the next multiple of n_devices
    if n_chains is not None and n_chains_per_whatever is not None:
        raise ValueError(
            f"Cannot specify both `n_chains` and `n_chains_per_{whatever_str}`"
        )
    elif n_chains is not None:
        n_chains_per_whatever = max(int(np.ceil(n_chains / n_devices)), 1)
        if n_chains_per_whatever * n_devices != n_chains:
            if jax.process_index() == 0:
                import warnings

                warnings.warn(
                    f"Using {n_chains_per_whatever} chains per {whatever_str} among {n_devices} {whatever_str}s "
                    f"(total={n_chains_per_whatever * n_devices} instead of n_chains={n_chains}). "
                    f"To directly control the number of chains on every {whatever_str}, specify "
                    f"`n_chains_per_{whatever_str}` when constructing the sampler. "
                    f"To silence this warning, either use `n_chains_per_{whatever_str}` or use `n_chains` "
                    f"that is a multiple of the number of {whatever_str}s",
                    category=UserWarning,
                    stacklevel=2,
                )
    return n_chains_per_whatever * n_devices


class MetropolisSampler(Sampler):
    r"""
    Metropolis-Hastings sampler for a Hilbert space according to a specific transition rule.

    The transition rule is used to generate a proposed state :math:`s^\prime`, starting from the
    current state :math:`s`. The move is accepted with probability

    .. math::

        A(s \rightarrow s^\prime) = \mathrm{min} \left( 1,\frac{P(s^\prime)}{P(s)} e^{L(s,s^\prime)} \right) ,

    where the probability being sampled from is :math:`P(s)=|M(s)|^p`. Here :math:`M(s)` is a
    user-provided function (the machine), :math:`p` is also user-provided with default value :math:`p=2`,
    and :math:`L(s,s^\prime)` is a suitable correcting factor computed by the transition kernel.

    The dtype of the sampled states can be chosen.
    """

    rule: MetropolisRule = None  # type: ignore
    """The Metropolis transition rule."""
    sweep_size: int = struct.field(pytree_node=False, default=None)
    """Number of sweeps for each step along the chain. Defaults to the number
    of sites in the Hilbert space."""
    n_chains: int = struct.field(pytree_node=False)
    """Total number of independent chains across all devices."""
    chunk_size: int | None = struct.field(pytree_node=False, default=None)
    """Chunk size for evaluating wave functions."""
    reset_chains: bool = struct.field(pytree_node=False, default=False)
    """If True, resets the chain state when `reset` is called on every new sampling."""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        rule: MetropolisRule,
        *,
        sweep_size: int = None,
        reset_chains: bool = False,
        n_chains: int | None = None,
        n_chains_per_rank: int | None = None,
        chunk_size: int | None = None,
        machine_pow: int = 2,
        dtype: DType = None,
    ):
        """
        Constructs a Metropolis Sampler.

        Args:
            hilbert: The Hilbert space to sample.
            rule: A `MetropolisRule` to generate random transitions from a given state as
                well as uniform random states.
            n_chains: The total number of independent Markov chains across all devices.
                Either specify this or `n_chains_per_rank`. If you have a single device, the two are equivalent;
                if you have multiple devices and `n_chains` is specified, then every device will host
                `n_chains/n_devices` chains. In general, we recommend specifying `n_chains_per_rank`
                as it is more portable.
            n_chains_per_rank: Number of independent chains on every device (default = 16).
            chunk_size: Chunk size for evaluating the ansatz while sampling. Must divide n_chains_per_rank.
            sweep_size: Number of sweeps for each step along the chain.
                This is equivalent to subsampling the Markov chain. (Defaults to the number of sites
                in the Hilbert space.)
            reset_chains: If True, resets the chain state when `reset` is called on every
                new sampling (default = False).
            machine_pow: The power to which the machine should be exponentiated to generate
                the pdf (default = 2).
            dtype: The dtype of the states sampled (default = np.float64).
        """

        # Validate the inputs
        if not isinstance(rule, MetropolisRule):
            raise TypeError(
                f"The second positional argument, rule, must be a MetropolisRule but "
                f"`type(rule)={type(rule)}`."
            )

        if not isinstance(reset_chains, bool):
            raise TypeError("reset_chains must be a boolean.")

        if sweep_size is None:
            sweep_size = hilbert.size

        # Default n_chains per rank, if unset
        if n_chains is None and n_chains_per_rank is None:
            # TODO set it to a few hundred if on GPU?
            n_chains_per_rank = 16

        if config.netket_experimental_sharding:
            device_count = jax.device_count()
        else:
            device_count = 1

        n_chains = _round_n_chains_to_next_multiple(
            n_chains,
            n_chains_per_rank,
            device_count,
            "rank",
        )
        n_chains_per_rank = n_chains // device_count

        if (
            chunk_size is not None
            and n_chains_per_rank > chunk_size
            and n_chains_per_rank % chunk_size != 0
        ):
            raise ValueError(
                f"Chunk size must divide number of chains per rank, {n_chains_per_rank}"
            )

        super().__init__(
            hilbert=hilbert,
            machine_pow=machine_pow,
            dtype=dtype,
        )

        self.n_chains = n_chains
        self.reset_chains = reset_chains
        self.rule = rule
        self.sweep_size = sweep_size

    def sample_next(
        self,
        machine: Callable | nn.Module,
        parameters: PyTree,
        state: SamplerState | None = None,
    ) -> tuple[SamplerState, jnp.ndarray]:
        """
        Samples the next state in the Markov chain.

        Args:
            machine: A Flax module or callable with the forward pass of the log-pdf.
                If it is a callable, it should have the signature
                :code:`f(parameters, σ) -> jnp.ndarray`.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If not specified, then initialize and reset it.

        Returns:
            state: The new state of the sampler.
            σ: The next batch of samples.

        Note:
            The return order is inverted wrt `sample` because when called inside of
            a scan function the first returned argument should be the state.
        """
        if state is None:
            state = self.reset(machine, parameters)

        return self._sample_next(wrap_afun(machine), parameters, state)

    @partial(jax.jit, static_argnums=1)
    def _init_state(self, machine, parameters, key):
        key_state, key_rule = jax.random.split(key)
        rule_state = self.rule.init_state(self, machine, parameters, key_rule)
        σ = jnp.zeros((self.n_batches, self.hilbert.size), dtype=self.dtype)
        σ = shard_along_axis(σ, axis=0)

        output_dtype = jax.eval_shape(machine.apply, parameters, σ).dtype
        log_prob = jnp.full((self.n_batches,), -jnp.inf, dtype=dtype_real(output_dtype))
        log_prob = shard_along_axis(log_prob, axis=0)

        state = MetropolisSamplerState(
            σ=σ, rng=key_state, rule_state=rule_state, log_prob=log_prob
        )
        # If we don't reset the chain at every sampling iteration, then reset it
        # now.
        if not self.reset_chains:
            key_state, rng = jax.jit(jax.random.split)(key_state)
            σ = self.rule.random_state(self, machine, parameters, state, rng)
            _assert_good_sample_shape(
                σ,
                (self.n_batches, self.hilbert.size),
                self.dtype,
                f"{self.rule}.random_state",
            )
            σ = shard_along_axis(σ, axis=0)
            state = state.replace(σ=σ, rng=key_state)
        return state

    @partial(jax.jit, static_argnums=1)
    def _reset(self, machine, parameters, state):
        rng = state.rng

        if self.reset_chains:
            rng, key = jax.random.split(state.rng)
            σ = self.rule.random_state(self, machine, parameters, state, key)
            _assert_good_sample_shape(
                σ,
                (self.n_batches, self.hilbert.size),
                self.dtype,
                f"{self.rule}.random_state",
            )
            σ = shard_along_axis(σ, axis=0)
        else:
            σ = state.σ

        # Recompute the log_probability of the current samples
        apply_machine = apply_chunked(
            machine.apply, in_axes=(None, 0), chunk_size=self.chunk_size
        )
        log_prob_σ = self.machine_pow * apply_machine(parameters, σ).real

        rule_state = self.rule.reset(self, machine, parameters, state)

        return state.replace(
            σ=σ,
            log_prob=log_prob_σ,
            rng=rng,
            rule_state=rule_state,
            n_steps_proc=jnp.zeros_like(state.n_steps_proc),
            n_accepted_proc=jnp.zeros_like(state.n_accepted_proc),
        )

    def _sample_next(self, machine, parameters, state):
        """
        Implementation of `sample_next` for subclasses of `MetropolisSampler`.

        If you subclass `MetropolisSampler`, you should override this and not `sample_next`
        itself, because `sample_next` contains some common logic.
        """
        apply_machine = apply_chunked(
            machine.apply, in_axes=(None, 0), chunk_size=self.chunk_size
        )

        def loop_body(i, state):
            # 1 to propagate for next iteration, 1 for uniform rng and n_chains for transition kernel
            new_rng, key1, key2 = jax.random.split(state.rng, 3)

            σp, log_prob_correction = self.rule.transition(
                self, machine, parameters, state, key1, state.σ
            )

            _assert_good_sample_shape(
                σp,
                (self.n_batches, self.hilbert.size),
                self.dtype,
                f"{self.rule}.transition",
            )
            proposal_log_prob = self.machine_pow * apply_machine(parameters, σp).real
            _assert_good_log_prob_shape(proposal_log_prob, self.n_batches, machine)

            uniform = jax.random.uniform(key2, shape=(self.n_batches,))
            if log_prob_correction is not None:
                do_accept = uniform < jnp.exp(
                    proposal_log_prob - state.log_prob + log_prob_correction
                )
            else:
                do_accept = uniform < jnp.exp(proposal_log_prob - state.log_prob)

            return state.replace(
                σ=jnp.where(do_accept.reshape(-1, 1), σp, state.σ),
                log_prob=jax.numpy.where(
                    do_accept.reshape(-1), proposal_log_prob, state.log_prob
                ),
                rng=new_rng,
                n_accepted_proc=state.n_accepted_proc + do_accept,
                n_steps_proc=state.n_steps_proc + self.n_batches,
            )

        new_state = jax.lax.fori_loop(0, self.sweep_size, loop_body, state)

        return new_state, (new_state.σ, new_state.log_prob)

    @partial(
        jax.jit, static_argnames=("machine", "chain_length", "return_log_probabilities")
    )
    def _sample_chain(
        self,
        machine,
        parameters,
        state,
        chain_length,
        return_log_probabilities: bool = False,
    ):
        """
        Samples `chain_length` batches of samples along the chains.

        Internal method used for jitting calls.

        Arguments:
            machine: A Flax module with the forward pass of the log-pdf.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler.
            chain_length: The length of the chains.

        Returns:
            σ: The next batch of samples.
            state: The new state of the sampler
        """
        state, (samples, log_probabilities) = jax.lax.scan(
            lambda state, _: self._sample_next(machine, parameters, state),
            state,
            xs=None,
            length=chain_length,
        )
        # make it (n_chains, n_samples_per_chain) as expected by netket.stats.statistics
        samples = jnp.swapaxes(samples, 0, 1)
        log_probabilities = jnp.swapaxes(log_probabilities, 0, 1)

        if return_log_probabilities:
            return (samples, log_probabilities), state
        else:
            return samples, state

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"\n  hilbert = {self.hilbert},"
            + f"\n  rule = {self.rule},"
            + f"\n  n_chains = {self.n_chains},"
            + f"\n  sweep_size = {self.sweep_size},"
            + f"\n  reset_chains = {self.reset_chains},"
            + f"\n  machine_power = {self.machine_pow},"
            + f"\n  dtype = {self.dtype}"
            + ")"
        )

    def __str__(self):
        return (
            f"{type(self).__name__}("
            + f"rule = {self.rule}, "
            + f"n_chains = {self.n_chains}, "
            + f"sweep_size = {self.sweep_size}, "
            + f"reset_chains = {self.reset_chains}, "
            + f"machine_power = {self.machine_pow}, "
            + f"dtype = {self.dtype})"
        )


def MetropolisLocal(hilbert, **kwargs) -> MetropolisSampler:
    r"""
    Sampler acting on one local degree of freedom.

    This sampler acts locally only on one local degree of freedom :math:`s_i`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s_N`,
    where :math:`s^\prime_i \neq s_i`.

    The transition probability associated to this
    sampler can be decomposed into two steps:

    1. One of the site indices :math:`i = 1\dots N` is chosen with uniform probability.

    2. Among all the possible (:math:`m - 1`) values that :math:`s^\prime_i` can take,
    one of them is chosen with uniform probability.

    For example, in the case of spin :math:`1/2` particles, :math:`m=2`
    and the possible local values are :math:`s_i = -1,+1`.
    In this case then :class:`MetropolisLocal` is equivalent to flipping a random spin.

    In the case of bosons, with occupation numbers
    :math:`s_i = 0, 1, \dots n_{\mathrm{max}}`, :class:`MetropolisLocal`
    would pick a random local occupation number uniformly between :math:`0`
    and :math:`n_{\mathrm{max}}` except the current :math:`s_i`.

    Args:
        hilbert: The Hilbert space to sample.
        n_chains: The total number of independent Markov chains across all devices.
            Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every device (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    from .rules import LocalRule

    return MetropolisSampler(hilbert, LocalRule(), **kwargs)


def MetropolisExchange(
    hilbert, *, clusters=None, graph=None, d_max=1, **kwargs
) -> MetropolisSampler:
    r"""
    This sampler acts locally only on two local degree of freedom :math:`s_i` and :math:`s_j`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N`,
    where in general :math:`s^\prime_i \neq s_i` and :math:`s^\prime_j \neq s_j`.
    The sites :math:`i` and :math:`j` are also chosen to be within a maximum graph
    distance of :math:`d_{\mathrm{max}}`.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A pair of indices :math:`i,j = 1\dots N`, and such that
       :math:`\mathrm{dist}(i,j) \leq d_{\mathrm{max}}`,
       is chosen with uniform probability.

    2. The sites are exchanged, i.e. :math:`s^\prime_i = s_j` and :math:`s^\prime_j = s_i`.

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum numbers
    are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where :math:`\sum_i s_i = \mathrm{constant}` is needed,
    otherwise the sampling would be strongly not ergodic.

    .. warning::

        If you are working with systems where the number of nodes in the physical lattice
        does not match the number of degrees of freedom, you must be careful!

        A typical example is a system of Spin-1/2 fermions on a lattice with N sites, where the
        first N degrees of freedom correspond to the spin down degrees of freedom and the
        next N degrees of freedom correspond to the spin up degrees of freedom.

        In this case, you tipically want to exchange only degrees of freedom of the same type.
        A simple way to achieve this is to double the graph:

        .. code-block:: python

            import netket as nk
            g = nk.graph.Square(5)
            hi = nk.hilbert.SpinOrbitalFermions(g.n_nodes, s=0.5)

            exchange_graph = nk.graph.disjoint_union(g, g)
            print("Exchange graph size:", exchange_graph.n_nodes)

            sa = nk.sampler.MetropolisExchange(hi, graph=exchange_graph, d_max=1)



    Args:
        hilbert: The Hilbert space to sample.
        d_max: The maximum graph distance allowed for exchanges.
        n_chains: The total number of independent Markov chains across all devices.
            Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every device (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites
            in the Hilbert space.
            This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new
            sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the
            pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).

    Examples:
          Sampling from a RBM machine in a 1D lattice of spin 1/2, using
          nearest-neighbor exchanges.

          >>> import netket as nk
          >>>
          >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
          >>> hi=nk.hilbert.Spin(s=0.5, N=g.n_nodes)
          >>>
          >>> # Construct a MetropolisExchange Sampler
          >>> sa = nk.sampler.MetropolisExchange(hi, graph=g)
          >>> print(sa)
          MetropolisSampler(rule = ExchangeRule(# of clusters: 200), n_chains = 16, sweep_size = 100, reset_chains = False, machine_power = 2, dtype = int8)
    """
    from .rules import ExchangeRule

    if isinstance(hilbert, SpinOrbitalFermions):
        warn = True
        if graph is not None and graph.n_nodes < hilbert.size:
            warn = True
        if jax.process_count() == 0 and warn:
            import warnings

            warnings.warn(
                "Using MetropolisExchange with SpinOrbitalFermions can yield unintended behavior."
                "Note that MetropolisExchange only exchanges fermions according to the graph edges "
                "and might not hop fermions of all the spin sectors (see `nk.samplers.rule.FermionHopRule`). "
                "We recommend using MetropolisFermionHop.",
                category=UserWarning,
                stacklevel=2,
            )

    rule = ExchangeRule(clusters=clusters, graph=graph, d_max=d_max)
    return MetropolisSampler(hilbert, rule, **kwargs)


def MetropolisHamiltonian(hilbert, hamiltonian, **kwargs) -> MetropolisSampler:
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
        hilbert: The Hilbert space to sample.
        hamiltonian: The operator used to perform off-diagonal transition.
        n_chains: The total number of independent Markov chains across all devices. Either specify this
            or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every devices (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in
            the Hilbert space. This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new
            sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the
            pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).

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
       >>> # Construct a MetropolisHamiltonian Sampler
       >>> sa = nk.sampler.MetropolisHamiltonian(hi, hamiltonian=ha)
       >>> print(sa)
       MetropolisSampler(rule = HamiltonianRuleJax(operator=IsingJax(J=1.0, h=1.0; dim=100)), n_chains = 16, sweep_size = 100, reset_chains = False, machine_power = 2, dtype = int8)
    """
    from .rules import HamiltonianRule

    rule = HamiltonianRule(hamiltonian)
    return MetropolisSampler(hilbert, rule, **kwargs)


def MetropolisGaussian(hilbert, sigma=1.0, **kwargs) -> MetropolisSampler:
    """This sampler acts on all particle positions simultaneously
    and proposes a new state according to a Gaussian distribution
    with width `sigma`.

    Args:
        hilbert: The continuous Hilbert space to sample.
        sigma: The width of the Gaussian proposal distribution (default = 1.0).
        n_chains: The total number of independent Markov chains across all JAX devices. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every JAX device (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    from netket.experimental.hilbert import ContinuousHilbert

    if not isinstance(hilbert, ContinuousHilbert):
        raise ValueError("This sampler only works for Continuous Hilbert spaces.")

    from .rules import GaussianRule

    rule = GaussianRule(sigma)
    return MetropolisSampler(hilbert, rule, **kwargs)


def MetropolisAdjustedLangevin(
    hilbert, dt=0.001, chunk_size=None, **kwargs
) -> MetropolisSampler:
    r"""This sampler acts on all particle positions simultaneously
    and takes a Langevin step [1]:

    .. math::
       x_{t+dt} = x_t + dt \nabla_x \log p(x) \vert_{x=x_t} + \sqrt{2 dt}\eta,

    where  :math:`\eta` is normal distributed noise :math:`\eta \sim \mathcal{N}(0,1)`.
    This sampler only works for continuous Hilbert spaces.

    [1]: https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm

    Args:
        hilbert: The continuous Hilbert space to sample.
        dt: Time step size for the Langevin dynamics (noise with variance 2*dt).
        chunk_size: Chunk size to compute the gradients of the log probability.
        n_chains: The total number of independent Markov chains across all JAX devices. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every JAX device (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    from netket.experimental.hilbert import ContinuousHilbert

    if not isinstance(hilbert, ContinuousHilbert):
        raise ValueError("This sampler only works for Continuous Hilbert spaces.")

    from .rules import LangevinRule

    rule = LangevinRule(dt=dt, chunk_size=chunk_size)
    return MetropolisSampler(hilbert, rule, **kwargs)


def MetropolisFermionHop(
    hilbert,
    *,
    clusters=None,
    graph=None,
    d_max=1,
    spin_symmetric=True,
    dtype=np.int8,
    **kwargs,
) -> MetropolisSampler:
    r"""
    This sampler moves (or hops) a random particle to a different but random empty mode.
    It works similar to MetropolisExchange, but only allows exchanges between occupied and unoccupied modes.

    Args:
        hilbert: The Hilbert space to sample.
        d_max: The maximum graph distance allowed for exchanges.
        spin_symmetric: (default True) If True, exchanges are only allowed between modes with the same spin projection.
        n_chains: The total number of independent Markov chains across all JAX devices. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every JAX device (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.int8).
    """
    from .rules import FermionHopRule

    rule = FermionHopRule(
        hilbert,
        clusters=clusters,
        graph=graph,
        d_max=d_max,
        spin_symmetric=spin_symmetric,
    )
    return MetropolisSampler(hilbert, rule, dtype=dtype, **kwargs)
