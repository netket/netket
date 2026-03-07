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

import flax
import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import netket as nk
from netket import jax as nkjax
from netket import config
from netket.hilbert import DiscreteHilbert
from netket.sampler import Sampler, SamplerState
from netket.utils.types import PRNGKeyT, DType

from netket.utils import struct
from netket.hilbert.constraint import SumConstraint, SumOnPartitionConstraint

#
#   Internal utils


def _pop_cache(variables: dict):
    """
    Split a Flax variable tree into non-cache variables and the optional cache.

    Args:
        variables: Model variables dictionary, potentially containing a :code:`"cache"` entry.

    Returns a tuple :code:`(variables_no_cache, cache_or_None)` where:
        - ``variables_no_cache`` is safe to pass repeatedly to ``model.apply``.
        - ``cache_or_None`` is the mutable recurrent/attention cache when present.
    """
    if "cache" in variables:
        variables, cache = flax.core.pop(variables, "cache")
        return variables, cache
    return variables, None


def _with_cache(variables_no_cache: dict, cache):
    """
    Reassemble a variable tree expected by ``model.apply``.

    This helper keeps cache handling centralized and avoids repeating
    ``{**variables, "cache": cache}`` in every sampling branch.
    """
    if cache is not None:
        return {**variables_no_cache, "cache": cache}
    return variables_no_cache


def _as_static_tuple(x) -> tuple:
    """Convert an array-like to a plain Python tuple (for pytree_node=False fields)."""
    return tuple(float(v) for v in jnp.asarray(x).reshape(-1).tolist())


def _as_static_int_tuple(x) -> tuple[int, ...]:
    """Convert an array-like to a plain Python int tuple."""
    return tuple(int(v) for v in jnp.asarray(x).reshape(-1).tolist())


def _log_prob_dtype():
    """Floating dtype used for log-probability accumulation buffers."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _ordered_site_tuple(
    model, variables, site_indices: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Compute site order as a hashable Python tuple for static-index JIT kernels.
    """
    idx = jnp.asarray(site_indices, dtype=jnp.int32)
    try:
        ordered = model.apply(variables, idx, method=model.reorder)
        return tuple(int(v) for v in jnp.asarray(ordered).reshape(-1).tolist())
    except (AttributeError, TypeError):
        return tuple(int(v) for v in site_indices)


def _is_uniform_spacing(local_states, atol: float = 1.0e-12) -> tuple[bool, float]:
    """
    Check whether local states from a Hilbert space form a uniform arithmetic progression.

    For such alphabets, feasibility of remaining-sum constraints can be checked
    by range + lattice tests, which is the key to fast exact prefix masking.
    """
    # Sort to compute consecutive spacings robustly.
    xs = jnp.sort(jnp.asarray(local_states, dtype=jnp.float64))

    if xs.size <= 1:
        return True, 1.0

    diffs = jnp.diff(xs)
    step = diffs[0]
    ok = jnp.all(jnp.abs(diffs - step) <= atol)
    return bool(ok), float(step)


def _normalise_masked(
    probabilities: jnp.ndarray, mask: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply a boolean mask to categorical probabilities and renormalize row-wise.

    Args:
        p_norm: Row-normalized masked probabilities.
        feasible: Per-row boolean indicating whether at least one masked entry existed.
    """
    # Zero out infeasible candidates.
    masked = jnp.where(mask, probabilities, 0.0)

    # Partition function per row.
    z = jnp.sum(masked, axis=-1, keepdims=True)

    # Row feasibility flag (at least one feasible candidate).
    feasible = z[:, 0] > 0.0

    # Keep numerical path finite even for infeasible rows, the caller tracks `feasible`.
    p_norm = jnp.where(z > 0.0, masked / z, probabilities)
    return p_norm, feasible


def _sum_feasible_mask(
    *,
    prefix_sum: jax.Array,
    candidate_vals: jax.Array,
    n_remaining: int,
    target: float,
    local_min: float,
    local_max: float,
    step: float,
    tol: float = 1.0e-12,
) -> jax.Array:
    """
    Feasibility oracle for one sum-constrained update with uniform local spacing.

    For every sample row and candidate local state value, this function checks
    whether the remaining number of sites can still satisfy the global target.
    The check consists of:
        1. Interval feasibility ``[lo, hi]`` from remaining min/max contributions.
        2. Lattice feasibility (if ``step != 0``) from uniform local spacing.
    """
    # Remaining amount required after choosing each candidate value.
    need = target - (prefix_sum[:, None] + candidate_vals[None, :])

    # Reachable interval using remaining sites.
    lo = n_remaining * local_min
    hi = n_remaining * local_max
    in_range = (need >= lo - tol) & (need <= hi + tol)

    # Degenerate step: interval test is sufficient.
    if abs(step) <= tol:
        return in_range

    # Arithmetic-lattice compatibility for uniform discrete alphabets.
    k = (need - lo) / step
    on_lattice = jnp.abs(k - jnp.round(k)) <= tol
    return in_range & on_lattice


#
#   Constraint plan builders


def _plan_sum_constraint(hilbert):
    """
    Build exact prefix-mask data for :class:`~netket.hilbert.constraint.SumConstraint`.

    The last site is treated as a deterministic dependent, but free-site sampling
    is constrained by an exact feasibility mask at each auto-regressive step. This guarantees
    the final dependent is representable in the Hilbert space's :code:`local_states` and preserves exactness.

    Returns None when local states are not uniformly spaced, otherwise a fully static/hashable tuple payload:
        (free_indices_tuple, target, local_min, local_max, local_step, slave_site)
    """
    N = hilbert.size
    target = float(hilbert.constraint.sum_value)
    local_states = jnp.asarray(hilbert.local_states, dtype=jnp.float32)

    # Prefix feasibility mask currently assumes a uniformly spaced local alphabet.
    spacing_ok, step = _is_uniform_spacing(local_states)
    if not spacing_ok:
        return None

    # All independent sites are free.
    free_indices = jnp.arange(N - 1, dtype=jnp.int32)

    # Return all constants required by the sampling kernel.
    return (
        tuple(int(v) for v in free_indices.tolist()),
        float(target),
        float(jnp.min(local_states)),
        float(jnp.max(local_states)),
        float(step),
        int(N - 1),
    )


def _plan_partition_constraint(hilbert):
    """
    Build exact prefix-mask data for NetKet :class:`~netket.hilbert.constraint.SumOnPartitionConstraint`.

    For each partition, the last site is the dependent and all earlier partition
    sites are free. Free-site updates are masked by partition-wise feasibility
    checks, yielding exact constrained sampling with zero rejection.

    Returns None when local states are not uniformly spaced. Otherwise a fully static/hashable tuple payload:
        (
          free_indices_tuple,
          partition_of_site_tuple,
          partition_sizes_tuple,
          partition_targets_tuple,
          partition_slaves_tuple,
          local_min,
          local_max,
          local_step,
        )
    """
    constraint = hilbert.constraint
    sizes = [int(s) for s in constraint.sizes]
    targets = [float(t) for t in constraint.sum_values]
    local_states = jnp.asarray(hilbert.local_states, dtype=jnp.float32)

    # Prefix feasibility mask currently assumes a uniformly spaced local alphabet.
    spacing_ok, step = _is_uniform_spacing(local_states)
    if not spacing_ok:
        return None

    free_list: list[int] = []
    partition_of_site: list[int] = []
    partition_slaves: list[int] = []
    off = 0
    for p, sz in enumerate(sizes):
        # Record partition index for each site in this partition.
        partition_of_site.extend([p] * sz)

        # Last site is a dependent, others are free.
        free_len = sz - 1
        free_list.extend(range(off, off + free_len))
        partition_slaves.append(off + free_len)
        off += sz

    # Return all constants required by the sampling kernel.
    return (
        tuple(int(v) for v in free_list),
        tuple(int(v) for v in partition_of_site),
        tuple(int(v) for v in sizes),
        tuple(float(v) for v in targets),
        tuple(int(v) for v in partition_slaves),
        float(jnp.min(local_states)),
        float(jnp.max(local_states)),
        float(step),
    )


#
#   Plan dispatch

_TWO_PHASE = "two_phase"
_SUM_PREFIX = "sum"
_PARTITION_PREFIX = "partition"
_REJECTION = "rejection"


def _build_plan(hilbert):
    """
    Inspect :code:`hilbert` and return the optimal sampling plan based on its constraint. For
    current NetKet constraints, there exists a concrete fast-path plan. For generic constraints,
    the fallback is rejection sampling, which will be slow.

    Returns

        For NetKet prefix-masked exact sampling::

            ("sum_prefix", data: tuple)
            ("partition_prefix", data: tuple)

        For rejection-based sampling::

            ("rejection", constraint_fn: callable)
    """

    constraint = getattr(hilbert, "constraint", None)

    #
    # SumConstraint
    # Use exact prefix masking when local states allow uniform-lattice tests.
    # Otherwise fall back to rejection for correctness.
    if isinstance(constraint, SumConstraint):
        data = _plan_sum_constraint(hilbert)
        if data is not None:
            return _SUM_PREFIX, data
        # Fallback to rejection sampling, should never happen for this constraint.
        return _REJECTION, constraint

    #
    # SumOnPartitionConstraint
    # Same strategy as SumConstraint: exact prefix masking when feasible,
    # otherwise rejection fallback.
    if isinstance(constraint, SumOnPartitionConstraint):
        data = _plan_partition_constraint(hilbert)
        if data is not None:
            return _PARTITION_PREFIX, data
        return _REJECTION, constraint

    #
    # Unconstrained Hilbert spaces
    if not getattr(hilbert, "constrained", False):
        N = hilbert.size
        return _TWO_PHASE, tuple(range(N)), lambda s: s

    # Any remaining callable generic constraint falls-back to rejection
    if constraint is not None and callable(constraint):
        return _REJECTION, constraint

    raise ValueError(
        f"ConstrainedARDirectSampler: cannot build a sampling plan for\n"
        f"  Hilbert type : {type(hilbert).__name__}\n"
        f"  Constraint   : {type(constraint).__name__ if constraint is not None else 'None'}\n\n"
        "To enable zero-rejection sampling, do ONE of the following:\n"
        "  1. For NetKet SumConstraint / SumOnPartitionConstraint, ensure\n"
        "     local states are uniformly spaced to enable fast exact prefix masking.\n"
        "  2. For any other constraint, make it a JAX-callable (B,N) -> (B,)bool,\n"
        "     the sampler will fall back to a compiled rejection while_loop.\n"
    )


class ARDirectSamplerState(SamplerState):
    key: PRNGKeyT
    """state of the random number generator."""

    def __init__(self, key):
        self.key = key
        super().__init__()


class UnconstrainedARDirectSampler(Sampler):
    r"""
    Direct sampler for autoregressive neural networks.

    This sampler only works with Flax models.
    This flax model must expose a specific method, `model.conditional`, which given
    a batch of samples and an index `i∈[0,self.hilbert.size]` must return the vector
    of partial probabilities at index `i` for the various (partial) samples provided.

    In short, if your model can be sampled according to a probability
    $ p(x) = p_1(x_1)p_2(x_2|x_1)\dots p_N(x_N|x_{N-1}\dots x_1) $ then
    `model.conditional(x, i)` should return $p_i(x)$.

    NetKet implements some autoregressive networks that can be used together with this
    sampler.
    """

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        machine_pow: None = None,
        dtype: DType = None,
    ):
        """
        Construct an autoregressive direct sampler.

        Args:
            hilbert: The Hilbert space to sample.
            dtype: The dtype of the states sampled (default = np.float64).

        Note:
            `ARDirectSampler.machine_pow` has no effect. Please set the model's `machine_pow` instead.
        """

        if machine_pow is not None:
            raise ValueError(
                "ARDirectSampler.machine_pow should not be used. Modify the model `machine_pow` directly."
            )

        if hilbert.constrained:
            raise ValueError(
                "Only unconstrained Hilbert spaces can be sampled autoregressively with "
                "this sampler. To sample constrained spaces, you must write your own (do get in "
                "touch with us. We are interested!)"
            )

        super().__init__(hilbert, machine_pow=2, dtype=dtype)
        # ensure machine_pow is a float, as it can be sometimes used around...
        self.machine_pow = float(self.machine_pow)

    @property
    def is_exact(sampler):
        """
        Returns `True` because the sampler is exact.

        The sampler is exact if all the samples are exactly distributed according to the
        chosen power of the variational state, and there is no correlation among them.
        """
        return True

    def _init_cache(self, model, σ, key):
        variables = model.init(key, σ, 0, method=model.conditional)
        cache = variables.get("cache")
        return cache

    def _init_state(self, model, variables, key):
        return ARDirectSamplerState(key=key)

    def _reset(self, model, variables, state):
        return state

    @partial(
        jax.jit, static_argnames=("model", "chain_length", "return_log_probabilities")
    )
    def _sample_chain(
        self,
        model,
        variables,
        state,
        chain_length,
        return_log_probabilities: bool = False,
    ):
        if "cache" in variables:
            variables, _ = flax.core.pop(variables, "cache")
        variables_no_cache = variables

        def scan_fun(carry, index):
            σ, cache, key, log_prob = carry
            if cache:
                variables = {**variables_no_cache, "cache": cache}
            else:
                variables = variables_no_cache
            new_key, key = jax.random.split(key)

            p, mutables = model.apply(
                variables,
                σ,
                index,
                method=model.conditional,
                mutable=["cache"],
            )
            cache = mutables.get("cache")

            local_states = jnp.asarray(self.hilbert.local_states, dtype=self.dtype)

            if return_log_probabilities:
                new_σ, new_p = nkjax.batch_choice(
                    key, local_states, p, return_prob=True
                )
                log_prob = log_prob + jnp.log(new_p)
            else:
                new_σ = nkjax.batch_choice(key, local_states, p)

            σ = σ.at[:, index].set(new_σ)

            return (σ, cache, new_key, log_prob), None

        new_key, key_init, key_scan = jax.random.split(state.key, 3)

        # Initialize a buffer for `σ` before generating a batch of samples
        σ = jnp.zeros(
            (self.n_batches * chain_length, self.hilbert.size),
            dtype=self.dtype,
        )

        if config.netket_experimental_sharding:
            σ = jax.lax.with_sharding_constraint(
                σ,
                NamedSharding(jax.sharding.get_abstract_mesh(), P("S")),
            )

        # Initialize `cache` before generating a batch of samples
        cache = self._init_cache(model, σ, key_init)
        if cache:
            variables = {**variables_no_cache, "cache": cache}
        else:
            variables = variables_no_cache

        indices = jnp.arange(self.hilbert.size)
        indices = model.apply(variables, indices, method=model.reorder)

        if return_log_probabilities:
            log_prob = jnp.zeros((self.n_batches * chain_length,))
        else:
            log_prob = None

        (σ, _, _, log_prob), _ = jax.lax.scan(
            scan_fun, (σ, cache, key_scan, log_prob), indices
        )
        σ = σ.reshape((self.n_batches, chain_length, self.hilbert.size))

        new_state = state.replace(key=new_key)

        if return_log_probabilities:
            log_prob = log_prob.reshape((self.n_batches, chain_length))
            return (σ, log_prob), new_state
        else:
            return σ, new_state


class ConstrainedARDirectSampler(Sampler):
    r"""
    Base class for constrained direct samplers for autoregressive neural networks.

    This sampler family only works with Flax models exposing
    ``model.conditional``. Given a batch of partial samples and an index
    ``i ∈ [0, self.hilbert.size)``, ``model.conditional`` must return the
    conditional probabilities over local states at site ``i``.

    In short, if your model admits a factorization

    .. math::

        p(x) = p_1(x_1) p_2(x_2|x_1)\dots p_N(x_N|x_{N-1}, \dots, x_1),

    then ``model.conditional(x, i)`` should return :math:`p_i(x)`.

    Subclasses implement concrete constrained strategies:
    - exact prefix-feasibility masking for known NetKet constraints,
    - exact rejection sampling fallback for generic callable constraints.
    """

    max_resampling_attempts: int = struct.field(pytree_node=False, default=4096)
    """Maximum number of rejection-loop iterations before raising a RuntimeError."""

    _local_states: tuple = struct.field(pytree_node=False, default=())
    """The allowed local states in the Hilbert space being sampled."""

    def __init__(
        self,
        hilbert,
        machine_pow=None,
        dtype=None,
        *,
        max_resampling_attempts: int = 4096,
    ):
        """
        Construct a constrained autoregressive sampler.

        Args:
            hilbert: The Hilbert space to sample.
            dtype: The dtype of the states sampled (default = np.float64).
            max_resampling_attempts: Maximum number of rejection-loop iterations before raising
              ``RuntimeError`` (applies only to the fallback rejection path).
        """
        if machine_pow is not None:
            raise ValueError(
                "ARDirectSampler.machine_pow should not be used. Modify the model `machine_pow` directly."
            )

        super().__init__(hilbert, machine_pow=2, dtype=dtype)

        # ensure machine_pow is a float, as it can be sometimes used around...
        self.machine_pow = float(self.machine_pow)

        self.max_resampling_attempts = int(max_resampling_attempts)
        self._local_states = _as_static_tuple(
            jnp.asarray(self.hilbert.local_states, dtype=self.dtype)
        )

    @property
    def is_exact(self) -> bool:
        """
        Returns `True` because the sampler is exact.

        The sampler is exact if all the samples are exactly distributed according to the
        chosen power of the variational state, and there is no correlation among them.
        """
        return True

    def _local_states_array(self) -> jnp.ndarray:
        """Returns the local states of the Hilbert space being sampled."""
        return jnp.asarray(self._local_states, dtype=self.dtype)

    def _init_cache(self, model, σ, key):
        variables = model.init(key, σ, 0, method=model.conditional)
        cache = variables.get("cache")
        return cache

    def _init_state(self, model, variables, key) -> ARDirectSamplerState:
        return ARDirectSamplerState(key=key)

    def _reset(self, model, variables, state) -> ARDirectSamplerState:
        return state

    @partial(
        jax.jit,
        static_argnames=("model", "chain_length", "return_log_probabilities"),
    )
    def _sampling_kernel_model_direct(
        self,
        model,
        variables,
        state,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        """
        Generic direct AR kernel that follows model conditionals over all sites.
        """
        variables_no_cache, _ = _pop_cache(variables)

        total = self.n_batches * chain_length
        n_sites = self.hilbert.size
        local_states = self._local_states_array()

        new_key, key_init, key_scan = jax.random.split(state.key, 3)
        sigma0 = jnp.zeros((total, n_sites), dtype=self.dtype)
        cache0 = self._init_cache(model, sigma0, key_init)

        try:
            site_order = model.apply(
                _with_cache(variables_no_cache, cache0),
                jnp.arange(n_sites, dtype=jnp.int32),
                method=model.reorder,
            ).astype(jnp.int32)
        except (AttributeError, TypeError):
            site_order = jnp.arange(n_sites, dtype=jnp.int32)

        if return_log_probabilities:
            logp0 = jnp.zeros((total,), dtype=_log_prob_dtype())

            def _step(carry, site_i):
                sigma, cache, key, logp = carry
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                new_col, new_p = nkjax.batch_choice(
                    key_draw, local_states, p, return_prob=True
                )
                sigma = sigma.at[:, site_i].set(new_col)
                logp = logp + jnp.log(new_p).astype(logp.dtype)
                return (sigma, cache, key, logp), None

            (sigma, _cache, _key, logp), _ = jax.lax.scan(
                _step,
                (sigma0, cache0, key_scan, logp0),
                site_order,
            )
        else:
            logp = None

            def _step(carry, site_i):
                sigma, cache, key = carry
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                new_col = nkjax.batch_choice(key_draw, local_states, p)
                sigma = sigma.at[:, site_i].set(new_col)
                return (sigma, cache, key), None

            (sigma, _cache, _key), _ = jax.lax.scan(
                _step,
                (sigma0, cache0, key_scan),
                site_order,
            )

        sigma = sigma.reshape((self.n_batches, chain_length, n_sites))
        valid = self.hilbert.constraint(sigma.reshape(total, n_sites)).reshape(
            (self.n_batches, chain_length)
        )
        all_accepted = jnp.all(valid)
        new_state = state.replace(key=new_key)

        if return_log_probabilities:
            return (
                (sigma, logp.reshape(self.n_batches, chain_length)),
                new_state,
                all_accepted,
            )
        return sigma, new_state, all_accepted

    @partial(
        jax.jit,
        static_argnames=(
            "model",
            "chain_length",
            "site_order",
            "return_log_probabilities",
        ),
    )
    def _sampling_kernel_model_direct_static(
        self,
        model,
        variables,
        state,
        chain_length: int,
        *,
        site_order: tuple[int, ...],
        return_log_probabilities: bool = False,
    ):
        """
        Static-index variant of `_sampling_kernel_model_direct` for FastARNN models.
        """
        variables_no_cache, _ = _pop_cache(variables)

        total = self.n_batches * chain_length
        n_sites = self.hilbert.size
        local_states = self._local_states_array()

        new_key, key_init, key_scan = jax.random.split(state.key, 3)
        sigma = jnp.zeros((total, n_sites), dtype=self.dtype)
        cache = self._init_cache(model, sigma, key_init)

        key = key_scan
        if return_log_probabilities:
            logp = jnp.zeros((total,), dtype=_log_prob_dtype())
            for site_i in site_order:
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                new_col, new_p = nkjax.batch_choice(
                    key_draw, local_states, p, return_prob=True
                )
                sigma = sigma.at[:, site_i].set(new_col)
                logp = logp + jnp.log(new_p).astype(logp.dtype)
        else:
            logp = None
            for site_i in site_order:
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                new_col = nkjax.batch_choice(key_draw, local_states, p)
                sigma = sigma.at[:, site_i].set(new_col)

        sigma = sigma.reshape((self.n_batches, chain_length, n_sites))
        valid = self.hilbert.constraint(sigma.reshape(total, n_sites)).reshape(
            (self.n_batches, chain_length)
        )
        all_accepted = jnp.all(valid)
        new_state = state.replace(key=new_key)

        if return_log_probabilities:
            return (
                (sigma, logp.reshape(self.n_batches, chain_length)),
                new_state,
                all_accepted,
            )
        return sigma, new_state, all_accepted

    def _sample_chain_model_direct(
        self,
        model,
        variables,
        state,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        """
        Host-side dispatcher for model-direct kernels with FastARNN handling.
        """
        if isinstance(model, nk.models.FastARNNSequential):
            variables_no_cache, _ = _pop_cache(variables)
            site_order = _ordered_site_tuple(
                model,
                variables_no_cache,
                tuple(range(self.hilbert.size)),
            )
            out, new_state, all_accepted = self._sampling_kernel_model_direct_static(
                model,
                variables,
                state,
                chain_length,
                site_order=site_order,
                return_log_probabilities=return_log_probabilities,
            )
        else:
            out, new_state, all_accepted = self._sampling_kernel_model_direct(
                model,
                variables,
                state,
                chain_length,
                return_log_probabilities=return_log_probabilities,
            )
        self._raise_if_not_all_accepted(all_accepted)
        return out, new_state

    def _raise_if_not_all_accepted(self, all_accepted):
        """
        Raise a clear host-side error when rejection-based paths did not fill all rows.
        """
        if not bool(all_accepted):
            raise RuntimeError(
                "ConstrainedARDirectSampler: rejection sampling failed to accept "
                f"all samples after {self.max_resampling_attempts} attempts. "
                "Consider increasing max_resampling_attempts."
            )

    def _sample_chain(
        self,
        model,
        variables,
        state,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        """
        A wrapper around subclass JIT kernels.

        The wrapper intentionally performs only one host-side action, converting the scalar acceptance
        flag to bool and raising a clear error if fallback rejection failed to fill all sample slots.
        """
        out, new_state, all_accepted = self._sampling_kernel(
            model,
            variables,
            state,
            chain_length,
            return_log_probabilities=return_log_probabilities,
        )

        self._raise_if_not_all_accepted(all_accepted)

        return out, new_state


class SumConstrainedARDirectSampler(ConstrainedARDirectSampler):
    r"""
    Direct sampler for autoregressive models on Hilbert spaces constrained with a
    :class:`~netket.hilbert.constraint.SumConstraint` constraint.

    This sampler uses an exact prefix-feasibility masking strategy.
    For each autoregressive site update, candidate local values that cannot be
    completed to the target sum are masked out before sampling. After all free
    sites are sampled, one dependent site is set deterministically to enforce
    the exact final sum. The method is rejection-free (for supported local-state values).
    """

    _free_indices: tuple[int, ...] = struct.field(pytree_node=False, default=())
    """Indices sampled autoregressively before deterministic sum closure."""

    _target: float = struct.field(pytree_node=False, default=0.0)
    """Target total sum enforced by ``SumConstraint``."""

    _local_min: float = struct.field(pytree_node=False, default=0.0)
    """Minimum value in the local state alphabet (used for feasibility bounds)."""

    _local_max: float = struct.field(pytree_node=False, default=0.0)
    """Maximum value in the local state alphabet (used for feasibility bounds)."""

    _local_step: float = struct.field(pytree_node=False, default=1.0)
    """Uniform spacing step of local states for lattice-feasibility checks."""

    _slave_site: int = struct.field(pytree_node=False, default=0)
    """Index of the deterministic dependent site set after free-site sampling."""

    def __init__(
        self,
        hilbert,
        payload,
        machine_pow=None,
        dtype=None,
        *,
        max_resampling_attempts: int = 4096,
    ):
        # Shared sampler initialization.
        super().__init__(
            hilbert,
            machine_pow=machine_pow,
            dtype=dtype,
            max_resampling_attempts=max_resampling_attempts,
        )

        # Unpack static payload produced by the plan builder.
        (
            free_indices,
            target,
            local_min,
            local_max,
            local_step,
            slave_site,
        ) = payload

        # Store all payload items as Python scalars/tuples for stable metadata.
        self._free_indices = tuple(int(v) for v in free_indices)
        self._target = float(target)
        self._local_min = float(local_min)
        self._local_max = float(local_max)
        self._local_step = float(local_step)
        self._slave_site = int(slave_site)

    def _sample_chain(
        self,
        model,
        variables,
        state,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        """
        Dispatch to a static-index kernel for FastARNN models.
        """
        if getattr(model, "_constraint_aware_autoregressive", False):
            return self._sample_chain_model_direct(
                model,
                variables,
                state,
                chain_length,
                return_log_probabilities=return_log_probabilities,
            )

        if isinstance(model, nk.models.FastARNNSequential):
            variables_no_cache, _ = _pop_cache(variables)
            site_order = _ordered_site_tuple(
                model, variables_no_cache, self._free_indices
            )
            out, new_state, all_accepted = self._sampling_kernel_static(
                model,
                variables,
                state,
                chain_length,
                site_order=site_order,
                return_log_probabilities=return_log_probabilities,
            )
            self._raise_if_not_all_accepted(all_accepted)
            return out, new_state

        return super()._sample_chain(
            model,
            variables,
            state,
            chain_length,
            return_log_probabilities=return_log_probabilities,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "model",
            "chain_length",
            "site_order",
            "return_log_probabilities",
        ),
    )
    def _sampling_kernel_static(
        self,
        model,
        variables,
        state,
        chain_length: int,
        *,
        site_order: tuple[int, ...],
        return_log_probabilities: bool = False,
    ):
        """
        Static-index constrained sum kernel, compatible with FastARNN conditionals.
        """
        variables_no_cache, _ = _pop_cache(variables)

        total = self.n_batches * chain_length
        n_sites = self.hilbert.size
        local_states = self._local_states_array()
        local_states_f32 = local_states.astype(jnp.float32)

        new_key, key_init, key_ar = jax.random.split(state.key, 3)
        sigma = jnp.zeros((total, n_sites), dtype=self.dtype)
        cache = self._init_cache(model, sigma, key_init)

        prefix_sum = jnp.zeros((total,), dtype=jnp.float32)
        feasible = jnp.ones((total,), dtype=jnp.bool_)
        key = key_ar

        if return_log_probabilities:
            logp = jnp.zeros((total,), dtype=_log_prob_dtype())
            for step_i, site_i in enumerate(site_order):
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                n_remaining = len(site_order) - step_i
                mask = _sum_feasible_mask(
                    prefix_sum=prefix_sum,
                    candidate_vals=local_states_f32,
                    n_remaining=n_remaining,
                    target=self._target,
                    local_min=self._local_min,
                    local_max=self._local_max,
                    step=self._local_step,
                )
                p_masked, row_feasible = _normalise_masked(p, mask)
                feasible = feasible & row_feasible
                new_col, new_p = nkjax.batch_choice(
                    key_draw,
                    local_states,
                    p_masked,
                    return_prob=True,
                )
                sigma = sigma.at[:, site_i].set(new_col)
                prefix_sum = prefix_sum + new_col.astype(jnp.float32)
                logp = logp + jnp.log(new_p).astype(logp.dtype)
        else:
            logp = None
            for step_i, site_i in enumerate(site_order):
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                n_remaining = len(site_order) - step_i
                mask = _sum_feasible_mask(
                    prefix_sum=prefix_sum,
                    candidate_vals=local_states_f32,
                    n_remaining=n_remaining,
                    target=self._target,
                    local_min=self._local_min,
                    local_max=self._local_max,
                    step=self._local_step,
                )
                p_masked, row_feasible = _normalise_masked(p, mask)
                feasible = feasible & row_feasible
                new_col = nkjax.batch_choice(key_draw, local_states, p_masked)
                sigma = sigma.at[:, site_i].set(new_col)
                prefix_sum = prefix_sum + new_col.astype(jnp.float32)

        slave = self._target - prefix_sum
        dists = jnp.abs(slave[:, None] - local_states_f32[None, :])
        idx = jnp.argmin(dists, axis=-1)
        slave_vals = local_states_f32[idx]
        sigma = sigma.at[:, self._slave_site].set(slave_vals.astype(self.dtype))
        slave_exact = jnp.abs(slave_vals - slave) <= 1.0e-6
        all_accepted = jnp.all(feasible & slave_exact)

        sigma = sigma.reshape(self.n_batches, chain_length, n_sites)
        new_state = state.replace(key=new_key)
        if return_log_probabilities:
            return (
                (sigma, logp.reshape(self.n_batches, chain_length)),
                new_state,
                all_accepted,
            )
        return sigma, new_state, all_accepted

    @partial(
        jax.jit, static_argnames=("model", "chain_length", "return_log_probabilities")
    )
    def _sampling_kernel(
        self,
        model,
        variables,
        state,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        # Split model variables and isolate mutable cache handling.
        variables_no_cache, _ = _pop_cache(variables)

        # Flattened sample count and static local alphabet.
        total = self.n_batches * chain_length
        n_sites = self.hilbert.size
        local_states = self._local_states_array()

        # PRNG split for output state, cache init, and AR draws.
        new_key, key_init, key_ar = jax.random.split(state.key, 3)
        sigma_seed = jnp.zeros((total, n_sites), dtype=self.dtype)

        # Initialize model cache once for the target batch shape.
        cache0 = self._init_cache(model, sigma_seed, key_init)
        free_indices = jnp.asarray(self._free_indices, dtype=jnp.int32)

        # Allow model-specific site reordering.
        try:
            ordered_free = model.apply(
                _with_cache(variables_no_cache, cache0),
                free_indices,
                method=model.reorder,
            ).astype(jnp.int32)
        except (AttributeError, TypeError):
            ordered_free = free_indices

        # Scan input carries both the step index (for remaining-site arithmetic)
        # and the actual site index to sample.
        scan_steps = (
            jnp.arange(free_indices.shape[0], dtype=jnp.int32),
            ordered_free,
        )

        # Feasibility arithmetic is done in float32 for speed and consistency.
        local_states_f32 = local_states.astype(jnp.float32)

        if return_log_probabilities:
            # Per-sample running state:
            # - logp: accumulated log conditional probability
            # - prefix_sum: running sum of already sampled free sites
            # - feasible: whether at least one feasible candidate remained at each step
            logp0 = jnp.zeros((total,), dtype=_log_prob_dtype())
            prefix_sum0 = jnp.zeros((total,), dtype=jnp.float32)
            feasible0 = jnp.ones((total,), dtype=jnp.bool_)

            def _step_sum_logp(carry, scan_in):
                sigma, cache, key, logp, prefix_sum, feasible = carry
                step_i, site_i = scan_in

                # Query unconstrained conditional probabilities.
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                n_remaining = free_indices.shape[0] - step_i

                # Build exact feasibility mask for every candidate local value.
                mask = _sum_feasible_mask(
                    prefix_sum=prefix_sum,
                    candidate_vals=local_states_f32,
                    n_remaining=n_remaining,
                    target=self._target,
                    local_min=self._local_min,
                    local_max=self._local_max,
                    step=self._local_step,
                )

                # Renormalize categorical probabilities over feasible support only.
                p_masked, row_feasible = _normalise_masked(p, mask)
                feasible = feasible & row_feasible

                # Sample constrained conditional and update running statistics.
                new_col, new_p = nkjax.batch_choice(
                    key_draw,
                    local_states,
                    p_masked,
                    return_prob=True,
                )
                sigma = sigma.at[:, site_i].set(new_col)
                prefix_sum = prefix_sum + new_col.astype(jnp.float32)
                logp = logp + jnp.log(new_p).astype(logp.dtype)
                return (sigma, cache, key, logp, prefix_sum, feasible), None

            # One fused scan over all free sites.
            (sigma, _cache, _key, logp, prefix_sum, feasible), _ = jax.lax.scan(
                _step_sum_logp,
                (sigma_seed, cache0, key_ar, logp0, prefix_sum0, feasible0),
                scan_steps,
            )
        else:
            # Same state as above without explicit log-probability tracking.
            prefix_sum0 = jnp.zeros((total,), dtype=jnp.float32)
            feasible0 = jnp.ones((total,), dtype=jnp.bool_)

            def _step_sum(carry, scan_in):
                sigma, cache, key, prefix_sum, feasible = carry
                step_i, site_i = scan_in

                # Query unconstrained conditional probabilities.
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                n_remaining = free_indices.shape[0] - step_i

                # Mask infeasible values and renormalize on feasible support.
                mask = _sum_feasible_mask(
                    prefix_sum=prefix_sum,
                    candidate_vals=local_states_f32,
                    n_remaining=n_remaining,
                    target=self._target,
                    local_min=self._local_min,
                    local_max=self._local_max,
                    step=self._local_step,
                )
                p_masked, row_feasible = _normalise_masked(p, mask)
                feasible = feasible & row_feasible

                # Draw constrained sample and update prefix sum.
                new_col = nkjax.batch_choice(key_draw, local_states, p_masked)
                sigma = sigma.at[:, site_i].set(new_col)
                prefix_sum = prefix_sum + new_col.astype(jnp.float32)
                return (sigma, cache, key, prefix_sum, feasible), None

            # One fused scan over all free sites.
            (sigma, _cache, _key, prefix_sum, feasible), _ = jax.lax.scan(
                _step_sum,
                (sigma_seed, cache0, key_ar, prefix_sum0, feasible0),
                scan_steps,
            )
            logp = None  # type: ignore[assignment]

        # Final deterministic slave-site closure to satisfy exact sum target.
        slave = self._target - prefix_sum
        dists = jnp.abs(slave[:, None] - local_states_f32[None, :])
        idx = jnp.argmin(dists, axis=-1)
        slave_vals = local_states_f32[idx]
        sigma = sigma.at[:, self._slave_site].set(slave_vals.astype(self.dtype))

        # Accept only if all rows stayed feasible and slave was exactly representable.
        slave_exact = jnp.abs(slave_vals - slave) <= 1.0e-6
        all_accepted = jnp.all(feasible & slave_exact)

        # Restore NetKet sample shape and return new sampler key.
        sigma = sigma.reshape(self.n_batches, chain_length, n_sites)
        new_state = state.replace(key=new_key)

        if return_log_probabilities:
            return (
                (sigma, logp.reshape(self.n_batches, chain_length)),
                new_state,
                all_accepted,
            )
        return sigma, new_state, all_accepted


class PartitionConstrainedARDirectSampler(ConstrainedARDirectSampler):
    r"""
    Direct sampler for autoregressive models with Hilbert spaces constrained with a
    :class:`~netket.hilbert.constraint.SumOnPartitionConstraint` constraint.

    This sampler generalizes the sum-prefix strategy to multiple partitions.
    During autoregressive updates, candidate values are masked using partition-
    wise feasibility checks so that every partition target remains reachable.
    After all free sites are sampled, one dependent site per partition is set
    deterministically to close each partition sum exactly. For supported local-state
    values this path is rejection-free and exact.
    """

    _free_indices: tuple[int, ...] = struct.field(pytree_node=False, default=())
    """Global indices of all free sites across partitions."""

    _partition_of_site: tuple[int, ...] = struct.field(pytree_node=False, default=())
    """Map from global site index to partition id."""

    _partition_sizes: tuple[int, ...] = struct.field(pytree_node=False, default=())
    """Number of sites in each partition."""

    _partition_targets: tuple[float, ...] = struct.field(pytree_node=False, default=())
    """Target sum for each partition."""

    _partition_slaves: tuple[int, ...] = struct.field(pytree_node=False, default=())
    """Dependent site index used to close each partition sum exactly."""

    _local_min: float = struct.field(pytree_node=False, default=0.0)
    """Minimum value in the local state alphabet."""

    _local_max: float = struct.field(pytree_node=False, default=0.0)
    """Maximum value in the local state alphabet."""

    _local_step: float = struct.field(pytree_node=False, default=1.0)
    """Uniform spacing step of local states for lattice-feasibility checks."""

    def __init__(
        self,
        hilbert,
        payload,
        machine_pow=None,
        dtype=None,
        *,
        max_resampling_attempts: int = 4096,
    ):
        # Shared sampler initialization.
        super().__init__(
            hilbert,
            machine_pow=machine_pow,
            dtype=dtype,
            max_resampling_attempts=max_resampling_attempts,
        )
        # Unpack static payload from the planner.
        (
            free_indices,
            partition_of_site,
            partition_sizes,
            partition_targets,
            partition_slaves,
            local_min,
            local_max,
            local_step,
        ) = payload
        # Keep payload metadata as Python tuples/scalars for stable JAX metadata.
        self._free_indices = tuple(int(v) for v in free_indices)
        self._partition_of_site = tuple(int(v) for v in partition_of_site)
        self._partition_sizes = tuple(int(v) for v in partition_sizes)
        self._partition_targets = tuple(float(v) for v in partition_targets)
        self._partition_slaves = tuple(int(v) for v in partition_slaves)
        self._local_min = float(local_min)
        self._local_max = float(local_max)
        self._local_step = float(local_step)

    def _sample_chain(
        self,
        model,
        variables,
        state,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        """
        Dispatch to a static-index kernel for FastARNN models.
        """
        if getattr(model, "_constraint_aware_autoregressive", False):
            return self._sample_chain_model_direct(
                model,
                variables,
                state,
                chain_length,
                return_log_probabilities=return_log_probabilities,
            )

        if isinstance(model, nk.models.FastARNNSequential):
            variables_no_cache, _ = _pop_cache(variables)
            site_order = _ordered_site_tuple(
                model, variables_no_cache, self._free_indices
            )
            out, new_state, all_accepted = self._sampling_kernel_static(
                model,
                variables,
                state,
                chain_length,
                site_order=site_order,
                return_log_probabilities=return_log_probabilities,
            )
            self._raise_if_not_all_accepted(all_accepted)
            return out, new_state

        return super()._sample_chain(
            model,
            variables,
            state,
            chain_length,
            return_log_probabilities=return_log_probabilities,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "model",
            "chain_length",
            "site_order",
            "return_log_probabilities",
        ),
    )
    def _sampling_kernel_static(
        self,
        model,
        variables,
        state,
        chain_length: int,
        *,
        site_order: tuple[int, ...],
        return_log_probabilities: bool = False,
    ):
        """
        Static-index constrained partition kernel, compatible with FastARNN.
        """
        variables_no_cache, _ = _pop_cache(variables)

        total = self.n_batches * chain_length
        n_sites = self.hilbert.size
        local_states = self._local_states_array()
        local_states_f32 = local_states.astype(jnp.float32)

        new_key, key_init, key_ar = jax.random.split(state.key, 3)
        sigma = jnp.zeros((total, n_sites), dtype=self.dtype)
        cache = self._init_cache(model, sigma, key_init)

        pnum = len(self._partition_sizes)
        partition_of_site = jnp.asarray(self._partition_of_site, dtype=jnp.int32)
        partition_sizes = jnp.asarray(self._partition_sizes, dtype=jnp.int32)
        partition_targets = jnp.asarray(self._partition_targets, dtype=jnp.float32)
        partition_slaves = tuple(int(v) for v in self._partition_slaves)
        step_is_zero = abs(self._local_step) <= 1.0e-12

        prefix_sums = jnp.zeros((total, pnum), dtype=jnp.float32)
        done_counts = jnp.zeros((pnum,), dtype=jnp.int32)
        feasible = jnp.ones((total,), dtype=jnp.bool_)
        key = key_ar

        if return_log_probabilities:
            logp = jnp.zeros((total,), dtype=_log_prob_dtype())
            for site_i in site_order:
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                part = partition_of_site[site_i]
                onehot_p = jax.nn.one_hot(part, pnum, dtype=jnp.float32)[None, None, :]
                sums_new = (
                    prefix_sums[:, None, :] + onehot_p * local_states_f32[None, :, None]
                )
                done_new = done_counts + jax.nn.one_hot(part, pnum, dtype=jnp.int32)
                rem = partition_sizes - done_new
                lo = rem[None, None, :] * self._local_min
                hi = rem[None, None, :] * self._local_max
                need = partition_targets[None, None, :] - sums_new
                in_range = (need >= lo - 1.0e-12) & (need <= hi + 1.0e-12)
                if step_is_zero:
                    mask = jnp.all(in_range, axis=-1)
                else:
                    k = (need - lo) / self._local_step
                    on_lattice = jnp.abs(k - jnp.round(k)) <= 1.0e-12
                    mask = jnp.all(in_range & on_lattice, axis=-1)
                p_masked, row_feasible = _normalise_masked(p, mask)
                feasible = feasible & row_feasible
                new_col, new_p = nkjax.batch_choice(
                    key_draw,
                    local_states,
                    p_masked,
                    return_prob=True,
                )
                sigma = sigma.at[:, site_i].set(new_col)
                prefix_sums = prefix_sums.at[:, part].add(new_col.astype(jnp.float32))
                done_counts = done_new
                logp = logp + jnp.log(new_p).astype(logp.dtype)
        else:
            logp = None
            for site_i in site_order:
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                part = partition_of_site[site_i]
                onehot_p = jax.nn.one_hot(part, pnum, dtype=jnp.float32)[None, None, :]
                sums_new = (
                    prefix_sums[:, None, :] + onehot_p * local_states_f32[None, :, None]
                )
                done_new = done_counts + jax.nn.one_hot(part, pnum, dtype=jnp.int32)
                rem = partition_sizes - done_new
                lo = rem[None, None, :] * self._local_min
                hi = rem[None, None, :] * self._local_max
                need = partition_targets[None, None, :] - sums_new
                in_range = (need >= lo - 1.0e-12) & (need <= hi + 1.0e-12)
                if step_is_zero:
                    mask = jnp.all(in_range, axis=-1)
                else:
                    k = (need - lo) / self._local_step
                    on_lattice = jnp.abs(k - jnp.round(k)) <= 1.0e-12
                    mask = jnp.all(in_range & on_lattice, axis=-1)
                p_masked, row_feasible = _normalise_masked(p, mask)
                feasible = feasible & row_feasible
                new_col = nkjax.batch_choice(key_draw, local_states, p_masked)
                sigma = sigma.at[:, site_i].set(new_col)
                prefix_sums = prefix_sums.at[:, part].add(new_col.astype(jnp.float32))
                done_counts = done_new

        slave_exact = jnp.ones((total,), dtype=jnp.bool_)
        for p, slave_site in enumerate(partition_slaves):
            slave = partition_targets[p] - prefix_sums[:, p]
            dists = jnp.abs(slave[:, None] - local_states_f32[None, :])
            idx = jnp.argmin(dists, axis=-1)
            slave_vals = local_states_f32[idx]
            sigma = sigma.at[:, slave_site].set(slave_vals.astype(self.dtype))
            slave_exact = slave_exact & (jnp.abs(slave_vals - slave) <= 1.0e-6)

        all_accepted = jnp.all(feasible & slave_exact)
        sigma = sigma.reshape(self.n_batches, chain_length, n_sites)
        new_state = state.replace(key=new_key)
        if return_log_probabilities:
            return (
                (sigma, logp.reshape(self.n_batches, chain_length)),
                new_state,
                all_accepted,
            )
        return sigma, new_state, all_accepted

    @partial(
        jax.jit, static_argnames=("model", "chain_length", "return_log_probabilities")
    )
    def _sampling_kernel(
        self,
        model,
        variables,
        state,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        # Split immutable variable tree and mutable cache usage.
        variables_no_cache, _ = _pop_cache(variables)

        # Flattened sample axis and static local alphabet.
        total = self.n_batches * chain_length
        n_sites = self.hilbert.size
        local_states = self._local_states_array()

        # PRNG split for outgoing state key, cache init, and AR draw stream.
        new_key, key_init, key_ar = jax.random.split(state.key, 3)
        sigma_seed = jnp.zeros((total, n_sites), dtype=self.dtype)

        # Initialize model cache once at the target batch shape.
        cache0 = self._init_cache(model, sigma_seed, key_init)
        free_indices = jnp.asarray(self._free_indices, dtype=jnp.int32)

        # Respect model-defined ordering over free sites when exposed.
        try:
            ordered_free = model.apply(
                _with_cache(variables_no_cache, cache0),
                free_indices,
                method=model.reorder,
            ).astype(jnp.int32)
        except (AttributeError, TypeError):
            ordered_free = free_indices

        # Partition metadata used by vectorized feasibility logic.
        pnum = len(self._partition_sizes)
        partition_of_site = jnp.asarray(self._partition_of_site, dtype=jnp.int32)
        partition_sizes = jnp.asarray(self._partition_sizes, dtype=jnp.int32)
        partition_targets = jnp.asarray(self._partition_targets, dtype=jnp.float32)
        partition_slaves = tuple(int(v) for v in self._partition_slaves)
        local_states_f32 = local_states.astype(jnp.float32)
        step_is_zero = abs(self._local_step) <= 1.0e-12

        if return_log_probabilities:
            # Running state per sample row and per partition.
            logp0 = jnp.zeros((total,), dtype=_log_prob_dtype())
            prefix_sums0 = jnp.zeros((total, pnum), dtype=jnp.float32)
            done_counts0 = jnp.zeros((pnum,), dtype=jnp.int32)
            feasible0 = jnp.ones((total,), dtype=jnp.bool_)

            def _step_part_logp(carry, site_i):
                sigma, cache, key, logp, prefix_sums, done_counts, feasible = carry

                # Query unconstrained conditional probabilities.
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                part = partition_of_site[site_i]

                # Candidate-wise partition sums for each possible local value.
                onehot_p = jax.nn.one_hot(part, pnum, dtype=jnp.float32)[None, None, :]
                sums_new = (
                    prefix_sums[:, None, :] + onehot_p * local_states_f32[None, :, None]
                )
                done_new = done_counts + jax.nn.one_hot(part, pnum, dtype=jnp.int32)
                rem = partition_sizes - done_new

                # Partition feasibility: interval bounds + optional lattice test.
                lo = rem[None, None, :] * self._local_min
                hi = rem[None, None, :] * self._local_max
                need = partition_targets[None, None, :] - sums_new
                in_range = (need >= lo - 1.0e-12) & (need <= hi + 1.0e-12)
                if step_is_zero:
                    mask = jnp.all(in_range, axis=-1)
                else:
                    k = (need - lo) / self._local_step
                    on_lattice = jnp.abs(k - jnp.round(k)) <= 1.0e-12
                    mask = jnp.all(in_range & on_lattice, axis=-1)

                # Restrict support to feasible candidates and renormalize.
                p_masked, row_feasible = _normalise_masked(p, mask)
                feasible = feasible & row_feasible

                # Draw constrained sample and update running partition state.
                new_col, new_p = nkjax.batch_choice(
                    key_draw,
                    local_states,
                    p_masked,
                    return_prob=True,
                )
                sigma = sigma.at[:, site_i].set(new_col)
                prefix_sums = prefix_sums.at[:, part].add(new_col.astype(jnp.float32))
                done_counts = done_new
                logp = logp + jnp.log(new_p).astype(logp.dtype)
                return (
                    sigma,
                    cache,
                    key,
                    logp,
                    prefix_sums,
                    done_counts,
                    feasible,
                ), None

            # Fused scan over all free sites.
            (sigma, _cache, _key, logp, prefix_sums, _done, feasible), _ = jax.lax.scan(
                _step_part_logp,
                (
                    sigma_seed,
                    cache0,
                    key_ar,
                    logp0,
                    prefix_sums0,
                    done_counts0,
                    feasible0,
                ),
                ordered_free,
            )
        else:
            # Same state without log-probability accumulation.
            prefix_sums0 = jnp.zeros((total, pnum), dtype=jnp.float32)
            done_counts0 = jnp.zeros((pnum,), dtype=jnp.int32)
            feasible0 = jnp.ones((total,), dtype=jnp.bool_)

            def _step_part(carry, site_i):
                sigma, cache, key, prefix_sums, done_counts, feasible = carry

                # Query unconstrained conditional probabilities.
                key, key_draw = jax.random.split(key)
                p, mutables = model.apply(
                    _with_cache(variables_no_cache, cache),
                    sigma,
                    site_i,
                    method=model.conditional,
                    mutable=["cache"],
                )
                cache = mutables.get("cache", None)
                part = partition_of_site[site_i]

                # Candidate-wise partition sums for each possible local value.
                onehot_p = jax.nn.one_hot(part, pnum, dtype=jnp.float32)[None, None, :]
                sums_new = (
                    prefix_sums[:, None, :] + onehot_p * local_states_f32[None, :, None]
                )
                done_new = done_counts + jax.nn.one_hot(part, pnum, dtype=jnp.int32)
                rem = partition_sizes - done_new

                # Partition feasibility: interval bounds + optional lattice test.
                lo = rem[None, None, :] * self._local_min
                hi = rem[None, None, :] * self._local_max
                need = partition_targets[None, None, :] - sums_new
                in_range = (need >= lo - 1.0e-12) & (need <= hi + 1.0e-12)
                if step_is_zero:
                    mask = jnp.all(in_range, axis=-1)
                else:
                    k = (need - lo) / self._local_step
                    on_lattice = jnp.abs(k - jnp.round(k)) <= 1.0e-12
                    mask = jnp.all(in_range & on_lattice, axis=-1)

                # Restrict support to feasible candidates and renormalize.
                p_masked, row_feasible = _normalise_masked(p, mask)
                feasible = feasible & row_feasible

                # Draw constrained sample and update running partition state.
                new_col = nkjax.batch_choice(key_draw, local_states, p_masked)
                sigma = sigma.at[:, site_i].set(new_col)
                prefix_sums = prefix_sums.at[:, part].add(new_col.astype(jnp.float32))
                done_counts = done_new
                return (sigma, cache, key, prefix_sums, done_counts, feasible), None

            # Fused scan over all free sites.
            (sigma, _cache, _key, prefix_sums, _done, feasible), _ = jax.lax.scan(
                _step_part,
                (sigma_seed, cache0, key_ar, prefix_sums0, done_counts0, feasible0),
                ordered_free,
            )
            logp = None  # type: ignore[assignment]

        # Deterministically set one slave site per partition.
        slave_exact = jnp.ones((total,), dtype=jnp.bool_)
        for p, slave_site in enumerate(partition_slaves):
            slave = partition_targets[p] - prefix_sums[:, p]
            dists = jnp.abs(slave[:, None] - local_states_f32[None, :])
            idx = jnp.argmin(dists, axis=-1)
            slave_vals = local_states_f32[idx]
            sigma = sigma.at[:, slave_site].set(slave_vals.astype(self.dtype))
            slave_exact = slave_exact & (jnp.abs(slave_vals - slave) <= 1.0e-6)

        # Final acceptance requires feasible prefix path and exact slave closure.
        all_accepted = jnp.all(feasible & slave_exact)

        # Restore NetKet sample shape and return updated sampler key.
        sigma = sigma.reshape(self.n_batches, chain_length, n_sites)
        new_state = state.replace(key=new_key)

        if return_log_probabilities:
            return (
                (sigma, logp.reshape(self.n_batches, chain_length)),
                new_state,
                all_accepted,
            )
        return sigma, new_state, all_accepted


class GenericConstrainedARDirectSampler(ConstrainedARDirectSampler):
    r"""
    Direct sampler for autoregressive models with Hilbert spaces using generic
    callable constraints.

    This sampler is the universal constrained fallback. It repeatedly proposes
    full autoregressive samples and accepts those satisfying the user-provided
    batched constraint callable ``(B, N) -> (B,) bool``.

    The rejection loop is compiled with ``jax.lax.while_loop`` and runs fully
    on device. The method is exact, but throughput depends on acceptance rate and
    is much slower than the autoregressive samplers implemented for the native
    NetKet constraints.
    """

    _check_fn: object = struct.field(pytree_node=False, default=None)
    """Callable constraint validator used in compiled rejection sampling."""

    def __init__(
        self,
        hilbert,
        check_fn,
        machine_pow=None,
        dtype=None,
        *,
        max_resampling_attempts: int = 4096,
    ):
        # Shared sampler initialization.
        super().__init__(
            hilbert,
            machine_pow=machine_pow,
            dtype=dtype,
            max_resampling_attempts=max_resampling_attempts,
        )
        # Constraint callable must be batched and JAX-compatible.
        self._check_fn = check_fn

    @partial(
        jax.jit, static_argnames=("model", "chain_length", "return_log_probabilities")
    )
    def _sampling_kernel(
        self,
        model,
        variables,
        state,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        # Split immutable variables and mutable cache path.
        variables_no_cache, _ = _pop_cache(variables)

        # Flattened sampling axis and local alphabet.
        total = self.n_batches * chain_length
        n_sites = self.hilbert.size
        local_states = self._local_states_array()

        # Key split for outgoing state, cache initialization, and AR stream.
        new_key, key_init, key_ar = jax.random.split(state.key, 3)
        sigma_seed = jnp.zeros((total, n_sites), dtype=self.dtype)

        # Initialize mutable cache once per sampling call.
        cache0 = self._init_cache(model, sigma_seed, key_init)
        check_fn = self._check_fn

        # Model-provided AR ordering when available.
        try:
            site_order = model.apply(
                _with_cache(variables_no_cache, cache0),
                jnp.arange(n_sites, dtype=jnp.int32),
                method=model.reorder,
            ).astype(jnp.int32)
        except (AttributeError, TypeError):
            site_order = jnp.arange(n_sites, dtype=jnp.int32)

        if return_log_probabilities:
            # One complete unconstrained AR proposal with log-probability.
            def _one_ar_pass(key):
                logp0 = jnp.zeros((total,), dtype=_log_prob_dtype())

                def _step(carry, site_i):
                    sigma, cache, k, logp = carry
                    k, k_draw = jax.random.split(k)
                    p, mutables = model.apply(
                        _with_cache(variables_no_cache, cache),
                        sigma,
                        site_i,
                        method=model.conditional,
                        mutable=["cache"],
                    )
                    cache = mutables.get("cache", None)
                    new_col, new_p = nkjax.batch_choice(
                        k_draw, local_states, p, return_prob=True
                    )
                    logp = logp + jnp.log(new_p).astype(logp.dtype)
                    return (sigma.at[:, site_i].set(new_col), cache, k, logp), None

                sigma0 = jnp.zeros((total, n_sites), dtype=self.dtype)
                (sigma, _, key_out, logp), _ = jax.lax.scan(
                    _step, (sigma0, cache0, key, logp0), site_order
                )
                return sigma, logp, key_out

            # Accumulators for accepted rows across rejection attempts.
            accepted0 = jnp.zeros((self.n_batches, chain_length), dtype=jnp.bool_)
            sigma_acc0 = jnp.zeros(
                (self.n_batches, chain_length, n_sites), dtype=self.dtype
            )
            logp_acc0 = jnp.zeros(
                (self.n_batches, chain_length), dtype=_log_prob_dtype()
            )

            def cond_fn(carry):
                # Keep iterating until all rows accepted or attempts exhausted.
                i, _k, accepted, _s, _lp = carry
                return (i < self.max_resampling_attempts) & (~jnp.all(accepted))

            def body_fn(carry):
                # Propose a full fresh batch and fill unresolved accepted rows.
                i, k, accepted, sigma_acc, logp_acc = carry
                k, k_use = jax.random.split(k)
                sigma_new, logp_new, k_next = _one_ar_pass(k_use)
                valid = check_fn(sigma_new).reshape(self.n_batches, chain_length)
                take = (~accepted) & valid
                sigma_new_r = sigma_new.reshape(self.n_batches, chain_length, n_sites)
                logp_new_r = logp_new.reshape(self.n_batches, chain_length)
                sigma_acc = jnp.where(take[:, :, None], sigma_new_r, sigma_acc)
                logp_acc = jnp.where(take, logp_new_r, logp_acc)
                return i + 1, k_next, accepted | valid, sigma_acc, logp_acc

            # Device-side rejection loop.
            _, key_out, accepted, sigma_acc, logp_acc = jax.lax.while_loop(
                cond_fn,
                body_fn,
                (
                    jnp.zeros((), dtype=jnp.int32),
                    key_ar,
                    accepted0,
                    sigma_acc0,
                    logp_acc0,
                ),
            )
        else:
            # Same proposal path without log-probability accumulation.
            def _one_ar_pass(key):
                def _step(carry, site_i):
                    sigma, cache, k = carry
                    k, k_draw = jax.random.split(k)
                    p, mutables = model.apply(
                        _with_cache(variables_no_cache, cache),
                        sigma,
                        site_i,
                        method=model.conditional,
                        mutable=["cache"],
                    )
                    cache = mutables.get("cache", None)
                    new_col = nkjax.batch_choice(k_draw, local_states, p)
                    return (sigma.at[:, site_i].set(new_col), cache, k), None

                sigma0 = jnp.zeros((total, n_sites), dtype=self.dtype)
                (sigma, _, key_out), _ = jax.lax.scan(
                    _step, (sigma0, cache0, key), site_order
                )
                return sigma, key_out

            # Accumulators for accepted rows across rejection attempts.
            accepted0 = jnp.zeros((self.n_batches, chain_length), dtype=jnp.bool_)
            sigma_acc0 = jnp.zeros(
                (self.n_batches, chain_length, n_sites), dtype=self.dtype
            )

            def cond_fn(carry):
                # Keep iterating until all rows accepted or attempts exhausted.
                i, _k, accepted, _s = carry
                return (i < self.max_resampling_attempts) & (~jnp.all(accepted))

            def body_fn(carry):
                # Propose a full fresh batch and fill unresolved accepted rows.
                i, k, accepted, sigma_acc = carry
                k, k_use = jax.random.split(k)
                sigma_new, k_next = _one_ar_pass(k_use)
                valid = check_fn(sigma_new).reshape(self.n_batches, chain_length)
                take = (~accepted) & valid
                sigma_new_r = sigma_new.reshape(self.n_batches, chain_length, n_sites)
                sigma_acc = jnp.where(take[:, :, None], sigma_new_r, sigma_acc)
                return i + 1, k_next, accepted | valid, sigma_acc

            _, key_out, accepted, sigma_acc = jax.lax.while_loop(
                cond_fn,
                body_fn,
                (
                    jnp.zeros((), dtype=jnp.int32),
                    key_ar,
                    accepted0,
                    sigma_acc0,
                ),
            )
            logp_acc = None  # type: ignore[assignment]

        # Emit updated sampler key and aggregate acceptance flag.
        new_state = state.replace(key=key_out)
        all_accepted = jnp.all(accepted)

        if return_log_probabilities:
            return (sigma_acc, logp_acc), new_state, all_accepted
        return sigma_acc, new_state, all_accepted


def ARDirectSampler(
    hilbert,
    machine_pow=None,
    dtype=None,
    *,
    max_resampling_attempts: int = 4096,
):
    r"""
    Direct sampler for autoregressive neural networks on unconstrained and constrained Hilbert spaces.

    This sampler expects a Flax model exposing ``model.conditional``. Given a
    batch of partial samples and a site index ``i``, ``model.conditional`` must
    return the conditional distribution over local states at ``i``.

    In short, for a model factorization

    .. math::

        p(x) = p_1(x_1) p_2(x_2|x_1) \dots p_N(x_N|x_{N-1}, \dots, x_1),

    ``model.conditional(x, i)`` should return :math:`p_i(x)`.


    This sampler supports both unconstrained and constrained spaces.

    - For unconstrained Hilbert spaces, sampling is exact autoregressive direct
      sampling with no rejection.
    - For :class:`~netket.hilbert.constraint.SumConstraint` and
      :class:`~netket.hilbert.constraint.SumOnPartitionConstraint` with uniformly
      spaced local states, sampling uses exact prefix-feasibility masking and
      deterministic dependent-site closure (zero rejection).
    - For generic callable constraints ``(B, N) -> (B,) bool`` (e.g. custom constraints
      which subclass :class:`~netket.hilbert.constraint.DiscreteHilbertConstraint`),
      sampling uses a JAX-compiled rejection loop. This remains exact but will be slower when
      acceptance rates are low.

    Args:
        hilbert: The Hilbert space to sample.
        machine_pow: Optional model power argument for unconstrained usage.
            Constrained samplers internally use ``machine_pow=2`` semantics.
        dtype: The dtype of sampled states.
        max_resampling_attempts: Maximum number of proposal rounds used by the
            generic rejection path.

    Returns:
        A sampler instance compatible with the provided Hilbert space and
        constraint type.
    """

    # Unconstrained spaces use the legacy ARDirectSampler directly.
    if not getattr(hilbert, "constrained", False):
        if machine_pow is None:
            return UnconstrainedARDirectSampler(hilbert=hilbert, dtype=dtype)
        return UnconstrainedARDirectSampler(
            hilbert=hilbert, machine_pow=machine_pow, dtype=dtype
        )

    # Build a static plan once and dispatch to the matching concrete sampler.
    plan = _build_plan(hilbert)
    kind = plan[0]

    if kind == _SUM_PREFIX:
        # SumConstraint path.
        _, payload = plan
        return SumConstrainedARDirectSampler(
            hilbert=hilbert,
            payload=payload,
            dtype=dtype,
            max_resampling_attempts=max_resampling_attempts,
        )

    if kind == _PARTITION_PREFIX:
        # SumOnPartitionConstraint path.
        _, payload = plan
        return PartitionConstrainedARDirectSampler(
            hilbert=hilbert,
            payload=payload,
            dtype=dtype,
            max_resampling_attempts=max_resampling_attempts,
        )

    if kind == _REJECTION:
        # Generic callable-constraint rejection path.
        _, check_fn = plan
        return GenericConstrainedARDirectSampler(
            hilbert=hilbert,
            check_fn=check_fn,
            dtype=dtype,
            max_resampling_attempts=max_resampling_attempts,
        )

    raise RuntimeError(f"Unknown autoregressive sampler plan kind: {kind}")
