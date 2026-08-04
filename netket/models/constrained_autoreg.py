"""
Constraint-aware autoregressive neural networks.

This module provides the constrained variants of the dense autoregressive models:

    - :class:`~netket.models.ConstrainedARNNDense`
    - :class:`~netket.models.ConstrainedFastARNNDense`

Both classes support:
    1. :class:`~netket.hilbert.constraint.SumConstraint`, and
    2. :class:`~netket.hilbert.constraint.SumOnPartitionConstraint`.

The constrained conditionals are constructed directly by masking infeasible local
values at each site and renormalizing. Deterministic closure is used on dependent
sites (last site for :code:`SumConstraint`, one dependent per partition for
:code:`SumOnPartitionConstraint`).
"""

import numpy as np

import jax
from jax import numpy as jnp

from netket.hilbert.constraint import SumConstraint, SumOnPartitionConstraint

from .autoreg import ARNNDense, ARNNConv1D, ARNNConv2D
from .fast_autoreg import FastARNNDense, FastARNNConv1D, FastARNNConv2D


def _is_uniform_spacing(local_states, atol: float = 1.0e-12) -> tuple[bool, float]:
    """
    Check whether local states form a uniform arithmetic progression.

    The constrained masking logic for sum/partition constraints uses range and
    lattice checks, which require uniformly spaced local-state values.
    """
    xs = np.sort(np.asarray(local_states, dtype=np.float64))

    if xs.size <= 1:
        return True, 1.0

    diffs = np.diff(xs)
    step = float(diffs[0])
    ok = bool(np.all(np.abs(diffs - step) <= atol))
    return ok, step


def _normalise_masked(probabilities: jax.Array, mask: jax.Array) -> jax.Array:
    """
    Normalize categorical probabilities after applying a boolean feasibility mask.

    If a row has no feasible entries, the unmasked probabilities are returned for
    numerical safety.
    """
    masked = jnp.where(mask, probabilities, 0.0)
    z = jnp.sum(masked, axis=-1, keepdims=True)
    return jnp.where(z > 0.0, masked / z, probabilities)


def _value_to_local_mask(
    *,
    values: jax.Array,
    local_states: jax.Array,
    tol: float = 1.0e-6,
) -> jax.Array:
    """
    Convert target local-state values into one-hot masks over `local_states`.

    Rows that are not representable within tolerance are mapped to all-False.
    """
    diff = jnp.abs(local_states[None, :] - values[:, None])
    idx = jnp.argmin(diff, axis=-1)
    dmin = jnp.take_along_axis(diff, idx[:, None], axis=-1)[:, 0]
    exact = dmin <= tol
    onehot = jax.nn.one_hot(idx, local_states.shape[0], dtype=jnp.bool_)
    return onehot & exact[:, None]


def _prefix_sum_before(inputs: jax.Array, index: jax.Array) -> jax.Array:
    """
    Return prefix sums `sum(inputs[:, :index], axis=1)` for dynamic JAX indices.
    """
    prefix = jnp.cumsum(inputs, axis=1)
    idx_prev = jnp.maximum(index - 1, 0)
    prev = jax.lax.dynamic_index_in_dim(prefix, idx_prev, axis=1, keepdims=False)
    return jnp.where(index > 0, prev, jnp.zeros((inputs.shape[0],), dtype=inputs.dtype))


def _build_partition_topology(
    sizes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build:
      - site -> partition map
      - dependent site index for each partition
    """
    part_of_site: list[int] = []
    slaves: list[int] = []
    off = 0
    for p, sz in enumerate(sizes):
        part_of_site.extend([p] * sz)
        slaves.append(off + sz - 1)
        off += sz
    return np.asarray(part_of_site, dtype=np.int32), np.asarray(slaves, dtype=np.int32)


class _ConstrainedARNNMixin:
    """
    Shared constrained-AR logic for dense and fast dense autoregressive models.

    Subclasses inherit all base model hyperparameters and architecture.
    """

    _constraint_aware_autoregressive: bool = True
    """Marker consumed by constrained samplers to enable model-direct exact sampling."""

    def __post_init__(self):
        super().__post_init__()

        constraint = getattr(self.hilbert, "constraint", None)
        if not isinstance(constraint, (SumConstraint, SumOnPartitionConstraint)):
            raise ValueError(
                "Constrained ARNN models currently support only "
                "SumConstraint and SumOnPartitionConstraint."
            )

        spacing_ok, _ = _is_uniform_spacing(self.hilbert.local_states)
        if not spacing_ok:
            raise ValueError(
                "Constrained ARNN models require uniformly spaced local_states."
            )

    def _sum_mask(
        self, inputs: jax.Array, index: jax.Array, base_p: jax.Array
    ) -> jax.Array:
        """
        Apply exact sum-constraint feasibility mask to one site conditional.
        """
        constraint = self.hilbert.constraint
        target = jnp.asarray(constraint.sum_value, dtype=jnp.float32)
        local_states = jnp.asarray(self.hilbert.local_states, dtype=jnp.float32)
        local_min = jnp.min(local_states)
        local_max = jnp.max(local_states)
        _, local_step = _is_uniform_spacing(self.hilbert.local_states)
        local_step = jnp.asarray(local_step, dtype=jnp.float32)

        slave_site = jnp.asarray(self.hilbert.size - 1, dtype=jnp.int32)
        index = jnp.asarray(index, dtype=jnp.int32)

        x = inputs.astype(jnp.float32)
        prefix_sum = _prefix_sum_before(x, index)

        # Remaining sites after choosing current value, including the slave site.
        n_remaining = (slave_site - index).astype(jnp.float32)
        need = target - (prefix_sum[:, None] + local_states[None, :])

        lo = n_remaining * local_min
        hi = n_remaining * local_max
        in_range = (need >= lo - 1.0e-6) & (need <= hi + 1.0e-6)

        k = (need - lo) / jnp.where(jnp.abs(local_step) > 1.0e-12, local_step, 1.0)
        on_lattice = jnp.abs(k - jnp.round(k)) <= 1.0e-6
        feasible = jnp.where(
            jnp.abs(local_step) > 1.0e-12, in_range & on_lattice, in_range
        )

        slave_value = target - prefix_sum
        slave_mask = _value_to_local_mask(values=slave_value, local_states=local_states)

        is_slave = index == slave_site
        mask = jnp.where(is_slave, slave_mask, feasible)
        return _normalise_masked(base_p, mask)

    def _partition_mask(
        self,
        inputs: jax.Array,
        index: jax.Array,
        base_p: jax.Array,
    ) -> jax.Array:
        """
        Apply exact partition-sum feasibility mask to one site conditional.
        """
        constraint = self.hilbert.constraint
        sizes = tuple(int(s) for s in constraint.sizes)
        targets = tuple(float(t) for t in constraint.sum_values)
        part_of_site_np, slaves_np = _build_partition_topology(sizes)

        pnum = len(sizes)
        index = jnp.asarray(index, dtype=jnp.int32)

        local_states = jnp.asarray(self.hilbert.local_states, dtype=jnp.float32)
        local_min = jnp.min(local_states)
        local_max = jnp.max(local_states)
        _, local_step = _is_uniform_spacing(self.hilbert.local_states)
        local_step = jnp.asarray(local_step, dtype=jnp.float32)

        part_of_site = jnp.asarray(part_of_site_np, dtype=jnp.int32)
        slaves = jnp.asarray(slaves_np, dtype=jnp.int32)
        sizes_arr = jnp.asarray(sizes, dtype=jnp.float32)
        targets_arr = jnp.asarray(targets, dtype=jnp.float32)

        part = part_of_site[index]
        onehot_part = jax.nn.one_hot(part, pnum, dtype=jnp.float32)

        x = inputs.astype(jnp.float32)

        site_part = jax.nn.one_hot(part_of_site, pnum, dtype=jnp.float32)
        sums_by_site = x[:, :, None] * site_part[None, :, :]
        csum = jnp.cumsum(sums_by_site, axis=1)
        idx_prev = jnp.maximum(index - 1, 0)
        prefix_sums_prev = jax.lax.dynamic_index_in_dim(
            csum, idx_prev, axis=1, keepdims=False
        )
        prefix_sums = jnp.where(
            index > 0,
            prefix_sums_prev,
            jnp.zeros((inputs.shape[0], pnum), dtype=jnp.float32),
        )

        count_csum = jnp.cumsum(site_part, axis=0)
        done_counts_prev = jax.lax.dynamic_index_in_dim(
            count_csum, idx_prev, axis=0, keepdims=False
        )
        done_counts = jnp.where(
            index > 0, done_counts_prev, jnp.zeros((pnum,), dtype=jnp.float32)
        )

        sums_new = (
            prefix_sums[:, None, :]
            + onehot_part[None, None, :] * local_states[None, :, None]
        )
        done_new = done_counts[None, :] + onehot_part[None, :]
        rem = sizes_arr[None, :] - done_new

        need = targets_arr[None, None, :] - sums_new
        lo = rem[:, None, :] * local_min
        hi = rem[:, None, :] * local_max
        in_range = (need >= lo - 1.0e-6) & (need <= hi + 1.0e-6)

        k = (need - lo) / jnp.where(jnp.abs(local_step) > 1.0e-12, local_step, 1.0)
        on_lattice = jnp.abs(k - jnp.round(k)) <= 1.0e-6
        feasible = jnp.where(
            jnp.abs(local_step) > 1.0e-12,
            jnp.all(in_range & on_lattice, axis=-1),
            jnp.all(in_range, axis=-1),
        )

        slave_value = targets_arr[part] - prefix_sums[:, part]
        slave_mask = _value_to_local_mask(values=slave_value, local_states=local_states)
        is_slave = jnp.any(slaves == index)
        mask = jnp.where(is_slave, slave_mask, feasible)
        return _normalise_masked(base_p, mask)

    def _apply_constraint_mask(
        self,
        inputs: jax.Array,
        index: jax.Array,
        base_p: jax.Array,
    ) -> jax.Array:
        """
        Dispatch per-site masking based on the Hilbert-space constraint type.
        """
        constraint = self.hilbert.constraint
        if isinstance(constraint, SumConstraint):
            return self._sum_mask(inputs, index, base_p)
        return self._partition_mask(inputs, index, base_p)

    def conditional(self, inputs, index):
        """
        Constraint-aware single-site conditional used by autoregressive samplers.
        """
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)

        base_p = super().conditional(inputs, index)
        return self._apply_constraint_mask(
            inputs, jnp.asarray(index, dtype=jnp.int32), base_p
        )

    def conditionals(self, inputs):
        """
        Constraint-aware conditionals over all sites.
        """
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)

        base_all = super().conditionals(inputs)
        indices = jnp.arange(self.hilbert.size, dtype=jnp.int32)
        base_t = jnp.swapaxes(base_all, 0, 1)
        masked_t = jax.vmap(
            lambda site_i, p_i: self._apply_constraint_mask(inputs, site_i, p_i)
        )(indices, base_t)
        return jnp.swapaxes(masked_t, 0, 1)

    def __call__(self, inputs):
        """
        Log-wavefunction consistent with constrained conditionals.

        For representable constrained states:
            exp(machine_pow * Re(logpsi)) = product_i p_i^constrained(x_i | x_<i).
        For non-representable states under deterministic closures:
            one or more factors become zero and logpsi is -inf.
        """
        squeeze = False
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
            squeeze = True

        probs = self.conditionals(inputs)
        idx = self.hilbert.states_to_local_indices(inputs)
        p_sel = jnp.take_along_axis(probs, idx[..., None], axis=-1)[..., 0]

        log_prob = jnp.sum(jnp.log(p_sel), axis=-1)
        log_psi = log_prob / float(self.machine_pow)
        return log_psi[0] if squeeze else log_psi


class ConstrainedARNNDense(_ConstrainedARNNMixin, ARNNDense):
    """
    Dense autoregressive wavefunction with exact built-in constraint handling.

    This model represents probability mass only on configurations that satisfy the
    Hilbert-space constraint. At each autoregressive step, infeasible local values
    are masked out and the conditional distribution is renormalized on the feasible
    support.

    Use this model when the Hilbert space is constrained by
    :class:`~netket.hilbert.constraint.SumConstraint` (single global fixed-sum sector)
    or :class:`~netket.hilbert.constraint.SumOnPartitionConstraint`
    (independent fixed-sum sectors over site partitions).

    For such fixed-sum constraints, one dependent site is closed deterministically so the
    final configuration satisfies the target exactly. The resulting log-amplitude is
    consistent with the constrained conditionals used during sampling, so
    `conditionals`, `conditional`, and `__call__` all describe the same constrained
    distribution.

    Assumptions:
        Local state values must lie on a uniformly spaced lattice.
    """


class ConstrainedFastARNNDense(_ConstrainedARNNMixin, FastARNNDense):
    """
    Fast autoregressive neural network with dense layers.

    See :class:`netket.models.FastARNNSequential` for a brief explanation
    of fast autoregressive sampling.

    This model represents probability mass only on configurations that satisfy the
    Hilbert-space constraint. At each autoregressive step, infeasible local values
    are masked out and the conditional distribution is renormalized on the feasible
    support.

    Use this model when the Hilbert space is constrained by
    :class:`~netket.hilbert.constraint.SumConstraint` (single global fixed-sum sector)
    or :class:`~netket.hilbert.constraint.SumOnPartitionConstraint`
    (independent fixed-sum sectors over site partitions).

    For such fixed-sum constraints, one dependent site is closed deterministically so the
    final configuration satisfies the target exactly. The resulting log-amplitude is
    consistent with the constrained conditionals used during sampling, so
    `conditionals`, `conditional`, and `__call__` all describe the same constrained
    distribution.

    Assumptions:
        Local state values must lie on a uniformly spaced lattice.
    """


class ConstrainedARNNConv1D(_ConstrainedARNNMixin, ARNNConv1D):
    """
    Constraint-aware 1D convolutional autoregressive neural network.

    This class keeps the same architecture and hyperparameters as
    :class:`~netket.models.ARNNConv1D` and enforces supported Hilbert constraints
    directly inside autoregressive conditionals.
    """


class ConstrainedFastARNNConv1D(_ConstrainedARNNMixin, FastARNNConv1D):
    """
    Constraint-aware fast 1D convolutional autoregressive neural network.

    This class keeps the same architecture and hyperparameters as
    :class:`~netket.models.FastARNNConv1D` and enforces supported Hilbert
    constraints directly inside autoregressive conditionals.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """


class ConstrainedARNNConv2D(_ConstrainedARNNMixin, ARNNConv2D):
    """
    Constraint-aware 2D convolutional autoregressive neural network.

    This class keeps the same architecture and hyperparameters as
    :class:`~netket.models.ARNNConv2D` and enforces supported Hilbert constraints
    directly inside autoregressive conditionals.
    """


class ConstrainedFastARNNConv2D(_ConstrainedARNNMixin, FastARNNConv2D):
    """
    Constraint-aware fast 2D convolutional autoregressive neural network.

    This class keeps the same architecture and hyperparameters as
    :class:`~netket.models.FastARNNConv2D` and enforces supported Hilbert
    constraints directly inside autoregressive conditionals.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """
