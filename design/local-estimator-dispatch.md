# Design: unified `local_estimators` dispatch and `LocalEstimators` protocol

## Goal

Make `InfidelityOperator` and `VarianceObservable` (and future nonlinear
observables) work with both `state.expect(op)` and
`state.expect_to_precision(op)`, with no code duplication and a clean,
layered API.

---

## Guiding principle

**Operators declare what their samples look like and how to summarise them.
The framework handles both one-shot and online estimation automatically.**

This requires a strict layering:

```
Layer 0 — Pure stats core          No JAX operators, no MCState
Layer 1 — LocalEstimators type     The interface object
Layer 2 — MultiOnlineStats         Online accumulator for K-dimensional data
Layer 3 — Dispatch infrastructure  MCState-aware dispatch function
Layer 4 — Default dispatch         Bridges existing get_local_kernel pattern
Layer 5 — Operator dispatches      InfidelityOperator, VarianceObservable, …
Layer 6 — MCState API              expect, expect_to_precision
```

Each layer depends only on layers below it.

---

## Layer 0 — Pure stats core

**New file**: `netket/_src/stats/vector_statistics.py`

These functions take raw JAX arrays and Python callables. They have zero
knowledge of operators, variational states, or dispatch.

```python
def vector_statistics(data: Array, combinator: Callable) -> Stats:
    """One-shot Stats from a single batch of K-dimensional local estimators.

    Args:
        data:        shape (n_chains, chain_len, K)
        combinator:  f: (K,) → scalar, must be JAX-traceable (for jacfwd)

    Returns:
        Stats with mean = combinator(mu), error_of_mean via delta method.
    """
    n_chains, chain_len, K = data.shape
    chain_means = jnp.mean(data, axis=1)          # (n_chains, K)
    mu          = jnp.mean(chain_means, axis=0)   # (K,)

    # Covariance of per-chain means → Cov(μ̂ᵢ, μ̂ⱼ) = Cov_chains / n_chains
    Sigma = jnp.cov(chain_means.T) / n_chains     # (K, K)

    J   = jax.jacfwd(combinator)(mu)              # (K,)  — free via autodiff
    err = jnp.sqrt(J @ Sigma @ J)

    # Per-chain scalar estimates for R_hat
    chain_scalars = jax.vmap(combinator)(chain_means)   # (n_chains,)
    R_hat = _r_hat(chain_scalars)

    return Stats(
        mean=combinator(mu),
        error_of_mean=err,
        variance=jnp.nan,          # not defined for nonlinear functionals
        R_hat=R_hat,
    )


def multi_online_statistics(
    data: Array,
    combinator: Callable,
    old_estimator: "MultiOnlineStats | None" = None,
    *,
    max_lag: int = 64,
) -> "MultiOnlineStats":
    """Accumulate K-dimensional local estimators into a MultiOnlineStats.

    Args:
        data:           shape (n_chains, chain_len, K)
        combinator:     f: (K,) → scalar, JAX-traceable
        old_estimator:  previous MultiOnlineStats, or None to start fresh
    """
    K = data.shape[-1]
    if old_estimator is None:
        old_estimator = MultiOnlineStats.empty(K, n_chains=data.shape[0],
                                               combinator=combinator,
                                               max_lag=max_lag)
    return old_estimator.update(data)
```

**Key properties:**
- `vector_statistics` requires `n_chains >= 2` (same as `R_hat`/`tau_corr_batch`).
- `combinator` only needs to be JAX-traceable, not linear. The delta method is
  first-order accurate for any smooth `f`; it is exact when `f` is linear (the
  existing scalar case is linear: `f(mu) = mu`).
- `Sigma` is computed from the K×n_chains array of per-chain means — no extra
  storage compared to tracking K independent `OnlineStats` instances.

---

## Layer 1 — `LocalEstimators` type

**New file**: `netket/_src/stats/local_estimators.py`

A lightweight container that is the **return type of the `local_estimators`
dispatch** and the **interface between operators and the statistics layer**.

```python
from typing import NamedTuple, Callable

class LocalEstimators(NamedTuple):
    data: jax.Array                  # (n_chains, chain_len)     scalar case
                                     # (n_chains, chain_len, K)  vector case
    combinator: Callable | None = None
    # None → scalar case (identity: the data IS the local estimator)

    def to_stats(self) -> Stats:
        """Compute summary statistics for this scalar local-estimator batch."""
        return _local_estimators_to_stats(self.data, self.combinator)

    def accumulate(
        self,
        old: "OnlineStats | MultiOnlineStats | None" = None,
        *,
        max_lag: int = 64,
    ) -> "OnlineStats | MultiOnlineStats":
        """Fold this batch into an online accumulator."""
        return _local_estimators_accumulate(self.data, self.combinator, old,
                                            max_lag=max_lag)
```

The standalone functions that back the methods (so they can be called without
constructing a `LocalEstimators` instance):

```python
def _local_estimators_to_stats(data, combinator) -> Stats:
    if combinator is None:
        # scalar path — uses the existing statistics() function
        return statistics(data)
    return vector_statistics(data, combinator)


def _local_estimators_accumulate(data, combinator, old, *, max_lag):
    if combinator is None:
        return online_statistics(data, old, max_lag=max_lag)
    return multi_online_statistics(data, combinator, old, max_lag=max_lag)
```

`LocalEstimators` is a plain Python `NamedTuple`. It is **not** a JAX pytree —
it lives only at the Python level between a `local_estimators` call and the
statistics call. It is never passed through `jax.jit`.

---

## Layer 2 — `MultiOnlineStats`

**New file**: `netket/_src/stats/multi_online_stats.py`

Online accumulator for K-dimensional local estimators. Mirrors the `OnlineStats`
interface so that `expect_to_precision` and `_check_not_converged` need no
special casing.

```python
class MultiOnlineStats(struct.Pytree):
    """Online accumulator for K-dimensional local estimators.

    Wraps K independent OnlineStats instances and applies the delta method
    at get_stats() time to produce a combined Stats for a smooth functional
    of the K marginal means.
    """

    # Static fields (not pytree leaves)
    combinator: Callable = struct.field(pytree_node=False)
    # The K sub-accumulators (pytree leaves via OnlineStats)
    estimators: tuple[OnlineStats, ...] = struct.field(pytree_node=True)

    @classmethod
    def empty(cls, K, *, n_chains, combinator, max_lag=64) -> "MultiOnlineStats":
        return cls(
            combinator=combinator,
            estimators=tuple(
                OnlineStats(n_chains, dtype=jnp.float64, max_lag=max_lag)
                for _ in range(K)
            ),
        )

    def update(self, data: Array) -> "MultiOnlineStats":
        """data: (n_chains, chain_len, K)"""
        new_estimators = tuple(
            est.update(data[..., k]) for k, est in enumerate(self.estimators)
        )
        return self.replace(estimators=new_estimators)

    def get_stats(self) -> Stats:
        """Delta-method Stats for the combined functional."""
        mu          = jnp.array([e.mean for e in self.estimators])    # (K,)
        chain_means = jnp.stack([e._chain_mean for e in self.estimators])  # (K, n_chains)
        n_chains    = chain_means.shape[1]
        Sigma       = jnp.cov(chain_means) / n_chains                 # (K, K)
        J           = jax.jacfwd(self.combinator)(mu)                 # (K,)
        err         = jnp.sqrt(J @ Sigma @ J)
        return Stats(
            mean=self.combinator(mu),
            error_of_mean=err,
            variance=jnp.nan,
            R_hat=jnp.nan,     # could compute from per-chain scalars if needed
        )

    # Implement same protocol as OnlineStats for expect_to_precision
    def to_dict(self):    return self.get_stats().to_dict()
    def to_compound(self): return self.get_stats().to_compound()
    def __repr__(self):   return repr(self.get_stats())
```

`combinator` is a `pytree_node=False` field — static, never differentiated
through, same pattern as `OnlineStats.max_lag`.

---

## Layer 3 — Dispatch infrastructure

**Modified file**: `netket/vqs/mc/common.py`

Add `local_estimators` as a `@dispatch.abstract` alongside the existing
`get_local_kernel_arguments` and `get_local_kernel`:

```python
@dispatch.abstract
def local_estimators(vstate: Any, op: Any, chunk_size: int | None) -> LocalEstimators:
    """
    Compute per-sample local estimator data for operator op on vstate.

    Returns a LocalEstimators whose .data has shape (n_chains, chain_len)
    for scalar estimators, or (n_chains, chain_len, K) for vector estimators.
    A non-None .combinator indicates a nonlinear functional of K marginal means.

    Default implementation uses get_local_kernel_arguments + get_local_kernel.
    Override this directly for operators that do not fit the kernel pattern.
    """
```

Export from `netket/vqs/mc/__init__.py` alongside the existing exports.

---

## Layer 4 — Default dispatch (bridges existing kernel pattern)

**Modified file**: `netket/vqs/mc/mc_state/expect.py`

The default dispatch that handles all operators currently supported via
`get_local_kernel_arguments` + `get_local_kernel`. This replaces the current
module-level `local_estimators()` function in `state.py`.

```python
@dispatch
def local_estimators(
    vstate: MCState, Ô: AbstractOperator, chunk_size: None
) -> LocalEstimators:
    σ, args = get_local_kernel_arguments(vstate, Ô)
    kernel  = get_local_kernel(vstate, Ô)

    n_chains = σ.shape[0]
    σ_flat   = σ.reshape(-1, σ.shape[-1]) if σ.ndim > 2 else σ

    data = _local_estimators_kernel(
        kernel, vstate._apply_fun, (n_chains,), vstate.variables, σ_flat, args
    )
    return LocalEstimators(data=data)    # combinator=None, scalar case


# Redirect the default @expect.dispatch through local_estimators
@dispatch
def expect(vstate: MCState, Ô: AbstractOperator, chunk_size: None) -> Stats:
    return local_estimators(vstate, Ô, chunk_size).to_stats()
```

The `@expect.dispatch` for `AbstractOperator` that existed before is now
automatically covered. Standard operators no longer need a separate
`@expect.dispatch` — they get it for free once they have a `local_estimators`
implementation (which they do via the default above).

---

## Layer 5 — Operator-specific dispatches

### InfidelityOperator
**Modified file**: `netket/experimental/observable/infidelity/expect.py`

Extract the kernel computation; the `@expect.dispatch` is deleted (covered
by Layer 4's default).

```python
@partial(jax.jit, static_argnames=("afun", "afun_t"))
def _infidelity_kernel_vals(afun, afun_t, params, params_t,
                             model_state, model_state_t, σ, σ_t, cv_coeff):
    log_val, log_val_t = get_kernels(
        afun, afun_t, params, params_t, σ, σ_t, model_state, model_state_t
    )
    if cv_coeff is not None:
        return (jnp.exp(log_val + log_val_t).real
                + cv_coeff * (jnp.exp(2 * (log_val + log_val_t).real) - 1))
    else:
        # cv_coeff=None: exp(log_val) * mean(exp(log_val_t))
        # The global mean term means kernel_vals[i] is not strictly a
        # per-sample quantity; documented limitation.
        return jnp.exp(log_val) * jnp.mean(jnp.exp(log_val_t))


@local_estimators.dispatch
def _(vstate: MCState, op: InfidelityOperator, chunk_size: None) -> LocalEstimators:
    n_chains = vstate.samples.shape[0]
    σ   = vstate.samples.reshape(-1, vstate.hilbert.size)
    σ_t = op.target_state.samples.reshape(-1, vstate.hilbert.size)
    data = _infidelity_kernel_vals(
        vstate._apply_fun,       op.target_state._apply_fun,
        vstate.parameters,       op.target_state.parameters,
        vstate.model_state,      op.target_state.model_state,
        σ, σ_t, op.cv_coeff,
    )
    return LocalEstimators(data=data.reshape(n_chains, -1))
```

The old `@expect.dispatch` and `infidelity_sampling_inner` are deleted;
`_infidelity_kernel_vals` is the extracted core.

### VarianceObservable
**Modified file**: `netket/experimental/observable/variance/expect.py`

The old `@expect.dispatch` and `@expect_and_grad.dispatch` are replaced by a
single `@local_estimators.dispatch`. `expect` is covered by Layer 4's default;
`expect_and_grad` still needs its own dispatch (see below).

```python
@local_estimators.dispatch
def _(vstate: MCState, op: VarianceObservable, chunk_size: int | None) -> LocalEstimators:
    local_kernel  = get_local_kernel(vstate, op.operator, chunk_size)
    local_kernel2 = get_local_kernel(vstate, op.operator_squared, chunk_size)
    σ, args  = get_local_kernel_arguments(vstate, op.operator)
    σ, args2 = get_local_kernel_arguments(vstate, op.operator_squared)

    W      = {"params": vstate.parameters, **vstate.model_state}
    n_chains = σ.shape[0]
    σ_flat = σ.reshape(-1, σ.shape[-1])

    O_loc  = local_kernel (vstate._apply_fun, W, σ_flat, args ).real
    O2_loc = local_kernel2(vstate._apply_fun, W, σ_flat, args2).real

    data = jnp.stack([O_loc, O2_loc], axis=-1).reshape(n_chains, -1, 2)
    # (n_chains, chain_len, 2)

    return LocalEstimators(
        data=data,
        combinator=lambda mu: mu[1] - mu[0] ** 2,
    )
```

Note: the `combinator` is a Python lambda. It is never stored inside a JAX
pytree (`LocalEstimators` is a plain NamedTuple); it is only called from Python
at statistics time.

`@expect_and_grad.dispatch` for `VarianceObservable` is **not** replaced — it
needs to differentiate through the computation and must remain separate.

---

## Layer 6 — MCState API

**Modified file**: `netket/vqs/mc/mc_state/state.py`

```python
# MCState.local_estimators — thin wrapper, unchanged external signature
@timing.timed
def local_estimators(
    self, op: AbstractOperator, *, chunk_size: int | None = None
) -> LocalEstimators:
    """
    Returns a LocalEstimators for operator op at the current samples.

    .data has shape (n_chains, chain_len) for scalar estimators or
    (n_chains, chain_len, K) for vector estimators.
    Call .to_stats() for a one-shot Stats, or .accumulate() for online use.
    """
    if chunk_size is None:
        chunk_size = self.chunk_size
    return local_estimators(self, op, chunk_size)     # dispatch call


# MCState.expect — now just calls local_estimators
@timing.timed
def expect(self, O: AbstractOperator) -> Stats:
    return self.local_estimators(O).to_stats()
```

**Breaking change**: `MCState.local_estimators()` previously returned a raw
`jax.Array`; it now returns `LocalEstimators`. Callers that do
`samples = state.local_estimators(op)` need to add `.data`. This is contained
— only internal netket code calls this method directly.

---

## Layer 6b — `expect_to_precision`

**Modified file**: `netket/_src/vqs/expect_to_precision.py`

The accumulation loop changes only in `_accumulate_stats`. The rest is
unchanged.

```python
def _accumulate_stats(state, op_leaves, active, old_stats, *, max_lag):
    active = set(active)
    new_stats = []
    for i, (op, old) in enumerate(zip(op_leaves, old_stats)):
        if i not in active:
            new_stats.append(old)
            continue
        le = state.local_estimators(op).block_until_ready()
        # le.accumulate() dispatches to online_statistics or
        # multi_online_statistics based on le.combinator
        new_stats.append(le.accumulate(old, max_lag=max_lag))
    return new_stats
```

`_check_not_converged` and the tqdm loop are unchanged — both `OnlineStats`
and `MultiOnlineStats` expose `.get_stats() → Stats`.

Note: `block_until_ready()` must be called on `le.data` (the JAX array), not
on the `LocalEstimators` NamedTuple:
```python
data = state.local_estimators(op)
data.data.block_until_ready()
le = data
```
or alternatively `local_estimators` returns `le` after a `.block_until_ready()`
call inside the dispatch. Implementation detail to be resolved during coding.

---

## File map

| File | Change | Notes |
|------|--------|-------|
| `netket/_src/stats/vector_statistics.py` | **new** | `vector_statistics`, `multi_online_statistics` |
| `netket/_src/stats/multi_online_stats.py` | **new** | `MultiOnlineStats` |
| `netket/_src/stats/local_estimators.py` | **new** | `LocalEstimators`, `_local_estimators_to_stats`, `_local_estimators_accumulate` |
| `netket/stats/__init__.py` | modified | export `LocalEstimators`, `MultiOnlineStats`, `vector_statistics`, `multi_online_statistics` |
| `netket/vqs/mc/common.py` | modified | add `local_estimators` dispatch abstract |
| `netket/vqs/mc/__init__.py` | modified | export `local_estimators` |
| `netket/vqs/mc/mc_state/expect.py` | modified | default `@local_estimators.dispatch`, redirect `@expect.dispatch` |
| `netket/vqs/mc/mc_state/state.py` | modified | `MCState.local_estimators` returns `LocalEstimators`; `MCState.expect` calls `.to_stats()` |
| `netket/_src/vqs/expect_to_precision.py` | modified | `_accumulate_stats` uses `le.accumulate()` |
| `netket/experimental/observable/infidelity/expect.py` | modified | replace `infidelity_sampling_inner` + `@expect.dispatch` with `@local_estimators.dispatch` |
| `netket/experimental/observable/variance/expect.py` | modified | replace `@expect.dispatch` with `@local_estimators.dispatch`; keep `@expect_and_grad.dispatch` |

---

## What operators get for free

An operator that implements only `@local_estimators.dispatch` automatically
works with:

- `state.expect(op)` — via `.to_stats()` in `MCState.expect`
- `state.expect_to_precision(op)` — via `.accumulate()` in `_accumulate_stats`
- Any future caller that goes through `state.local_estimators(op)`

An operator still needs a separate `@expect_and_grad.dispatch` because the
gradient path must differentiate through the local estimator computation,
which cannot be recovered from `LocalEstimators` after the fact.

---

## Constraints and non-requirements

| Property | Required? | Notes |
|---|---|---|
| `combinator` linear | No | Delta method works for any smooth (JAX-traceable) f |
| `combinator` JAX-traceable | Yes | Only for `jax.jacfwd` at stats time, not inside JIT |
| `n_chains >= 2` | For vector case | Needed for Sigma matrix; same as existing R_hat constraint |
| `data` inside JAX pytree | No | `LocalEstimators` is NamedTuple; `data` is a JAX array |
| `combinator` inside JAX pytree | No | Stored as static field in `MultiOnlineStats` |

---

## Implementation order

1. `vector_statistics.py` + `multi_online_stats.py` + `local_estimators.py` — pure, fully testable in isolation
2. `multi_online_statistics` functional API + tests against known analytical cases (variance of Gaussian)
3. Write standalone usage docs (see below) — confirms the API boundary is clean before any dispatch wiring
4. `local_estimators` dispatch abstract in `common.py`
5. Default dispatch in `mc_state/expect.py`; update `MCState.local_estimators` and `MCState.expect`
6. Update `expect_to_precision`
7. `InfidelityOperator` dispatch
8. `VarianceObservable` dispatch
9. Delete dead code: old `infidelity_sampling_inner`, old `@expect.dispatch` for both operators, old module-level `local_estimators` in `state.py`

---

## Standalone usage examples

These examples use only `netket.stats` — no `MCState`, no operators, no
sampling machinery. They serve as the canonical documentation for the pure
API and as the reference for what the dispatch layer must produce.

### 1. Scalar case — plain expectation value

```python
import jax.numpy as jnp
from netket.stats import LocalEstimators, online_statistics

# Simulate: 8 chains, 500 samples each, estimating E[X] where X ~ N(3, 1)
key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (8, 500)) + 3.0   # (n_chains, chain_len)

# One-shot
le = LocalEstimators(data=data)
stats = le.to_stats()
# stats.mean ≈ 3.0,  stats.error_of_mean ≈ 1/sqrt(4000)

# Online (simulating the expect_to_precision loop)
acc = None
for batch in jnp.split(data, 10, axis=1):       # 10 batches of 50
    le  = LocalEstimators(data=batch)
    acc = le.accumulate(acc, max_lag=32)

stats = acc.get_stats()
# stats.mean ≈ 3.0,  stats.error_of_mean decreases with each batch
```

### 2. Vector case — variance via delta method

```python
import jax
import jax.numpy as jnp
from netket.stats import LocalEstimators, multi_online_statistics

# True distribution: X ~ N(mu=2, sigma=1.5)  →  Var[X] = 2.25
key  = jax.random.PRNGKey(1)
X    = jax.random.normal(key, (16, 1000)) * 1.5 + 2.0   # (n_chains, chain_len)
X2   = X ** 2

data = jnp.stack([X, X2], axis=-1)   # (n_chains, chain_len, K=2)
combinator = lambda mu: mu[1] - mu[0] ** 2

# One-shot
le    = LocalEstimators(data=data, combinator=combinator)
stats = le.to_stats()
# stats.mean ≈ 2.25,  stats.error_of_mean via delta method

# Confirm delta method error matches bootstrap (sanity check)
# Bootstrap Var[Var_hat] ≈ (mu4 - mu2²) / N  where mu4 = E[(X-mu)^4]
```

### 3. V-score — nonlinear functional, same K=2 data

```python
# V-score = Var[H] / E[H]^2, using the same (X, X^2) data from above
v_score_combinator = lambda mu: (mu[1] - mu[0] ** 2) / mu[0] ** 2

le    = LocalEstimators(data=data, combinator=v_score_combinator)
stats = le.to_stats()
# stats.mean ≈ 2.25 / 4.0 = 0.5625
# stats.error_of_mean: delta method propagates through the ratio
```

### 4. Online accumulation converging to a target

```python
from netket.stats import LocalEstimators

# Demonstrate that error_of_mean shrinks as 1/sqrt(N)
key = jax.random.PRNGKey(2)
combinator = lambda mu: mu[1] - mu[0] ** 2

acc = None
errors = []
for i in range(200):
    subkey = jax.random.fold_in(key, i)
    X    = jax.random.normal(subkey, (8, 100)) * 1.5 + 2.0
    X2   = X ** 2
    data = jnp.stack([X, X2], axis=-1)

    le  = LocalEstimators(data=data, combinator=combinator)
    acc = le.accumulate(acc, max_lag=32)
    errors.append(acc.get_stats().error_of_mean)

# errors[-1] should be ~ 1/sqrt(8 * 100 * 200) times the single-batch error
```

### 5. MultiOnlineStats directly (lower-level API)

```python
from netket.stats import MultiOnlineStats, multi_online_statistics

combinator = lambda mu: mu[1] - mu[0] ** 2

# Functional API (mirrors online_statistics)
acc = None
for batch in batches:                         # batch: (n_chains, chain_len, 2)
    acc = multi_online_statistics(batch, combinator, acc, max_lag=64)

stats = acc.get_stats()

# Object API (mirrors OnlineStats)
acc = MultiOnlineStats.empty(K=2, n_chains=8, combinator=combinator)
for batch in batches:
    acc = acc.update(batch)

stats = acc.get_stats()
```

### 6. Implementing a custom operator that uses this (no MCState)

```python
# Hypothetical: an operator that computes V-score directly.
# The local estimator data is just two stacked arrays — no netket internals.

def compute_v_score_local_estimators(
    log_psi,           # callable (variables, σ) → log amplitude
    variables,         # parameter pytree
    H_local_kernel,    # callable (log_psi, variables, σ, args) → H_loc
    H2_local_kernel,   # callable for H^2
    σ,                 # (n_chains, chain_len, N) samples
    H_args,            # extra args for H kernel
    H2_args,           # extra args for H^2 kernel
) -> LocalEstimators:
    n_chains = σ.shape[0]
    σ_flat   = σ.reshape(-1, σ.shape[-1])
    W        = variables

    H_loc  = H_local_kernel (log_psi, W, σ_flat, H_args ).real
    H2_loc = H2_local_kernel(log_psi, W, σ_flat, H2_args).real

    data = jnp.stack([H_loc, H2_loc], axis=-1).reshape(n_chains, -1, 2)
    return LocalEstimators(
        data=data,
        combinator=lambda mu: (mu[1] - mu[0] ** 2) / mu[0] ** 2,
    )

# This function has no import from netket.vqs — only netket.stats.LocalEstimators.
# Connecting it to MCState is just one @local_estimators.dispatch wrapper.
```
