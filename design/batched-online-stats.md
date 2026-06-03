# Design: a leading batch axis for `OnlineStatsBatch` / `LocalEstimatorsBatch`

**Status:** proposal
**Scope:** `netket/_src/stats/online_stats/`, `netket/_src/stats/local_estimators.py`, `netket/_src/vqs/expect_to_precision.py`
**Author:** discussion notes, 2026-06

---

## 1. Summary

NetKet's K-channel statistics stack (`LocalEstimatorsBatch`, `OnlineStatsBatch`,
`_delta_method_stats`) estimates a smooth nonlinear functional of several
correlated Monte Carlo estimators — variance, susceptibility, Rényi entropy —
together with a delta-method error bar. It does this with **one reduction over
the chains axis**: a single channel-mean vector `X`, a single channel covariance
`Cov`, a single sample count.

That single-reduction assumption is exactly right when the K channels are
*different functions of one shared sample stream*. It is silently wrong when a
caller needs `N` quantities that each come from a *disjoint* sample stream — the
canonical case being a **per-replica / per-parameter-point observable** (e.g. a
susceptibility matrix evaluated at `n_points` different variational parameter
sets, each sampled from its own `|ψ_{θ_p}|²`).

This document explains why the limitation is structural (not a shape or a
propagation problem), demonstrates it numerically, and proposes a minimal,
backward-compatible fix: an **optional leading `n_points` batch axis**, realized
by `jax.vmap`-ing the existing — unchanged — reduction math over that axis. The
reduction kernels are not rewritten; they are lifted.

---

## 2. Background: how the K-channel path works today

The relevant objects:

- **`LocalEstimators`** — scalar per-sample estimator, `data: (n_chains, chain_len)`.
  For linear observables. `to_stats()` → `nk.stats.statistics`.
- **`LocalEstimatorsBatch`** (`local_estimators.py:132`) — K-channel per-sample
  estimator, `data: (n_chains, chain_len, K)`, plus a JAX-traceable
  `combinator: (K,) -> scalar | array`. Used for nonlinear observables that need
  several channels to form the final quantity (variance needs `⟨H⟩`, `⟨H²⟩`;
  a susceptibility matrix needs the `d` and `d⊗d` channels).
- **`OnlineStatsBatch`** (`accumulator_batch.py:71`) — the streaming counterpart:
  K independent `OnlineStats` accumulators (one per channel) plus the combinator.
- **`_delta_method_stats`** (`accumulator_batch.py:31`) — given the combinator
  `f`, the channel means `X: (K,)`, and the channel-mean covariance `Cov: (K,K)`,
  returns `f(X)` with error `√(Jᵀ Cov J)`, `J = ∂f/∂X` via `jax.jacfwd`. Uses
  `jax.eval_shape` to dispatch: scalar combinator → `Stats`, array combinator →
  `StatsBatch` of arbitrary shape.

The one-shot reduction (`LocalEstimatorsBatch.to_stats`, `local_estimators.py:194`):

```python
chain_means = jnp.mean(self.data, axis=1)   # (n_chains, K)
X   = jnp.mean(chain_means, axis=0)         # (K,)
dev = chain_means - X[None, :]
Cov = (dev.T @ dev) / n_chains**2           # (K, K)   — one reduction
return _delta_method_stats(self.combinator, X, Cov)
```

The online reduction (`OnlineStatsBatch.get_stats`, `accumulator_batch.py:191`) is
identical in spirit, with `X` and the per-chain means pulled from the K
accumulators. Both compute **exactly one** `(X, Cov)` pair over the whole chains
axis.

This is the deliberate, autocorrelation-aware MCMC error: reducing over
*chain means* (rather than raw samples) absorbs within-chain autocorrelation into
the between-chain spread, so `√(Jᵀ Cov J)` is a valid standard error even for
correlated MCMC draws.

---

## 3. The limitation

> **One accumulator = one sample stream, with one mean and one normalization.
> Per-point estimation needs `n_points` independent reductions over disjoint
> chain subsets, each with its own mean and its own count.**

It is worth being precise about what is *not* the problem, because the natural
first guesses ("make the output bigger", "add more channels") both fail for the
same underlying reason.

### 3.1 The output shape is *not* the limitation

`_delta_method_stats` discovers the combinator's output shape with
`jax.eval_shape(f, X)` and stores results in a `StatsBatch` of any shape. A
combinator `(K,) -> (n_points, p, p)` is perfectly legal. So one might hope to
pack `n_points` into the channel axis — `K = n_points · (p + p²)` — and let the
combinator reshape. The mean would come out fine. The error would not. (§3.3.)

### 3.2 The error propagation is *not* the limitation

`err²[i] = Jᵢᵀ Cov Jᵢ` is fully general and correct for any differentiable `f`.
When output entry `(p,i,j)` depends only on channel-block `p` of `X`, the
Jacobian row `J[(p,i,j), ·]` is zero outside block `p`, so cross-block covariance
terms `Cov[block p, block p′]`, `p ≠ p′`, are annihilated. This part of the
intuition is correct. The contamination is *within* the surviving diagonal
block. (§3.3.)

### 3.3 What actually breaks: the within-block covariance

To pack `n_points` disjoint streams into K shared channels you must **pad**: a
sample drawn at point `p` has a legitimate value only in block `p`; the other
blocks must be filled (with zeros). A physical sample at `θ_p` is distributed as
`|ψ_{θ_p}|²` and is simply *not a sample* for any other point.

Write the per-chain block-`p` mean as `m_c` (zero for chains not belonging to
point `p`). Then the global channel mean is the **shrunk** value
`X[block p] = (n_p / n_chains) · m̄_p`, and

```
Cov[block p, block p]
   = (1/n_chains²) [ Σ_{c ∈ S_p}(m_c − X_bp)(m_c − X_bp)ᵀ
                     + (n_chains − n_p) · X_bp · X_bpᵀ ]
                            ↑ the OTHER points' chains, sitting at 0
```

Three independent errors live in this block — the block `J` *does* read:

1. **Contamination.** The `n_chains − n_p` chains of other points sit at `0` in
   block `p`, so each deposits a `+X_bp X_bpᵀ` term onto point `p`'s own diagonal
   block. This is *not* a cross-block term, so `J`'s zeros do not remove it.
2. **Wrong normalization.** Divided by `n_chains²` (all chains) instead of
   `n_p²` (point `p`'s chains).
3. **Wrong subtracted mean.** Deviations are taken around the shrunk global
   `X_bp`, not point `p`'s own mean `m̄_p`.

The mean survives all of this because it is *linear*: a constant rescale by
`n_chains / n_p` (or, equivalently, `×n_points` for equal splits) inside the
combinator recovers `m̄_p`. The covariance is quadratic in the deviations and
cannot be un-contaminated by any channel encoding or combinator, because the
chains of different points are **pooled before the combinator ever runs**.

### 3.4 Numerical demonstration

Two independent observables. Point 0 is sampled by 2 chains with chain-means
`[2, 0]`; point 1 by 2 chains with chain-means `[9, 11]`. The correct standard
error of each point, from its own two chains, is `0.7071`.

Encoding this into one `LocalEstimatorsBatch` with `K = 2` padded channels
(chain 0 → `(2,0)`, chain 1 → `(0,0)`, chain 2 → `(0,9)`, chain 3 → `(0,11)`),
combinator `X ↦ 2·X` to undo the mean shrink, and calling `to_stats()`:

| quantity | point 0 | point 1 |
|---|---|---|
| correct (own chains)        | mean 1, err **0.7071** | mean 10, err **0.7071** |
| padded single reduction     | mean 1, err **0.8660** | mean 10, err **5.0498** |

The means are exact. Point 0's error is inflated `0.866/0.707 = 1.22×`; point 1's
is inflated `7.1×`. The hand-derivation of point 0 (`err₀² = 4·0.1875 = 0.75`,
`√0.75 = 0.866`) matches the code exactly. Note that point 1's *values* (9, 11)
never entered point 0's error — exactly as the Jacobian-block argument predicts —
yet point 1 still corrupted point 0's bar purely through the count of its chains
and point 0's own shrunk mean.

The fix below (vmap over a leading points axis) returns `0.7071 / 0.7071`, with no
padding and no rescaling combinator.

---

## 4. Goals and non-goals

**Goals**

- Correct per-point **value and error** for observables whose K channels come
  from `n_points` disjoint sample streams.
- Preserve NetKet's autocorrelation-aware (chain-means) error.
- Make `expect_to_precision` work per-point with **no new driver logic**.
- Zero behavioral change for existing rank-3 callers (variance, scalar
  susceptibility): byte-identical results.
- Keep the reduction *kernels* untouched — lift, don't rewrite.

**Non-goals**

- Multi-axis batching (`batch_shape` of rank > 1). The design leaves the door
  open (`batch_shape` is a tuple) but the first implementation supports a single
  leading axis.
- Changing the statistical definitions (delta method, Welford, Geyer ACF).
- Cross-point correlations. Points are assumed independent (they are — different
  sample streams); we deliberately do *not* form the `(n_points·K)²` joint
  covariance.

---

## 5. Design

### 5.1 The invariant and the discriminator

The whole design rests on one invariant — **one reduction per point** — achieved
by giving the K-channel containers an optional **leading `n_points` axis** and
`jax.vmap`-ing the existing reduction over it.

A single discriminator drives every code path: **array rank**.

| `data` shape | meaning | path |
|---|---|---|
| `(n_chains, chain_len, K)` | one pooled reduction (today) | unchanged |
| `(n_points, n_chains, chain_len, K)` | `n_points` independent reductions | new vmapped path |

The `combinator: (K,) -> out_shape` is **unchanged** and remains *per-point* — it
never sees the points axis. The output simply gains a leading `n_points`: a
scalar combinator → `StatsBatch` of shape `(n_points,)`; a `(p,p)` combinator →
`(n_points, p, p)`.

Because rank-3 inputs take exactly today's code path, backward compatibility is
automatic and total.

### 5.2 Why vmap (and not the alternatives)

**Alternatives considered:**

- **(A) More channels / block-padding.** Rejected — §3.3 shows it cannot produce
  correct errors. This is the option that *looks* like it should work and does
  not; documenting why is half the value of this design.
- **(B) `n_points × K` grid of plain `OnlineStats`, looped in Python.** Correct,
  and requires zero changes to `OnlineStats`. Rejected as the primary design
  because it does not vmap: the per-point loop builds an `n_points × K` pytree,
  recompiles poorly, and scales badly in `n_points`. Retained as a possible
  stepping-stone / reference oracle in tests.
- **(C) Native batched `OnlineStats` with all scalar properties batched.**
  Rejected — `mean`, `variance`, `tau_corr_acf`, `R_hat` are host-side: they call
  `.item()`, branch on Python `if`, and use NumPy data-dependent indexing
  (`tau_corr_acf`). Batching them is invasive and mostly unnecessary, since the
  batched *consumer* (`OnlineStatsBatch`) builds `X`/`Cov` from raw arrays, not
  from those scalar properties.
- **(D) vmap the pure kernel, keep counters scalar.** **Chosen.** The reduction
  math (`_update_arrays` in `kernels.py`, and the `Cov` build in `get_stats`) is
  already a pure function of arrays. We `vmap` it over the leading axis. The
  array *state* carries the points axis; the host-side scalar counters
  (`_n_samples_total`, `_buf_len`) stay scalar because they are identical across
  points (same number of samples per point per update).

### 5.3 Data-flow

```
op.local_estimators(state)                       # foundation: thin dispatch
   └─> LocalEstimatorsBatch(data, combinator)    # data: (n_points, n_chains, chain_len, K)
         ├─ .to_stats()                          # one-shot:  vmap(reduce) over axis 0
         └─ online_statistics(le, acc)
               └─> OnlineStatsBatch               # K OnlineStats, each batch_shape=(n_points,)
                     ├─ .update(data)             # vmap(_update_arrays) over axis 0
                     └─ .get_stats()              # vmap(Cov build + _delta_method_stats)
                           └─> StatsBatch          # mean/err: (n_points, *out_shape)
```

### 5.4 Where the single-chain fallback lands

`LocalEstimatorsBatch.to_stats` already has a `n_chains < 2` fallback that
flattens samples (`local_estimators.py:208-211`); `OnlineStatsBatch.get_stats`
does not (it returns NaN below 2 chains, `accumulator_batch.py:203`). The common
per-point regime is **1 chain per point** (`n_chains=4, n_points=4` → 1 chain
each). Under the vmap, the `n_chains < 2` test is *static* (n_chains is known at
trace time, identical for all points), so the existing fallback branch is simply
chosen inside the per-point function — the inconsistency disappears for free in
both the one-shot and online paths.

---

## 6. Implementation

The kernel `_update_arrays` (`kernels.py`) is **not modified**. The net surface:

| file | edit |
|---|---|
| `kernels.py` | none (vmapped, not modified) |
| `accumulator.py` | `__init__` + `batch_shape`; `n_chains` → `[-1]`; add `batch_shape`/`mean_array`; `update` vmap branch |
| `accumulator_batch.py` | `from_data` rank-4; `get_stats` vmap branch |
| `local_estimators.py` | `to_stats` vmap branch |
| `operations.py` | none (rank-4 flows through unchanged) |
| `expect_to_precision.py` | 2-line type relax |

### 6.1 Phase 1 — one-shot `LocalEstimatorsBatch.to_stats`

Self-contained and independently shippable; fully fixes value + error for
non-streaming use.

```python
def to_stats(self):
    def _reduce(d):                       # d: (n_chains, chain_len, K) — current body verbatim
        n_chains = d.shape[0]
        chain_means = jnp.mean(d, axis=1)
        X = jnp.mean(chain_means, axis=0)
        if n_chains < 2:                  # static under vmap → one branch traced
            flat = d.reshape(-1, d.shape[-1])
            dev = flat - X[None, :]
            Cov = (dev.T @ dev) / flat.shape[0] ** 2
        else:
            dev = chain_means - X[None, :]
            Cov = (dev.T @ dev) / n_chains ** 2
        return _delta_method_stats(self.combinator, X, Cov)

    if self.data.ndim == 3:
        return _reduce(self.data)
    return jax.vmap(_reduce)(self.data)   # (n_points, ...) → StatsBatch on axis 0
```

`_delta_method_stats` returns a `Stats`/`StatsBatch` pytree, which `jax.vmap`
stacks cleanly on the leading axis.

### 6.2 Phase 2a — `OnlineStats` leading axis (`accumulator.py`)

Constructor — prepend `batch_shape` to every array; counters stay scalar:

```python
def __init__(self, n_chains, dtype, *, decay=None, max_lag=64, batch_shape=()):
    ...
    bs = tuple(batch_shape)                       # () today; (n_points,) per-point
    self._chain_count = jnp.zeros((*bs, n_chains), dtype=jnp.float64)
    self._chain_mean  = jnp.zeros((*bs, n_chains), dtype=dtype)
    self._chain_M2    = jnp.zeros((*bs, n_chains), dtype=jnp.float64)
    self._cross_sum  = jnp.zeros((*bs, n_chains, acf_len), dtype=jnp.float64)
    self._m1_sum     = jnp.zeros((*bs, n_chains, acf_len), dtype=jnp.float64)
    self._m2_sum     = jnp.zeros((*bs, n_chains, acf_len), dtype=jnp.float64)
    self._pair_count = jnp.zeros((*bs, n_chains, acf_len), dtype=jnp.float64)
    self._chain_buf  = jnp.zeros((*bs, n_chains, max_lag), dtype=jnp.float64)
    self._buf_len    = jnp.array(0, dtype=jnp.int32)   # shared across points
```

Shape accessors — `n_chains` is the last axis now:

```python
@property
def n_chains(self) -> int:
    return self._chain_mean.shape[-1]             # was shape[0]

@property
def batch_shape(self) -> tuple:
    return self._chain_mean.shape[:-1]            # () unbatched; (n_points,) batched

@property
def mean_array(self):
    """Weighted mean over chains, shape == batch_shape (no host-side .item())."""
    total = jnp.sum(self._chain_count, axis=-1)
    return jnp.sum(self._chain_count * self._chain_mean, axis=-1) / total
```

`update` — vmap the kernel when batched:

```python
def update(self, data) -> "OnlineStats":
    data = jnp.asarray(data)
    batched = self._chain_mean.ndim == 2

    if not batched:
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2:
            raise ValueError(f"data must be 1D or 2D, got {data.ndim}D")
        kernel = _update_arrays
    else:
        if data.ndim != 3:
            raise ValueError(f"batched data must be 3D, got {data.ndim}D")
        kernel = jax.vmap(                         # map states 0..7 and data; share decay/max_lag/buf_len
            _update_arrays,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None, 0),
        )

    n_chains_new        = data.shape[-2]
    n_samples_per_chain = data.shape[-1]
    if n_chains_new != self.n_chains:
        raise ValueError(
            f"Number of chains changed: expected {self.n_chains}, got {n_chains_new}"
        )

    (chain_count, chain_mean, chain_M2,
     cross_sum, m1_sum, m2_sum, pair_count, chain_buf) = kernel(
        self._chain_count, self._chain_mean, self._chain_M2,
        self._cross_sum, self._m1_sum, self._m2_sum, self._pair_count,
        self._chain_buf, self._decay, self.max_lag, self._buf_len, data,
    )
    new_buf_len = jnp.minimum(self._buf_len + n_samples_per_chain, self.max_lag)
    return self.replace(
        _chain_count=chain_count, _chain_mean=chain_mean, _chain_M2=chain_M2,
        _n_samples_total=self._n_samples_total + n_chains_new * n_samples_per_chain,
        _cross_sum=cross_sum, _m1_sum=m1_sum, _m2_sum=m2_sum,
        _pair_count=pair_count, _chain_buf=chain_buf, _buf_len=new_buf_len,
    )
```

The scalar properties (`mean`, `variance`, `R_hat`, `tau_*`) stay unchanged and
valid only for `batch_shape == ()`. Add a guard that raises if called on a
batched instance, so a stray `.item()` fails loudly rather than silently.

### 6.3 Phase 2b — `OnlineStatsBatch` (`accumulator_batch.py`)

`from_data` — accept rank-4 and propagate `batch_shape`:

```python
@classmethod
def from_data(cls, data, combinator, *, max_lag=64):
    data = jnp.asarray(data)
    if data.ndim not in (3, 4):
        raise ValueError(f"data must be 3D or 4D, got {data.ndim}D")
    K = data.shape[-1]
    batch_shape = data.shape[:-3]          # () rank-3; (n_points,) rank-4
    n_chains = data.shape[-3]
    dtype = jnp.result_type(data.dtype, jnp.float64)
    estimators = tuple(
        OnlineStats(n_chains, dtype=dtype, max_lag=max_lag, batch_shape=batch_shape)
        for _ in range(K)
    )
    return cls(estimators=estimators, combinator=combinator).update(data)
```

`update` (`accumulator_batch.py:158`) needs no change: `data[..., k]` slices the
channel axis, leaving `(n_points, n_chains, chain_len)` for the batched
`OnlineStats.update`.

`get_stats` — build `(n_points, …)` arrays, vmap the existing Cov + delta-method:

```python
def get_stats(self):
    X = jnp.stack([e.mean_array for e in self.estimators], axis=-1)   # (*batch, K)
    n_chains = self.n_chains

    if not self.estimators[0].batch_shape:           # rank-3: today's path verbatim
        if self.n_chains < 2 or any(isnan(float(jnp.real(m))) for m in X):
            return _delta_method_stats(self.combinator, X, None)
        chain_means = jnp.stack([e.chain_means for e in self.estimators])  # (K, n_chains)
        dev = chain_means - X[:, None]
        Cov = (dev @ dev.T) / n_chains**2
        return _delta_method_stats(self.combinator, X, Cov)

    # rank-4: one independent reduction per point
    chain_means = jnp.stack([e.chain_means for e in self.estimators], axis=-1)  # (*b, n_chains, K)

    def per_point(Xp, cmp):                          # Xp:(K,)  cmp:(n_chains,K)
        dev = cmp - Xp[None, :]
        Cov = (dev.T @ dev) / n_chains**2
        sb = _delta_method_stats(self.combinator, Xp, Cov)
        nan = jnp.any(jnp.isnan(jnp.real(Xp)))       # frozen-chain guard, per point
        return sb.replace(error_of_mean=jnp.where(nan, jnp.nan, sb.error_of_mean))

    return jax.vmap(per_point)(X, chain_means)       # StatsBatch, leading n_points axis
```

The host-side NaN short-circuit (`accumulator_batch.py:203`) cannot stay
host-side per point, so the batched branch replaces it with a vmappable
`jnp.where` on the error. `_delta_method_stats` itself is unchanged.

### 6.4 Phase 3 — wiring

`operations.py` — **no change**. The `LocalEstimatorsBatch` branch of
`online_statistics` (`operations.py:105`) already forwards `data.data` to
`online_statistics_batch`; rank-4 now flows through because `from_data` accepts
it. The `@partial(jax.jit, static_argnames=("max_lag",))` decorator is
rank-agnostic.

`expect_to_precision.py` — relax the type (pure duck-typing; the body only uses
`state.sample`, `state.local_estimators`, `state.sampler`):

```python
# from netket.vqs.mc import MCState   →   from netket.vqs import VariationalState
def expect_to_precision(state: VariationalState, op, *, ...):
```

Convergence already works batched: `_summary_error_and_scale`
(`expect_to_precision.py:14-23`) detects `OnlineStatsBatch` and reduces with
`jnp.max(jnp.abs(error_of_mean))`, so sampling stops when the **worst point**
meets tolerance.

---

## 7. Correctness argument

Per-point correctness reduces to a single statement: after the vmap, point `p`'s
reduction sees **only point `p`'s slice** `data[p]` of shape
`(n_chains, chain_len, K)`. Therefore:

- `X[p]` is the mean of point `p`'s own channels — no shrink, no other-point
  pollution.
- `Cov[p]` is built from `(chain_means_p − X[p])` over point `p`'s own chains,
  normalized by point `p`'s own `n_chains²`.
- All three defects of §3.3 (contamination, wrong normalization, wrong mean)
  vanish identically, because they were artifacts of pooling, and there is no
  pooling across points anymore.

This is the same reduction that the rank-3 path applies to a single stream —
applied `n_points` times, independently. The numerical check in §3.4 (`0.7071`
both points) is the gating test.

The online path equals the one-shot path by the standard Welford/accumulator
guarantee, applied per point under the vmap (the kernel is the same pure
function in both).

---

## 8. Testing

In `test/stats/test_online_stats.py` and `test/stats/test_local_estimators.py`:

1. **Ground-truth correctness (gate).** The §3.4 toy: rank-4 `to_stats`
   per-point errors must equal independent `nk.stats.statistics` calls per point
   (`0.7071`), and must *differ* from the padded single-reduction result
   (`0.866 / 5.05`) — locking in the bug it fixes.
2. **Online == one-shot.** Accumulate the same data as one rank-4 batch vs. across
   N `update` calls; `get_stats()` must match to tolerance.
3. **Backward compatibility.** Rank-3 inputs reproduce current `main` byte-for-byte
   (golden values for variance + scalar susceptibility).
4. **Single chain per point.** `n_chains=1` per point exercises the single-chain
   fallback; assert finite (non-NaN) errors.
5. **Frozen chains.** Zero within-chain variance at one point → that point's error
   is NaN, others finite (the per-point `jnp.where` guard).
6. **`expect_to_precision` end-to-end.** A per-point observable converges all
   points (worst-point gating); the relaxed `VariationalState` signature accepts a
   foundational state.
7. **vmap-vs-oracle.** Optionally cross-check the vmapped result against the
   alternative-(B) Python `n_points × K` grid as an independent oracle.

---

## 9. Rollout

- **Phase 1** alone is shippable and fixes all *non-streaming* per-point use
  (lowest risk; one method, one file).
- **Phases 2–3** add the streaming path and `expect_to_precision` integration.
- Land on a branch; the §3.4 ground-truth test is the merge gate.
- The foundational per-replica observable then collapses to a thin
  `local_estimators` dispatch returning rank-4 data — most of the current
  hand-rolled per-replica driver disappears onto this generic path.

### Backward compatibility

Total. Every rank-3 input — every current caller — takes the original code path
unchanged. `batch_shape` defaults to `()`. No public signature changes except the
`expect_to_precision` type *relaxation* (strictly widens the accepted set).

---

## 10. API examples

One-shot, per-point susceptibility matrix at `n_points` parameter values:

```python
# data: (n_points, n_chains, chain_len, K),  K = p + p²  (the d and d⊗d channels)
le = LocalEstimatorsBatch(data, combinator=chi_matrix)   # chi_matrix: (K,) -> (p, p)
sb = le.to_stats()
sb.mean           # (n_points, p, p)
sb.error_of_mean  # (n_points, p, p)  — each point reduced independently
```

Streaming accumulation:

```python
acc = None
for _ in range(n_steps):
    state.sample(n_discard_per_chain=0)
    le  = state.local_estimators(op)     # rank-4 LocalEstimatorsBatch
    acc = online_statistics(le, acc)     # OnlineStatsBatch, points axis carried
sb = acc.get_stats()                     # (n_points, p, p)
```

Drive to precision per point (stops when all points meet `rtol`):

```python
stats = expect_to_precision(state, op, rtol=1e-2)
```

---

## 11. Open questions

- **Multi-axis `batch_shape`.** The tuple-valued `batch_shape` admits rank > 1,
  but `update`/`get_stats` currently special-case a single leading axis. Generalize
  only if a concrete need appears.
- **Per-point `max_lag` / `decay`.** Currently shared across points (one scalar
  `_buf_len`, one `decay`). Per-point schedules would require batching those
  counters — out of scope; revisit if per-point thermalization diverges.
- **Loud-failure guard on scalar properties.** Decide whether a batched
  `OnlineStats.mean` should `raise` or return `mean_array`; this design recommends
  `raise` to avoid silent `.item()` surprises.
- **`Stats` vs `StatsBatch` for `n_points == 1`.** A scalar combinator with
  `n_points == 1` yields a `StatsBatch` of shape `(1,)`. Confirm downstream
  consumers accept that rather than expecting a bare `Stats`.
