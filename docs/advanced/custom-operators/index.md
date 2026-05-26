# Custom operators and observables

NetKet exposes several extension paths for custom observables and operators.
They differ in how much you implement yourself, whether gradients come for free,
and whether {meth}`~netket.vqs.MCState.expect_to_precision` works automatically.
All three rely on NetKet's multiple-dispatch extension mechanism; if you need
the background first, start with {doc}`../custom_expect` and in particular
{ref}`multiple-dispatch`.

Use the table below as the entry point, then jump to the tutorial section that
matches the interface you want to implement.

```{list-table}
:header-rows: 1
:widths: 17 22 10 14 16 22 19

* - Path
  - What you implement
  - {meth}`~netket.vqs.MCState.expect`
  - {meth}`~netket.vqs.MCState.expect_and_grad`
  - {meth}`~netket.vqs.MCState.expect_to_precision`
  - Best when
  - Start here
* - Scalar operator kernel interface
  - Subclass {class}`~netket.operator.AbstractOperator`, then define
    {func}`~netket.vqs.get_local_kernel_arguments` and
    {func}`~netket.vqs.get_local_kernel`.
  - Automatic
  - Automatic
  - Automatic for scalar local estimators
  - The observable is a standard operator whose expectation is the mean of one
    local-estimator channel.
  - {ref}`custom-operator-lean-interface`
* - Explicit {func}`~netket.vqs.mc.local_estimators` dispatch
  - Register {func}`~netket.vqs.mc.local_estimators` and return either
    {class}`~netket.stats.LocalEstimators` or
    {class}`~netket.stats.LocalEstimatorsBatch`.
  - Automatic
  - No
  - Automatic
  - You need multi-channel local estimators, delta-method error propagation,
    or {meth}`~netket.vqs.MCState.expect_to_precision` without gradients.
  - {ref}`local-estimators-vscore-example`
* - Explicit observable dispatch
  - Register {func}`~netket.vqs.expect` for your custom observable, and optionally
    {func}`~netket.vqs.expect_and_grad`.
  - Yes
  - Only if you define it
  - No
  - You want full control over the computation, or the observable does not fit
    NetKet's local-estimator interfaces.
  - {ref}`custom-observable-dispatch`
```

**Recommended starting points**

- If you need gradients and your observable is a standard scalar operator, start with {ref}`custom-operator-lean-interface`.
- If you do not need gradients and you want {meth}`~netket.vqs.MCState.expect_to_precision`, start with {ref}`local-estimators-vscore-example`.
- If neither interface fits, fall back to {ref}`custom-observable-dispatch` and implement the exact expectation logic yourself.

```{toctree}
:maxdepth: 2

expect-and-grad
local-estimators
```
