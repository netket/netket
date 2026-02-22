(advanced_custom_callbacks)=
# The Run Loop and Callback Hooks

This page describes the internal structure of the optimization loop executed by
{class}`~netket.driver.AbstractVariationalDriver` and the hooks available to
{class}`~netket.callbacks.AbstractCallback` implementations.

If you only need simple per-step logic, see the legacy callback interface described in the
[Drivers user guide](../user-guides/drivers).  The hook-based API described here is more
powerful and lets you intervene at every stage of a step, including before and after the
gradient computation, before the parameter update, and at the very start or end of a run.

## The run loop

The pseudocode below shows every point at which a callback can be invoked during
{meth}`AbstractVariationalDriver.run() <netket.driver.AbstractVariationalDriver.run>`.
The labels in angle brackets (`<hook_name>`) correspond directly to the methods of
{class}`~netket.callbacks.AbstractCallback`.

```python
# driver.run(n_iter, ...)

callbacks.on_run_start(step, driver)                          # <on_run_start>

for step in range(step_count, step_count + n_iter):
    step_log_data = {}
    step_attempt = 0

    while True:  # inner loop: allows step rejection and retry
        callbacks.on_step_start(step, step_log_data, driver)  # <on_step_start>

        driver.reset_step()  # resets the sampler

        callbacks.on_compute_update_start(                     # <on_compute_update_start>
            step, step_log_data, driver
        )
        loss_stats, dp = driver.compute_loss_and_update()

        reject = callbacks.on_compute_update_end(              # <on_compute_update_end>
            step, step_log_data, driver
        )
        if reject:
            step_attempt += 1
            continue   # retry from on_step_start
        else:
            break      # step accepted

    # step accepted — loss is added to step_log_data
    # step_count and parameters are still at the current (pre-update) values
    callbacks.before_parameter_update(step, step_log_data, driver)  # <before_parameter_update>
    driver.update_parameters(dp)

    callbacks.on_step_end(step, step_log_data, driver)        # <on_step_end>
    step_count += 1

callbacks.on_run_end(step_count, driver)                      # <on_run_end>

# If StopRun is raised anywhere, on_run_end is still called.
# If any other exception is raised, on_run_error is called instead.
# callbacks.on_run_error(step_count, error, driver)           # <on_run_error>
```

## Hook reference

The table below summarises all hooks in calling order, their signature, and their intended use.

| Hook | Called when | Return value |
|------|-------------|--------------|
| {meth}`~netket.callbacks.AbstractCallback.on_run_start` | Once, before the iteration loop | — |
| {meth}`~netket.callbacks.AbstractCallback.on_step_start` | At the start of every step (and every retry) | — |
| {meth}`~netket.callbacks.AbstractCallback.on_compute_update_start` | After `reset_step()`, before `compute_loss_and_update()` | — |
| {meth}`~netket.callbacks.AbstractCallback.on_compute_update_end` | After `compute_loss_and_update()`, before accepting the step | `bool` — if `True`, the step is retried |
| {meth}`~netket.callbacks.AbstractCallback.before_parameter_update` | After the step is accepted; `step_count` and parameters are still at their current values | — |
| {meth}`~netket.callbacks.AbstractCallback.on_step_end` | After parameters have been updated | — |
| {meth}`~netket.callbacks.AbstractCallback.on_run_end` | Once, after the loop finishes (also called on `StopRun`) | — |
| {meth}`~netket.callbacks.AbstractCallback.on_run_error` | When an exception (other than `StopRun`) terminates the loop | — |

### Hook signatures

```python
def on_run_start(self, step: int, driver) -> None: ...

def on_step_start(self, step: int, log_data: dict, driver) -> None: ...

def on_compute_update_start(self, step: int, log_data: dict, driver) -> None: ...

def on_compute_update_end(self, step: int, log_data: dict, driver) -> bool: ...

def before_parameter_update(self, step: int, log_data: dict, driver) -> None: ...

def on_step_end(self, step: int, log_data: dict, driver) -> None: ...

def on_run_end(self, step: int, driver) -> None: ...

def on_run_error(self, step: int, error: Exception, driver) -> None: ...
```

`step` is the current value of `driver.step_count` — a monotonically increasing integer
that is **not** reset between consecutive calls to `run()`.

`log_data` is the per-step dictionary that will eventually be passed to loggers.
Callbacks may add arbitrary keys to it.

## Implementing a custom callback

Subclass {class}`~netket.callbacks.AbstractCallback` and override only the hooks you need.
All hooks have no-op default implementations, so you only need to implement the ones relevant
to your use case.

```python
import netket as nk

class MyCallback(nk.callbacks.AbstractCallback):

    def on_run_start(self, step, driver):
        print(f"Starting optimisation from step {step}")

    def before_parameter_update(self, step, log_data, driver):
        # parameters and step_count still reflect the current step
        log_data["my_quantity"] = compute_something(driver.state)

    def on_run_end(self, step, driver):
        print(f"Finished at step {step}")
```

Pass the callback to {meth}`~netket.driver.AbstractVariationalDriver.run`:

```python
gs.run(n_iter=300, out="output", callback=MyCallback())
```

Multiple callbacks can be provided as a list:

```python
gs.run(n_iter=300, callback=[MyCallback(), nk.callbacks.EarlyStopping(...)])
```

## Stopping the optimisation

To gracefully stop the run from inside a callback, raise {class}`~netket.callbacks.StopRun`
(or a subclass of it).  The driver will call `on_run_end` before exiting.

```python
class StopAfterConvergence(nk.callbacks.AbstractCallback):
    def on_step_end(self, step, log_data, driver):
        if is_converged(log_data):
            raise nk.callbacks.StopRun("Energy converged.")
```

## Injecting custom samples

The `on_compute_update_start` hook runs after `reset_step()` has cleared the sample
cache but before `compute_loss_and_update()` has drawn any new samples.  This makes it
the right place to inject a custom set of configurations that the driver will use for
that step instead of sampling from the Markov chain.

The way this works: `reset_step()` calls `driver.state.reset()`, which sets the internal
cache `driver.state._samples = None`.  When `compute_loss_and_update()` later accesses
`driver.state.samples`, the property sees that `_samples is None` and draws new samples.
If you assign to `_samples` before that point, the property returns your array directly
and no MCMC sampling takes place.

```python
class FixedSamplesCallback(nk.callbacks.AbstractCallback):
    def __init__(self, samples):
        self.samples = samples  # pre-computed array of configurations

    def on_compute_update_start(self, step, log_data, driver):
        # Bypass MCMC: use the same fixed samples at every step.
        driver.state._samples = self.samples
```

:::{note}
`_samples` is an internal field of {class}`~netket.vqs.MCState` (note the leading
underscore).  It is available as a supported extension point here, but the rest of
the public API — in particular {attr}`~netket.vqs.MCState.samples`,
{meth}`~netket.vqs.MCState.sample`, and {meth}`~netket.vqs.MCState.reset` — should
still be used for all other interactions with the variational state.
:::

## Rejecting a step

{meth}`~netket.callbacks.AbstractCallback.on_compute_update_end` may return `True` to
signal that the current step should be discarded and recomputed from `on_step_start`.
This can be used, for example, to implement an accept/reject scheme based on the computed
loss or gradient norm.

```python
class RejectBadSteps(nk.callbacks.AbstractCallback):
    def on_compute_update_end(self, step, log_data, driver):
        grad_norm = compute_norm(driver._dp)
        if grad_norm > 1e6:
            return True   # reject and retry
        return False
```

The number of retries for the current step is available as `driver._step_attempt`.

## Callback ordering

When multiple callbacks are active, they are sorted by
{attr}`~netket.callbacks.AbstractCallback.callback_order` (ascending) before every hook is
called.  The default value is `0`; built-in loggers use `10` so that user callbacks run
first and can populate `log_data` before it is written to disk.

This ordering is especially relevant for `before_parameter_update`: callbacks that estimate
observables (order 0) will always run before loggers that snapshot and write `log_data`
(order 10), guaranteeing that the logged data is complete.

Override `callback_order` to control the relative execution order of your callback:

```python
class EarlyCallback(nk.callbacks.AbstractCallback):
    @property
    def callback_order(self):
        return -10   # runs before everything else
```

## Relationship to legacy callbacks

Legacy callbacks — plain callables with the signature
`(step, log_data, driver) -> bool` — are transparently wrapped and invoked during
`before_parameter_update` (at `callback_order=0`).  They can still be used, but the
hook-based API described on this page gives access to more stages of the loop and is
recommended for new code.
