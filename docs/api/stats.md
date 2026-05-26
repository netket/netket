(netket_stats_api)=
# netket.stats

```{eval-rst}
.. currentmodule:: netket.stats

```
## Stats

Monte Carlo statistics functions.

```{eval-rst}
.. autosummary::
   :toctree: _generated/stats
   :nosignatures:

   statistics
   Stats
   StatsBatch
```

## Online accumulation

These classes accumulate statistics incrementally across batches of samples,
without keeping all data in memory.
{class}`OnlineStats` is used internally by
{meth}`~netket.vqs.MCState.expect_to_precision` and
{meth}`~netket.vqs.MCState.check_mc_convergence` for scalar local estimators,
while {class}`OnlineStatsBatch` is the corresponding accumulator for
multi-channel local estimators.

```{eval-rst}
.. autosummary::
   :toctree: _generated/stats
   :nosignatures:

   online_statistics
   OnlineStats
   online_statistics_batch
   OnlineStatsBatch
```

## Local estimators

{class}`LocalEstimators` is returned by
{meth}`~netket.vqs.MCState.local_estimators` for scalar observables.
{class}`LocalEstimatorsBatch` is returned for nonlinear observables such as the
variance, where several per-sample channels must be combined through the delta
method.

```{eval-rst}
.. autosummary::
   :toctree: _generated/stats
   :nosignatures:

   LocalEstimators
   LocalEstimatorsBatch
```
