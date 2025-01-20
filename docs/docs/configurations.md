# Configuration Options

NetKet exposes a few configuration options which can be set through environment variables by doing something like
```bash
# without exporting it
NETKET_DEBUG=1 python ...

# by exporting it
export NETKET_DEBUG=1
python ...

# by setting it within python
python
>>> import os
>>> os.environ["NETKET_DEBUG"] = "1"
>>> import netket as nk
>>> print(netket.config.netket_debug)
True
```
Some configuration options can also be changed at runtime by setting it as:
```python
>>> import netket as nk
>>> nk.config.netket_debug = True
>>> ...
```

You can always query the value of an option by accessing the `nk.config` module:
```python
>>> import netket as nk
>>> print(nk.config.netket_debug)
False
>>> nk.config.netket_debug = True
>>> print(nk.config.netket_debug)
True
```

Please note that not all configurations can be set at runtime, and some will raise an error.

Options are used to activate experimental or debug functionalities or to disable some parts of netket.
Please keep in mind that all options related to experimental or internal functionalities might be removed in a future release.

# List of configuration options

`````{list-table}
:header-rows: 1
:widths: 5 2 10 20

* - Name
  - Values **[default]**
  - Changeable
  - Description

* - `NETKET_DEBUG`
  - True/**[False]**
  - yes
  - Enable debug logging in many netket functions.

* - `NETKET_ENABLE_X64`
  - **[True]**/False
  - yes
  - Enable (or disable) double precision in NetKet and jax.

* - `NETKET_EXPERIMENTAL`
  - True/**[False]**
  - yes
  - Enable experimental features such as gradients of non-hermitian operators.

* - `NETKET_MPI_WARNING`
  - **[True]**/False
  - no
  - Raise a warning when running python under MPI without mpi4py and other mpi dependencies installed.

* - `NETKET_MPI`
  - **[True]**/False
  - no
  - When true, NetKet will always attempt to load (and initialize) MPI. If this flag is `0` `mpi4py` and `mpi4jax` will not be imported. This can be used to prevent crashes with some `MPI` variants such as Cray which cannot be initialised when not running under `mpirun`.

* - `NETKET_USE_PLAIN_RHAT`
  - **[True]**/False
  - yes
  - By default, NetKet uses the split-R̂ Gelman-Rubin diagnostic in `netket.stats.statistics`,
    which detects non-stationarity in the MCMC chains (in addition to the classes of
    chain-mixing failures detected by plain R) since version 3.4.
    Enabling this flag restores the previous behavior of using plain (non-split) Rhat.

* - `NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION`
  - True/**[False]**
  - yes
  - The integrated autocorrelation time $\tau_c$ is computed separately for each chain $c$.
    To summarize it for the user, `Stats.tau_corr` is changed to contain the average over all
    chains and a new field `Stats.tau_corr_max` is added containing the maximum autocorrelation
    among all chains (which helps to identify outliers). Using the average $\tau$ over all chains
    seems like a good choice as it results in a low-variance estimate
    (see [here](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr) for a good
    discussion).

* - `NETKET_SPHINX_BUILD`
  - True/**[False]**
  - no
  - Set to True when building documentation with Sphinx. Disables some decorators. This is for internal use only.

* - `NETKET_EXPERIMENTAL_SHARDING`
  - True/**[False]**
  - no
  - Flag to turn on experimental support for multiple jax devices. When True, NetKet will distribute the markov chains/samples uniformly across all available jax devices and utilize them for computations.

* - `NETKET_EXPERIMENTAL_SHARDING_CPU`
  - integer
  - no
  - Convenience helper to set the flag `XLA_FLAGS='--xla_force_host_platform_device_count=XX', forcing jax to use multiple threads as separate cpu devices.

`````
