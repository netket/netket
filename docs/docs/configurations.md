# Configuration Options

NetKet exposes a few configuration options which can be set through environment variables or (in some cases) by accessing {func}`netket.config.update(option_name, value)`.

Options are used to activate experimental or debug functionalities or to disable some parts of netket. 

The supported options are the following:

`````{list-table}
:header-rows: 1
:widths: 10 5 10 20

* - Name
  - Values [default]
  - Changeable
  - Description
* - `NETKET_DEBUG`
  - [False]/True
  - yes
  - Enable debug logging in many netket functions.
* - `NETKET_EXPERIMENTAL`
  - **[False]**/True
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
        Enabling this flag restores the previous behavior of using plain (non-split) Rhat..
* - `NETKET_SPHINX_BUILD`
  - **[False]**/True
  - no
  - Set to True when building documentation with Sphinx. Disables some decorators.
  
`````