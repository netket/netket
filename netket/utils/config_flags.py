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

import os
import sys
import warnings
from textwrap import dedent


def bool_env(varname: str, default: bool) -> bool:
    """Read an environment variable and interpret it as a boolean.
    True: 'y', 'yes', 't', 'true', 'on', and '1';
    False: 'n', 'no', 'f', 'false', 'off', and '0'.

    Args:
        varname: the name of the variable
        default: the default boolean value
    """
    val = os.getenv(varname, str(default))
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r} for environment {varname!r}")


def int_env(varname: str, default: int) -> int:
    """Read an environment variable and interpret it as an integer."""
    return int(os.getenv(varname, default))


def get_env(varname: str, type, default: int | bool) -> int | bool:
    if type is int:
        return int_env(varname, default)
    elif type is bool:
        return bool_env(varname, default)  # type: ignore
    else:
        raise TypeError(f"Unknown type {type}")


class ReadOnlyDict:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class Config:
    _HAS_DYNAMIC_ATTRIBUTES = True

    def __init__(self):
        self._values = {}
        self._types = {}
        self._editable_at_runtime = {}

        self._readonly = ReadOnlyDict(self._values)
        self._callbacks = {}

    def define(
        self,
        name,
        type,
        default,
        *,
        help,
        runtime=False,
        callback=None,
        lazy=False,
    ):  # noqa: W0613
        """
        Defines a new flag

        Args:
            name: the flag name, should be an uppercase string like "NETKET_XXX"
            type: should be the type (bool, int) of the flag
            default: default value
            help: a string to use as description of this flag
            runtime: whether the flag can be modified at runtime
            callback: an optional callback function taking the value as argument
                that is executed when the flag is changed
            lazy: do not call the callback at init.
        """
        if name in self._values:
            raise KeyError(f"Flag {name} already defined.")

        self._types[name] = type
        self._editable_at_runtime[name] = runtime
        self._values[name] = get_env(name, type, default)
        self._callbacks[name] = callback

        if callback is not None and not lazy:
            callback(self._values[name])

        @property
        def _read_config(self):
            return self.FLAGS[name]

        @_read_config.setter
        def _read_config(self, value):
            if self._callbacks[name] is not None:
                self._callbacks[name](value)
            self.update(name, value)

        setattr(Config, name.lower(), _read_config)

    @property
    def FLAGS(self):
        """
        The flags of this instance of netket
        """
        return self._readonly

    def update(self, name, value):
        """
        Updates a configuration variable in netket.

        Args:
            name: the name of the variable
            value: the new value
        """
        name = name.upper()

        if not self._editable_at_runtime[name]:
            raise RuntimeError(
                f"\n\nFlag `{name}` can only be set through an environment "
                "variable before importing netket.\n"
                "Try launching python with:\n\n"
                f"\t{name}={self.FLAGS[name]} python\n\n"
                "or execute the following snippet BEFORE importing netket:\n\n"
                "\t>>>import os\n"
                f'\t>>>os.environ["{name}"]="{self.FLAGS[name]}"\n'
                "\t>>>import netket as nk\n\n"
            )

        if not isinstance(value, self._types[name]):
            raise TypeError(
                f"Configuration {name} must be a {self._types[name]}, but the "
                f"value {value} is a {type(value)}."
            )

        self._values[name] = self._types[name](value)

    def __repr__(self):
        txt = "\nGlobal configurations for NetKet\n"
        for k, v in self._values.items():
            txt = txt + f" - {k} = {v}\n"

        return txt

    def __dir__(self):
        return list(k.lower() for k in self._values.keys())


config = Config()
FLAGS = config.FLAGS

config.define(
    "NETKET_DEBUG",
    bool,
    default=False,
    help="Enable debug logging in many netket functions.",
    runtime=True,
)

config.define(
    "NETKET_EXPERIMENTAL",
    bool,
    default=False,
    help="Enable experimental features.",
    runtime=True,
)

config.define(
    "NETKET_MPI_WARNING",
    bool,
    default=True,
    help=dedent(
        """
        Raise a warning when running python under MPI
        without mpi4py and other mpi dependencies installed.
        """
    ),
    runtime=False,
)

config.define(
    "NETKET_MPI",
    bool,
    default=True,
    help=dedent(
        """
        Prevent NetKet from using (and initializing) MPI. If this flag is
        `0` `mpi4py` and `mpi4jax` will not be imported.
        """
    ),
    runtime=False,
)

config.define(
    "NETKET_USE_PLAIN_RHAT",
    bool,
    default=False,
    help=dedent(
        """
        By default, NetKet uses the split-R̂ Gelman-Rubin diagnostic in `netket.stats.statistics`,
        which detects non-stationarity in the MCMC chains (in addition to the classes of
        chain-mixing failures detected by plain R) since version 3.4.
        Enabling this flag restores the previous behavior of using plain (non-split) Rhat.
        """
    ),
    runtime=True,
)

config.define(
    "NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION",
    bool,
    default=False,
    help=dedent(
        """
        The integrated autocorrelation time $\tau_c$ is computed separately for each chain $c$.
        To summarize it for the user, `Stats.tau_corr` is changed to contain the average over all
        chains and a new field `Stats.tau_corr_max` is added containing the maximum autocorrelation
        among all chains (which helps to identify outliers). Using the average $\tau$ over all chains
        seems like a good choice as it results in a low-variance estimate
        (see [here](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr) for a good
        discussion).
        """
    ),
    runtime=True,
)

# TODO: removed in january 2025, defaults to True now.
config.define(
    "NETKET_EXPERIMENTAL_DISABLE_ODE_JIT",
    bool,
    default=True,
    help=dedent(
        """
        Deprecated: jax does not support reentrant callbacks anymore.

        Disables the jitting of the whole ode solver, mainly used within TDVP solvers.
        The jitting is sometimes incompatible with GPU-based calculations, and on large
        calculations it gives negligible speedups so it might be beneficial to disable it.
        """
    ),
    runtime=True,
)


def _setup_experimental_sharding_cpu(n_procs):
    if n_procs > 1:
        if "jax" in sys.modules:
            warnings.warn(
                "must load NetKet before jax if using experimental_sharding_cpu"
            )

        flags = os.environ.get("XLA_FLAGS", "")
        flags = f"{flags} --xla_force_host_platform_device_count={n_procs}"
        os.environ["XLA_FLAGS"] = flags


config.define(
    "NETKET_EXPERIMENTAL_SHARDING_CPU",
    int,
    default=0,
    help=dedent(
        """
        Set to >=1 to force JAX to use multiple threads as separate devices on cpu.
        Sets the XLA_FLAGS='--xla_force_host_platform_device_count=#' environment variable.
        Disabled by default.
        """
    ),
    runtime=False,
    callback=_setup_experimental_sharding_cpu,
)


def _update_x64(val):
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", val)


# This flag is setup to mirror JAX_ENABLE_X64 with True default. any of the two
# Can be explicitly set.
config.define(
    "NETKET_ENABLE_X64",
    bool,
    default=bool_env(
        "JAX_ENABLE_X64", True
    ),  # respect explicit JAX_ENABLE_X64 settings
    help=dedent(
        """
        Enables double-precision for Jax. Equivalent to `JAX_ENABLE_X64` but defaults to
        True instead of False, as it is required throughout NetKet. By setting this flag
        to False NetKet will run without double-precision everywhere.
        """
    ),
    runtime=True,
    callback=_update_x64,
)


config.define(
    "NETKET_SPHINX_BUILD",
    bool,
    default=False,
    help=dedent(
        """
        Set to True when building documentation with Sphinx. Disables some decorators.
        """
    ),
    runtime=True,
)


def _setup_experimental_sharding(val):
    if val:
        from jax import config as jax_config

        jax_config.update("jax_threefry_partitionable", True)


config.define(
    "NETKET_EXPERIMENTAL_SHARDING",
    bool,
    default=int_env("NETKET_EXPERIMENTAL_SHARDING_CPU", 0) > 0,
    help=dedent(
        """
        Enables highly expermiental support of netket for running on multiple jax devices.

        Supports both multiple local devices, as well as global ones in a multi-process environment.
        See https://jax.readthedocs.io/en/latest/multi_process.html#initializing-the-cluster for
        how to initialize the latter.
        Distributes chains and samples equally among all available devices.

        Hybrid parallelization with MPI is not supported, enabling NETKET_EXPERIMENTAL_SHARDING
        disables mpi.
        """
    ),
    runtime=False,
    callback=_setup_experimental_sharding,
)


def _recompute_default_device(val):
    if val is False:
        import jax

        jax.config.update("jax_default_device", None)
    else:
        from netket.utils.mpi.gpu_autorank_util import autoset_default_gpu

        autoset_default_gpu()


config.define(
    "NETKET_MPI_AUTODETECT_LOCAL_GPU",
    bool,
    default=False,
    runtime=True,
    callback=_recompute_default_device,
    lazy=True,
    help=dedent(
        """
        If there are more than 1 device per MPI rank, and if they are GPU devices,
        Set the default device by querying the local MPI rank.
        """
    ),
)


config.define(
    "NETKET_RANDOM_STATE_FALLBACK_WARNING",
    bool,
    default=True,
    runtime=True,
    help=dedent(
        """
        Print a warning every time you use a fallback that could never stop running
        when using random_state of constrained hilbert spaces with custom
        constraints.
        """
    ),
)


config.define(
    "NETKET_EXPERIMENTAL_SHARDING_FAST_SERIALIZATION",
    bool,
    default=False,
    runtime=True,
    help=dedent(
        """
        If True (Defaults False) does not gather data on the master process when
        using flax.serialization methods. This allows to use orbax-checkpoint with
        higher efficiency.
        """
    ),
)


config.define(
    "NETKET_SPIN_ORDERING_WARNING",
    bool,
    default=True,
    runtime=True,
    help=dedent(
        """
        If True (Defaults True) warns if the ordering of spins in the Hilbert space
        is not declared.
        """
    ),
)
