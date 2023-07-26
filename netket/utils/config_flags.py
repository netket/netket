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

from typing import Union

import os
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


def get_env(varname: str, type, default: Union[int, bool]) -> Union[int, bool]:
    if type is int:
        return int_env(varname, default)
    elif type is bool:
        return bool_env(varname, default)
    else:
        raise TypeError(f"Unknown type {type}")


class ReadOnlyDict:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class Config:
    def __init__(self):
        self._values = {}
        self._types = {}
        self._editable_at_runtime = {}

        self._readonly = ReadOnlyDict(self._values)

    def define(self, name, type, default, *, help, runtime=False):  # noqa: W0613
        """
        Defines a new flag
        """
        if name in self._values:
            raise KeyError(f"Flag {name} already defined.")

        self._types[name] = type
        self._editable_at_runtime[name] = runtime
        self._values[name] = get_env(name, type, default)

        @property
        def _read_config(self):
            return self.FLAGS[name]

        @_read_config.setter
        def _read_config(self, value):
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

config.define(
    "NETKET_EXPERIMENTAL_DISABLE_ODE_JIT",
    bool,
    default=True,
    help=dedent(
        """
        Disables the jitting of the whole ode solver, mainly used within TDVP solvers.
        The jitting is sometimes incompatible with GPU-based calculations, and on large
        calculations it gives negligible speedups so it might be beneficial to disable it.
        """
    ),
    runtime=True,
)

config.define(
    "NETKET_ENABLE_X64",
    bool,
    default=True,
    help=dedent(
        """
        ....
        """
    ),
    runtime=False,
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
