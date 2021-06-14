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
        raise ValueError("invalid truth value %r for environment %r" % (val, varname))


def int_env(varname: str, default: int) -> int:
    """Read an environment variable and interpret it as an integer."""
    return int(os.getenv(varname, default))


def get_env(varname: str, type, default: int):
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
    def FLAGS(self):
        """
        The flags of this instance of netket
        """
        return self._readonly

    def update(self, name, value):
        """
        Updates a variable in netket

        Args:
            name: the name of the variable
            value: the new value
        """
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
    runtime=False,
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
