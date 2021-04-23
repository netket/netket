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

import json
import dataclasses
import tarfile
import time
from io import BytesIO

import os
from os import path as _path
import numpy as np
import jax

from flax import serialization


def save_binary_to_tar(tar_file, byte_data, name):
    abuf = BytesIO(byte_data)

    # Contruct the info object with the correct length
    info = tarfile.TarInfo(name=name)
    info.size = len(abuf.getbuffer())

    # actually save the data to the tar file
    tar_file.addfile(tarinfo=info, fileobj=abuf)


class TarStateLog:
    """
        Tar Logger, serializing the variables of the variational state during a run.

    : bool flag indicating whever to store variables in a tar file. The tar archive will
                    contain a file with numbers going from 0 to N, and every file corresponds to the variables of
                    the variational state at that step.


        Data is serialized to a tar archive as many files named `[0.mpack, 1.mpack, ...]`, where
        the number increases by 1 every time it's called.

        The tar file inside is not flushed to disk (closed) until this object is deleted or python
        is shut down.
    """

    def __init__(
        self,
        output_prefix: str,
        mode: str = "write",
        save_every: int = 1,
    ):
        """
        Construct a Tar Logger.

        Args:
            output_prefix: the name of the output files before the extension
            save_every: every how many iterations the variables should be saved. (default 1)
            mode: Specify the behaviour in case the file already exists at this output_prefix. Options are
                - `[w]rite`: (default) overwrites file if it already exists;
                - `[a]ppend`: appends to the file if it exists, overwise creates a new file;
                - `[x]` or `fail`: fails if file already exists;
        """
        super().__init__()

        # Shorthands for mode
        if mode == "w":
            mode = "write"
        elif mode == "a":
            mode = "append"
        elif mode == "x":
            mode = "fail"

        if not ((mode == "write") or (mode == "append") or (mode == "fail")):
            raise ValueError(
                "Mode not recognized: should be one of `[w]rite`, `[a]ppend` or `[x]`(fail)."
            )

        file_exists = _path.exists(output_prefix + ".tar")

        if file_exists and mode == "append":
            # if there is only the .mpacck file but not the json one, raise an error
            if not _path.exists(output_prefix + ".log"):
                raise ValueError(
                    "History file does not exists, but wavefunction file does. Please change `output_prefix or set mode=`write`."
                )

        elif file_exists and mode == "fail":
            raise ValueError(
                "Output file already exists. Either delete it manually or change `output_prefix`."
            )

        dir_name = _path.dirname(output_prefix)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        self._prefix = output_prefix
        self._file_mode = mode

        self._save_every = save_every
        self._old_step = 0
        self._steps_notsaved = 0

        self._runtime_taken = 0.0

        # tar
        self._tar_file = None
        self._closed = False

    def _create_tar_file(self):
        if self._tar_file is None:
            self._tar_file = tarfile.TarFile(self._prefix + ".tar", self._file_mode[0])
            self._tar_step = 0
            if self._file_mode == "a":
                self._tar_step = int(self._tar_file.getnames()[-1]) + 1

    def close(self):
        if not self._closed and self._tar_file is not None:
            self._tar_file.close()
            self._closed = True

    def __call__(self, step, item, variational_state):
        old_step = self._old_step

        if self._steps_notsaved % self._save_every == 0 or step == old_step - 1:
            self._save_variables(variational_state)

        self._old_step = step
        self._steps_notsaved += 1

    def _save_variables(self, variational_state):
        if self._tar_file is None:
            self._create_tar_file()

        _time = time.time()
        binary_data = serialization.to_bytes(variational_state.variables)
        save_binary_to_tar(self._tar_file, binary_data, str(self._tar_step) + ".mpack")
        self._tar_step += 1
        self._runtime_taken += time.time() - _time

    def __del__(self):
        if hasattr(self, "_closed"):
            self.close()

    def flush(self, variational_state):
        pass

    def __repr__(self):
        return f"TarLog('{self._prefix}', mode={self._file_mode})"

    def __str__(self):
        _str = self.__repr__()
        _str = _str + f"\n  Runtime cost: {self._runtime_taken}"
        return _str
