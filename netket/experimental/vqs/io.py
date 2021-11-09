# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
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

import tarfile as _tarfile
from os import path as _path

from flax import serialization as _serialization
from netket.utils.types import PyTree as _PyTree


def variables_from_file(filename: str, variables: _PyTree):
    """
    Loads the variables of a variational state from a `.mpack` file.

    Args:
        filename: the file containing the variables. Assumes a .mpack
            extension and adds it if missing and no file exists.
        variables: An object variables with the same structure and shape
            of the object to be deserialized.

    Returns:
        a PyTree like variables

    Examples:
       Serializing the data:

       >>> import netket as nk
       >>> import flax
       >>> # construct an RBM model on 10 spins
       >>> vstate = nk.variational.MCState(
       ...      nk.sampler.MetropolisLocal(nk.hilbert.Spin(0.5)**10),
       ...      nk.models.RBM())
       >>> with open("test.mpack", 'wb') as file:
       ...     bytes_written = file.write(flax.serialization.to_bytes(vstate.variables))
       >>> print(bytes_written)
       1052
       >>>
       >>> # Deserialize the data
       >>>
       >>> del vstate
       >>> # construct an RBM model on 10 spins
       >>> vstate2 = nk.variational.MCState(
       ...      nk.sampler.MetropolisLocal(nk.hilbert.Spin(0.5)**10),
       ...      nk.models.RBM())
       >>> # Load the data by passing the model
       >>> vars = nk.variational.experimental.variables_from_file("test.mpack",
       ...                                                        vstate2.variables)
       >>> # update the variables of vstate with the loaded data.
       >>> vstate2.variables = vars
    """
    if not _path.isfile(filename):
        if filename[-6:] != ".mpack":
            filename = filename + ".mpack"

    with open(filename, "rb") as f:
        return _serialization.from_bytes(variables, f.read())


def variables_from_tar(filename: str, variables: _PyTree, i: int):
    """
    Loads the variables of a variational state from the i-th element of a `.tar`
    archive.

    Args:
        filename: the tar archive name. Assumes a .tar
            extension and adds it if missing and no file exists.
        variables: An object variables with the same structure and shape
            of the object to be deserialized.
        i: the index of the variables to load
    """
    if not _path.isfile(filename):
        if filename[-4:] != ".tar":
            filename = filename + ".tar"

    with _tarfile.TarFile(filename, "r") as file:
        info = file.getmember(str(i) + ".mpack")
        with file.extractfile(info) as f:
            return _serialization.from_bytes(variables, f.read())
