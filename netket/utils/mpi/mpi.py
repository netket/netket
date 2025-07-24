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

# type: ignore

import types


mpi4py_available = False
mpi4jax_available = False
MPI_py_comm = None  # type: ignore
MPI_jax_comm = None  # type: ignore
n_nodes = 1
node_number = 0
rank = 0

FakeMPI = types.ModuleType("FakeMPI", "FakeMPI Module")
FakeMPI.COMM_WORLD = None

MPI = FakeMPI
