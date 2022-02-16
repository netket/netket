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

from netket.experimental.hilbert import SpinOrbitalFermions
from netket.utils.dispatch import dispatch


@dispatch
def random_state(hilb: SpinOrbitalFermions, key, batches: int, *, dtype):
    return random_state(hilb._fock, key, batches, dtype)


@dispatch
def flip_state_scalar(hilb: SpinOrbitalFermions, key, state, index):
    return flip_state_scalar(hilb._fock, key, state, index)


@dispatch
def flip_state_batch(hilb: SpinOrbitalFermions, key, state, index):
    return flip_state_batch(hilb._fock, key, state, index)
