# Copyright 2023 The NetKet Authors - All rights reserved.
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

import abc

from netket.hilbert import AbstractHilbert


class AbstractObservable(abc.ABC):
    """Abstract class for quantum Observables.
    This class prototypes the methods
    needed by a class satisfying the Operator concept.
    """

    _hilbert: AbstractHilbert
    r"""The hilbert space associated to this operator."""

    def __init__(self, hilbert: AbstractHilbert):
        self._hilbert = hilbert

    @property
    def hilbert(self) -> AbstractHilbert:
        r"""The hilbert space associated to this operator."""
        return self._hilbert

    def __repr__(self):
        return f"{type(self).__name__}(hilbert={self.hilbert})"
