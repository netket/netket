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

    An observable is a general object that defines a quantity that
    can be comptued starting from a variational state. Observables
    should be computed using the method `expect` of the variational
    states, and derivatives of this expectation value can sometimes
    also be computed using the method `expect_and_grad`.

    All operators are Observables, but some observables are not
    operators (for example, the entanglement entropy observable
    does is not an operator).

    This class determines the basic methods that an observable
    must implement to work correctly with NetKet.
    """

    _hilbert: AbstractHilbert
    r"""The hilbert space associated to this observable."""

    def __init__(self, hilbert: AbstractHilbert):
        self._hilbert = hilbert

    @property
    def hilbert(self) -> AbstractHilbert:
        r"""The hilbert space associated to this observable."""
        return self._hilbert

    def __repr__(self):
        return f"{type(self).__name__}(hilbert={self.hilbert})"
