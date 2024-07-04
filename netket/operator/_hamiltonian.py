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

from ._discrete_operator import DiscreteOperator


class SpecialHamiltonian(DiscreteOperator):
    def to_local_operator(self):
        raise NotImplementedError(
            f"Must implemented to_local_operator for {type(self)}"
        )

    def conjugate(self, *, concrete: bool = True):
        return self.to_local_operator().conjugate(concrete=concrete)

    def __add__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res = res.__iadd__(other)
            if res is not NotImplemented:
                return res

        return self.to_local_operator() + other

    def __sub__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res = res.__isub__(other)
            if res is not NotImplemented:
                return res

        return self.to_local_operator() - other

    def __radd__(self, other):
        return self.to_local_operator().__radd__(other)

    def __rsub__(self, other):
        return self.to_local_operator().__rsub__(other)

    def __iadd__(self, other):
        if type(self) is type(other):
            res = self._iadd_same_hamiltonian(other)
            return res

        return NotImplemented

    def __isub__(self, other):
        if type(self) is type(other):
            res = self._isub_same_hamiltonian(other)
            return res

        return NotImplemented

    def __mul__(self, other):
        return self.to_local_operator().__mul__(other)

    def __rmul__(self, other):
        return self.to_local_operator().__rmul__(other)

    def _op__matmul__(self, other):
        if hasattr(other, "to_local_operator"):
            other = other.to_local_operator()
        return self.to_local_operator().__matmul__(other)

    def __neg__(self):
        return -1 * self.to_local_operator()

    def _op__rmatmul__(self, other):
        if hasattr(other, "to_local_operator"):
            other = other.to_local_operator()

        return self.to_local_operator().__matmul__(other)
