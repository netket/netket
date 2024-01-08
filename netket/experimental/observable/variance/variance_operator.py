# Copyright 2022 The NetKet Authors - All rights reserved.
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

from netket.operator import AbstractOperator


class VarianceOperator(AbstractOperator):
    def __init__(self, op):
        super().__init__(op.hilbert)
        self._op = op
        self._op2 = op @ op

    @property
    def op(self):
        return self._op

    @property
    def op2(self):
        return self._op2

    @property
    def dtype(self):
        return float

    def __eq__(self, o):
        if isinstance(o, VarianceOperator):
            return o.op == self.op and o.op2 == self.op2
        return False

    @property
    def is_hermitian(self):
        return True

    def __repr__(self):
        return f"VarianceOperator(op={self.op})"
