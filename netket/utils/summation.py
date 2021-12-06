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

from .struct import dataclass
from .types import Scalar


@dataclass
class KahanSum:
    """
    Accumulator implementing Kahan summation [1], which reduces
    the effect of accumulated floating-point error.

    [1] https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    """

    value: Scalar
    compensator: Scalar = 0.0

    def __add__(self, other: Scalar):
        delta = other - self.compensator
        new_value = self.value + delta
        new_compensator = (new_value - self.value) - delta
        return KahanSum(new_value, new_compensator)
