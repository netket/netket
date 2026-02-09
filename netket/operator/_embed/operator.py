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


from netket.hilbert import TensorHilbert

from .._abstract_operator import AbstractOperator

from .base import EmbedOperator


class EmbedGenericOperator(EmbedOperator, AbstractOperator):
    def __init__(
        self,
        hilbert: TensorHilbert,
        operator: AbstractOperator,
        subspace: int,
    ):
        if not isinstance(hilbert, TensorHilbert):
            raise TypeError(
                "Argument hilbert to EmbedOperator must be a TensorHilbert. "
                f"However the type is:\n\n{type(hilbert)}\n"
            )
        if not isinstance(operator, AbstractOperator):
            raise TypeError(
                "Argument to EmbedOperator not ok: second argument not an operator. "
            )
        super().__init__(hilbert, operator, subspace)
