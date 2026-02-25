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

from typing import TYPE_CHECKING
from flax import serialization


if TYPE_CHECKING:
    from flax import nnx


from flax import nnx

# Flax version 0.10.0 and later have a bug in nnx.to_linen()
# And cannot serialize some NodeDef Mappings that they put in the model_state.
try:

    def serialize_flat_mapping(NodeDef):
        return {}

    def deserialize_flat_mapping(NodeDef, _):
        return NodeDef

    serialization.register_serialization_state(
        nnx.graph.NodeDef,
        serialize_flat_mapping,
        deserialize_flat_mapping,
    )
except Exception:
    pass
