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

"""
This module attempts to autodetect when a model/Module passed to netket
comes from one of the several jax packages in existance.

It it comes from jax, flax, haiku or whatever else, and then extracts
the two functions that are really needed (init_fun and apply_fun).

If you want to add support for another framework, you should add
a new file in this folder and include it here.
"""


from .base import maybe_wrap_module, registered_frameworks

from . import flax, jax, haiku
