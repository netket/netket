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

"""
This module contains two classes used to index into hilbert spaces
inheriting from `HomogeneousHilbert`.

Those classes provide an informal API that can be used to extend or override
netket's indexing logic, which is particularly relevant when working with
constrained Hilbert spaces.

----------------------------------------------------------------------------
    This is not part of NetKet's public API and may change at any moment!
----------------------------------------------------------------------------

An hilbert indexing class should respect the `HilbertIndex` protocol
defined in the file `base.py`.

"""

from .base import HilbertIndex, is_indexable, max_states
from .constraints import ConstrainedHilbertIndex, optimalConstrainedHilbertindex
from .unconstrained import LookupTableHilbertIndex
from .uniform_tensor import UniformTensorProductHilbertIndex
