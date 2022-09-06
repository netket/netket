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

from netket.utils.moduletools import export, hide_unexported

from .axial import inversion_group as _inv_group
from .axial import C as _C
from ._point_group import PointGroup


hide_unexported(__name__)

__all__ = [
    "icosahedral_rotations",
    "icosahedral",
]


@export
def I() -> PointGroup:  # noqa: E743
    """
    Rotational symmetries of an icosahedron with two vertices on z axis
    and two others in the xz plane.
    """
    g1 = _C(5, axis=(0, 0, 1))
    g2 = _C(5, axis=(2 / 5**0.5, 0, 1 / 5**0.5))
    g = (g2 @ g1).remove_duplicates()
    g = (g @ g2).remove_duplicates()
    g = (g @ g1).remove_duplicates()
    return g


icosahedral_rotations = I


@export
def Ih() -> PointGroup:
    """
    Symmetry group of an icosahedron with two vertices on z axis
    and two others in the xz plane.
    """
    return _inv_group() @ I()


icosahedral = Ih
