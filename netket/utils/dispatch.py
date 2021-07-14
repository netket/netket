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

from plum import dispatch, parametric, convert  # noqa: F401


# A simple type to represent a compile-constant True/False type
class Bool:
    """A class representing a static True/False that can be used for dispatch."""

    pass


class TrueT(Bool):
    """A class representing a static True value that can be used for dispatch."""

    def __bool__(self):
        return True


class FalseT(Bool):
    """A class representing a static False value that can be used for dispatch."""

    def __bool__(self):
        return False
