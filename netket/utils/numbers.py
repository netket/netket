# Copyright 2021-2023 The NetKet Authors - All rights reserved.
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

from numbers import Number

from typing import Any

from .dispatch import dispatch
from .types import Array
from .static_number import StaticZero  # noqa: F401


@dispatch
def dtype(x: Number):
    return type(x)


@dispatch
def dtype(x: Array):  # noqa: F811, E0102
    return x.dtype


@dispatch
def dtype(x: None):  # noqa: F811, E0102
    return None


@dispatch
def dtype(x: type):  # noqa: F811, E0102
    if issubclass(x, Number):
        return x
    raise TypeError(f"type {x} is not a numeric type")


@dispatch
def dtype(x: Any):  # noqa: F811, E0102
    if hasattr(x, "dtype"):
        return x.dtype
    raise TypeError(f"cannot deduce dtype of object type {type(x)}: {x}")


@dispatch
def is_scalar(_: Any):
    return False


@dispatch
def is_scalar(_: Number):  # noqa: F811, E0102
    return True


@dispatch
def is_scalar(x: Array):  # noqa: F811, E0102
    return x.ndim == 0
