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

from typing import Literal, Union

from plum import dispatch, parametric, convert  # noqa: F401


# Todo: deprecated in netket 3.10/august 2023 . To eventually remove.
def __getattr__(name):
    if name in ["TrueT", "FalseT", "Bool"]:
        from netket.utils import warn_deprecation as _warn_deprecation

        _warn_deprecation(
            """
            The variables `nk.utils.dispatch.{TrueT|FalseT|Bool}` are deprecated. Their usages
            should instead be replaced by the following objects:

                `TrueT` should be replaced by `typing.Literal[True]`
                `FalseT` should be replaced by `typing.Literal[False]`
                `Bool` should be replaced by `bool`
            """
        )
        # Deprecated signature-types for True and False
        # TrueT = Literal[True]
        # FalseT = Literal[False]
        # Bool = Union[TrueT, FalseT]
        if name == "TrueT":
            return Literal[True]
        elif name == "FalseT":
            return Literal[False]
        elif name == "Bool":
            return Union[Literal[True], Literal[False]]

    raise AttributeError(f"module {__name__} has no attribute {name}")
