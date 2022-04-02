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

from netket.utils import warn_deprecation
from inspect import signature


def info(obj, depth=None):
    if hasattr(obj, "info") and callable(obj.info):
        return obj.info(depth)
    else:
        return str(obj)


def ensure_step_value(preconditioner):
    """Adds a dummy `step_value` argument to preconditioners that lack it."""
    if "step_value" not in signature(preconditioner).parameters:
        # Not accepting step_value is deprecated but supported for now
        warn_deprecation(
            "Preconditioners should accept an optional `step_value` argument."
        )

        def new_preconditioner(vstate, rhs, step_value=None):
            return preconditioner(vstate, rhs)

        return new_preconditioner
    else:
        return preconditioner
