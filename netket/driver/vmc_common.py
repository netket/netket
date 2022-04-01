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


def info(obj, depth=None):
    if hasattr(obj, "info") and callable(obj.info):
        return obj.info(depth)
    else:
        return str(obj)


def apply_preconditioner(self):
    try:
        # Default: preconditioner accepts step_value
        self._dp = self.preconditioner(
            self.state, self._loss_grad, step_value=self.step_count
        )
    except TypeError:
        # Not accepting step_value is deprecated but supported for now
        warn_deprecation(
            "Preconditioners should accept an optional `step_value` argument."
        )
        self._dp = self.preconditioner(self.state, self._loss_grad)
