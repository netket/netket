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

from netket.utils import accum_histories_in_tree


class RuntimeLog:
    """
    Runtim Logger, that can be passed with keyword argument `logger` to Monte
    Carlo drivers in order to serialize the outpit data of the simulation.

    This logger keeps the data in memory, and does not save it to disk.
    """

    def __init__(self):
        """
        Crates a Runtime Logger.
        """
        self._data = None
        self._old_step = 0

    def __call__(self, step, item, variational_state):
        self._data = accum_histories_in_tree(self._data, item, step=step)
        self._old_step = step

    @property
    def data(self):
        """
        The dictionary of logged data.
        """
        return self._data

    def __getitem__(self, key):
        return self.data[key]

    def flush(self, variational_state):
        pass
