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

import pytest  # noqa: F401


def pytest_addoption(parser):
    parser.addoption(
        "--sampler",
        action="store",
        default="",
        help="sampler: exact, metropolis, metropolispt or pt, local, hamiltonian, custom, autoregressive, gaussian",
    )

    parser.addoption(
        "--mpow",
        action="store",
        default="single",
        help="mpow: single, all, 1,2,3...",
    )
