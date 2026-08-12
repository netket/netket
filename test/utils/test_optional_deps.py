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

import sys
from types import ModuleType

import pytest

from netket.utils.optional_deps import import_optional_dependency

from .. import common

pytestmark = common.skipif_distributed


@pytest.fixture
def fake_module(monkeypatch):
    """Install a fake module with a given `__version__` in `sys.modules`."""

    def _make(version):
        name = "_netket_fake_optional_dep"
        module = ModuleType(name)
        if version is not None:
            module.__version__ = version
        monkeypatch.setitem(sys.modules, name, module)
        return name, module

    return _make


def test_missing_module():
    with pytest.raises(ModuleNotFoundError, match="pip install '_netket_missing_dep"):
        import_optional_dependency("_netket_missing_dep", descr="something")


def test_missing_module_version_spec():
    with pytest.raises(ModuleNotFoundError, match=r">=1\.2,<2\.0"):
        import_optional_dependency(
            "_netket_missing_dep",
            minimum_version="1.2",
            maximum_version="2.0",
            descr="something",
        )


@pytest.mark.parametrize("version", ["1.0.0", "1.2.3", "1.9.0", "2.0.0"])
def test_supported_version(fake_module, version):
    name, module = fake_module(version)
    assert (
        import_optional_dependency(
            name, minimum_version="1.0.0", maximum_version="3.0.0", descr="something"
        )
        is module
    )


def test_version_too_old(fake_module):
    name, _ = fake_module("0.9.0")
    with pytest.raises(ImportError, match="0.9.0"):
        import_optional_dependency(
            name, minimum_version="1.0.0", descr="something", extra_msg="extra hint"
        )


def test_version_too_new(fake_module):
    name, _ = fake_module("1.0.0")
    with pytest.raises(ImportError, match="extra hint"):
        import_optional_dependency(
            name, maximum_version="1.0.0", descr="something", extra_msg="extra hint"
        )


def test_no_version_attribute_is_accepted(fake_module):
    """If the version cannot be determined, no check can be performed."""
    name, module = fake_module(None)
    assert (
        import_optional_dependency(
            name, minimum_version="1.0.0", maximum_version="2.0.0", descr="something"
        )
        is module
    )
