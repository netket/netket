# Copyright 2022 The NetKet Authors - All rights reserved.
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

from typing import Any
from collections.abc import Iterable


import rich


def __repr_from_rich__(self):
    """
    default __repr__ that calls __rich__
    """
    console = rich.get_console()
    with console.capture() as capture:
        console.print(self, end="")
    return capture.get()


def _repr_mimebundle_from_rich_(
    self, include: Iterable[str], exclude: Iterable[str], **kwargs: Any
) -> dict[str, str]:
    from rich.jupyter import _render_segments

    console = rich.get_console()
    segments = list(console.render(self, console.options))  # type: ignore
    html = _render_segments(segments)
    text = console._render_buffer(segments)
    data = {"text/plain": text, "text/html": html}
    if include:
        data = {k: v for (k, v) in data.items() if k in include}
    if exclude:
        data = {k: v for (k, v) in data.items() if k not in exclude}
    return data


def rich_repr(clz):
    """
    Class decorator setting the repr method to use
    `rich`.
    """
    setattr(clz, "__repr__", __repr_from_rich__)
    setattr(clz, "_repr_mimebundle_", _repr_mimebundle_from_rich_)
    return clz
